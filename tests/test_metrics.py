"""Focused observability tests that do not require external services."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.tools.base import ToolResult
from starlette.testclient import TestClient

import job_mcp.main as main_module
from job_mcp.core.system1.engine import LazyLayaEngine
from job_mcp.main import make_asgi_app, mcp
from job_mcp.models.schemas import Job
from job_mcp.utils.metrics import (
    ToolMetricsMiddleware,
    canonical_source_name,
    canonical_tool_name,
)


@pytest.mark.asyncio
async def test_tool_metrics_records_error_result_without_transforming_result():
    middleware = ToolMetricsMiddleware()
    result = ToolResult(structured_content={"success": False})
    context = SimpleNamespace(message=SimpleNamespace(name="run_job_scout"))

    async def next_handler(received):
        assert received is context
        return result

    assert await middleware.on_call_tool(context, next_handler) is result


@pytest.mark.asyncio
async def test_tool_metrics_reraises_exception_unchanged():
    middleware = ToolMetricsMiddleware()
    context = SimpleNamespace(message=SimpleNamespace(name="unregistered-tool"))
    error = RuntimeError("unchanged")

    async def next_handler(received):
        raise error

    with pytest.raises(RuntimeError) as raised:
        await middleware.on_call_tool(context, next_handler)
    assert raised.value is error
    assert canonical_tool_name("unregistered-tool") == "unknown"


@pytest.mark.asyncio
async def test_every_registered_tool_has_a_bounded_metric_label():
    tools = await mcp.list_tools()
    assert all(canonical_tool_name(tool.name) != "unknown" for tool in tools)


def test_metric_labels_and_job_provenance_are_bounded_and_private():
    assert canonical_source_name("comeet") == "comeet"
    assert canonical_source_name("candidate-controlled-source") == "other"
    job = Job(job_id="1", title="Engineer", company="Acme")
    job._system1_inference_origin = "real"
    assert "_system1_inference_origin" not in job.model_dump()


def test_engine_fallback_provenance_is_not_real_and_job_serialization_stays_private():
    engine = LazyLayaEngine(model_name="data/models/nonexistent", auto_release_after_batch=False)
    result = engine.predict_match_scoring_batch([
        {"job_title": "Engineer", "job_desc": "Python role", "cv_text": "Python developer"}
    ])[0]
    assert result["_system1_inference_origin"] == "fallback"

    job = Job(job_id="1", title="Engineer", company="Acme")
    job._system1_inference_origin = result["_system1_inference_origin"]
    assert "_system1_inference_origin" not in job.model_dump()


@pytest.mark.asyncio
async def test_already_applied_real_inference_is_not_a_triage_disqualification(monkeypatch):
    job = Job(job_id="already-applied", title="Engineer", company="Acme", match_score=10)
    job._system1_inference_origin = "real"
    cache = MagicMock()
    cache.get_all.return_value = []
    aggregator = MagicMock()
    aggregator.registry.list_sources.return_value = []
    aggregator.fetch_all_jobs = AsyncMock(return_value=[job])
    ledger = MagicMock()
    ledger.is_applied.return_value = True
    counter = MagicMock()
    labels = MagicMock()
    counter.labels.return_value = labels
    monkeypatch.setattr(main_module, "SYSTEM1_TRIAGE_TOTAL", counter)
    monkeypatch.setattr(main_module, "_get_cache", lambda ctx: cache)
    monkeypatch.setattr(main_module, "_get_aggregator", lambda ctx: aggregator)
    monkeypatch.setattr(main_module, "_get_ledger", lambda ctx: ledger)
    monkeypatch.setattr(main_module, "filter_jobs", lambda *args, **kwargs: [job])

    result = await main_module.run_job_scout(auto_bookmark=False)

    assert result["success"] is True
    labels.inc.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_inference_is_excluded_from_triage_metrics(monkeypatch):
    job = Job(job_id="fallback", title="Engineer", company="Acme", match_score=10)
    job._system1_inference_origin = "fallback"
    cache = MagicMock()
    cache.get_all.return_value = []
    aggregator = MagicMock()
    aggregator.registry.list_sources.return_value = []
    aggregator.fetch_all_jobs = AsyncMock(return_value=[job])
    ledger = MagicMock()
    ledger.is_applied.return_value = False
    counter = MagicMock()
    labels = MagicMock()
    counter.labels.return_value = labels
    monkeypatch.setattr(main_module, "SYSTEM1_TRIAGE_TOTAL", counter)
    monkeypatch.setattr(main_module, "_get_cache", lambda ctx: cache)
    monkeypatch.setattr(main_module, "_get_aggregator", lambda ctx: aggregator)
    monkeypatch.setattr(main_module, "_get_ledger", lambda ctx: ledger)
    monkeypatch.setattr(main_module, "filter_jobs", lambda *args, **kwargs: [job])

    result = await main_module.run_job_scout(auto_bookmark=False)

    assert result["success"] is True
    labels.inc.assert_not_called()


@pytest.mark.asyncio
async def test_uncategorized_score_gap_is_excluded_from_triage_metrics(monkeypatch):
    job = Job(job_id="score-gap", title="Engineer", company="Acme", match_score=60)
    job._system1_inference_origin = "real"
    cache = MagicMock()
    cache.get_all.return_value = []
    aggregator = MagicMock()
    aggregator.registry.list_sources.return_value = []
    aggregator.fetch_all_jobs = AsyncMock(return_value=[job])
    ledger = MagicMock()
    ledger.is_applied.return_value = False
    counter = MagicMock()
    labels = MagicMock()
    counter.labels.return_value = labels
    monkeypatch.setattr(main_module, "SYSTEM1_TRIAGE_TOTAL", counter)
    monkeypatch.setattr(main_module, "_get_cache", lambda ctx: cache)
    monkeypatch.setattr(main_module, "_get_aggregator", lambda ctx: aggregator)
    monkeypatch.setattr(main_module, "_get_ledger", lambda ctx: ledger)
    monkeypatch.setattr(main_module, "filter_jobs", lambda *args, **kwargs: [job])

    result = await main_module.run_job_scout(auto_bookmark=False)

    assert result["success"] is True
    labels.inc.assert_not_called()


def test_metrics_endpoint_is_mounted_and_bypasses_probe_handling():
    client = TestClient(make_asgi_app(), follow_redirects=False)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "access-control-allow-origin" not in response.headers
    assert "mcp_tool_execution_duration_seconds" in response.text
    trailing_response = client.get("/metrics/")
    assert trailing_response.status_code == 200
    assert "text/plain" in trailing_response.headers["content-type"]
    assert "mcp_tool_execution_duration_seconds" in trailing_response.text


@pytest.mark.asyncio
async def test_metrics_middleware_registered_once():
    assert sum(isinstance(item, ToolMetricsMiddleware) for item in mcp.middleware) == 1
