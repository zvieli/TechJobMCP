"""Tests for structured observability logging."""

import json
import logging
import uuid

import pytest

from job_mcp.utils.logger import get_logger, generate_trace_id
from job_mcp.models.schemas import ToolResponse


class TestStructuredLogger:
    def test_get_logger_returns_bound_logger(self):
        log = get_logger("test.observability")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "bind")

    def test_logger_outputs_json(self, capfd):
        log = get_logger("test.json_output")
        log.info("test_event", key="value")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        assert lines, "No output captured on stderr"
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "test_event"
        assert parsed["key"] == "value"
        assert "timestamp" in parsed

    def test_logger_sanitizes_secrets(self, capfd):
        log = get_logger("test.sanitize")
        log.info("auth_check", token="Bearer sk-abc123xyz789secret")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "sk-abc123xyz789secret" not in line
        assert "[REDACTED]" in line

    def test_logger_sanitizes_cookies(self, capfd):
        log = get_logger("test.sanitize_cookie")
        log.info("request", cookie="session=abc123;auth=xyz789")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "abc123" not in line
        assert "[REDACTED]" in line

    def test_logger_sanitizes_emails(self, capfd):
        log = get_logger("test.sanitize_email")
        log.info("user_login", email="user@example.com")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "user@example.com" not in line
        assert "[EMAIL_REDACTED]" in line

    def test_log_level_from_env(self, monkeypatch, capfd):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        # Force re-creation of logger with new level
        log = get_logger("test.level_check_" + uuid.uuid4().hex[:6])
        log.info("should_not_appear")
        log.warning("should_appear", data="visible")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        # Only warning should appear
        assert any("should_appear" in l for l in lines)
        assert not any("should_not_appear" in l for l in lines)

    def test_logger_positional_args(self, capfd):
        log = get_logger("test.positional")
        log.info("Hello %s, number %d", "world", 123)
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "Hello world, number 123"

    def test_logger_exception_stack_trace(self, capfd):
        log = get_logger("test.exception")
        try:
            raise ValueError("test exception for logger")
        except ValueError:
            log.exception("an error occurred")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "an error occurred"
        assert "exception" in parsed
        assert "ValueError: test exception for logger" in parsed["exception"]


class TestGenerateTraceId:
    def test_returns_hex_string(self):
        tid = generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 8
        int(tid, 16)  # Should not raise

    def test_uniqueness(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestToolResponseTraceId:
    def test_trace_id_optional_default_none(self):
        resp = ToolResponse(success=True, message="ok")
        assert resp.trace_id is None

    def test_trace_id_set(self):
        resp = ToolResponse(success=True, message="ok", trace_id="abcd1234")
        assert resp.trace_id == "abcd1234"

    def test_trace_id_in_dump(self):
        resp = ToolResponse(success=True, message="ok", trace_id="ef567890")
        d = resp.model_dump()
        assert d["trace_id"] == "ef567890"


class TestSourceHttpTelemetry:
    """Unit tests for outbound HTTP request telemetry across job sources."""

    @pytest.mark.asyncio
    async def test_hiremetech_api_fetch_telemetry(self, capfd):
        from unittest.mock import AsyncMock
        from job_mcp.core.api_client import fetch_jobs_via_api

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jobs": [
                {"id": "1", "title": "Python Dev", "company_name": "Acme"},
                {"id": "2", "title": "Frontend Dev", "company_name": "Beta"},
            ]
        })
        mock_request = AsyncMock()
        mock_request.get = AsyncMock(return_value=mock_response)

        jobs = await fetch_jobs_via_api(mock_request, page=1, size=10)
        assert len(jobs) == 2

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        events = [l for l in lines if l.get("event") == "HTTP API request completed"]
        assert len(events) >= 1
        event = events[-1]
        assert event["source"] == "hiremetech"
        assert event["status"] == 200
        assert event["jobs_count"] == 2
        assert isinstance(event["duration_ms"], (int, float))
        assert "url" in event

    @pytest.mark.asyncio
    async def test_comeet_http_ats_telemetry(self, capfd):
        from unittest.mock import AsyncMock, MagicMock
        import httpx
        from job_mcp.sources.comeet import ComeetSource, ComeetCompany

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"position_uid": "p1", "name": "DevOps Engineer", "location": "Tel Aviv"}
        ]
        mock_client.get.return_value = mock_resp

        company = ComeetCompany(uid="c1", name="TestCo", token="tok123")
        source = ComeetSource(companies=[company], client=mock_client)

        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 1

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        events = [l for l in lines if l.get("event") == "HTTP ATS request completed"]
        assert len(events) >= 1
        event = events[-1]
        assert event["source"] == "comeet"
        assert event["status"] == 200
        assert event["company"] == "TestCo"
        assert event["positions_count"] == 1
        assert isinstance(event["duration_ms"], (int, float))
        assert "url" in event

    @pytest.mark.asyncio
    async def test_alljobs_http_feed_telemetry(self, capfd):
        from unittest.mock import AsyncMock, MagicMock
        import httpx
        from job_mcp.sources.alljobs import AllJobsSource

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        cat_resp = MagicMock(spec=httpx.Response)
        cat_resp.status_code = 200
        cat_resp.url = "https://www.alljobs.co.il/SearchResultsMobile.ashx?action=getSearchEngineData"
        cat_resp.json.return_value = {"Categories": [{"CategoryID": 235, "CategoryName": "Software"}]}

        feed_resp = MagicMock(spec=httpx.Response)
        feed_resp.status_code = 200
        feed_resp.url = "https://www.alljobs.co.il/SearchResultsMobile.ashx?action=getJobs"
        feed_resp.json.return_value = {
            "Jobs": [
                {"JobID": 101, "JobTitle": "Backend", "CompanyName": "Alpha"}
            ]
        }

        async def _mock_get(url, **kwargs):
            if "categories" in kwargs.get("params", {}):
                return cat_resp
            return feed_resp

        mock_client.get.side_effect = _mock_get
        source = AllJobsSource(client=mock_client)

        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 1

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        feed_events = [l for l in lines if l.get("event") == "HTTP feed request completed"]
        assert len(feed_events) >= 1
        for ev in feed_events:
            assert ev["source"] == "alljobs"
            assert ev["status"] == 200
            assert "items_count" in ev
            assert isinstance(ev["duration_ms"], (int, float))
            assert "url" in ev

    @pytest.mark.asyncio
    async def test_aggregator_source_fetch_telemetry(self, capfd):
        from job_mcp.models.schemas import Job
        from job_mcp.sources.base import BaseJobSource
        from job_mcp.sources import SourceRegistry
        from job_mcp.sources.aggregator import JobAggregator

        class MockSource(BaseJobSource):
            source_id = "test_src"
            display_name = "Test Source"

            async def fetch_jobs(self, preferences=None, limit=50):
                return [Job(job_id="s1", title="Engineer", company="Company")]

            async def check_health(self):
                return True

        registry = SourceRegistry()
        registry.register(MockSource())
        aggregator = JobAggregator(registry=registry)

        jobs = await aggregator.fetch_all_jobs()
        assert len(jobs) == 1

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        agg_events = [l for l in lines if l.get("event") == "Source fetch completed"]
        assert len(agg_events) >= 1
        event = agg_events[-1]
        assert event["source_id"] == "test_src"
        assert event["jobs_count"] == 1
        assert isinstance(event["duration_ms"], (int, float))


class TestJobActionObservability:
    """Unit tests verifying structured event logging for bookmark_job and delete_job."""

    @pytest.mark.asyncio
    async def test_bookmark_job_structured_logging_cached(self, capfd):
        from unittest.mock import MagicMock
        from fastmcp import Context
        from job_mcp.core.api_client import JobCache
        from job_mcp.main import bookmark_job
        from job_mcp.models.schemas import Job

        cache = JobCache(ttl_minutes=10)
        test_job = Job(
            job_id="comeet_123",
            title="Senior Backend Dev",
            company="Acme Corp",
            source="comeet",
        )
        cache.update([test_job])

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await bookmark_job(job_id="comeet_123", ctx=ctx)
        assert res["success"] is True

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        bm_events = [l for l in lines if l.get("event") == "Bookmarked job"]
        assert len(bm_events) >= 1
        event = bm_events[-1]
        assert event["job_id"] == "comeet_123"
        assert event["title"] == "Senior Backend Dev"
        assert event["company"] == "Acme Corp"
        assert event["source"] == "comeet"

    @pytest.mark.asyncio
    async def test_bookmark_job_structured_logging_uncached(self, capfd):
        from unittest.mock import MagicMock
        from fastmcp import Context
        from job_mcp.core.api_client import JobCache
        from job_mcp.main import bookmark_job

        cache = JobCache(ttl_minutes=10)
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await bookmark_job(job_id="comeet_999", ctx=ctx)
        assert res["success"] is True

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        bm_events = [l for l in lines if l.get("event") == "Bookmarked job"]
        assert len(bm_events) >= 1
        event = bm_events[-1]
        assert event["job_id"] == "comeet_999"
        assert event["title"] is None
        assert event["company"] is None
        assert event["source"] is None

    @pytest.mark.asyncio
    async def test_delete_job_structured_logging_and_dismiss_cached(self, capfd):
        from unittest.mock import MagicMock
        from fastmcp import Context
        from job_mcp.core.api_client import JobCache
        from job_mcp.main import delete_job
        from job_mcp.models.schemas import Job

        cache = JobCache(ttl_minutes=10)
        test_job = Job(
            job_id="comeet_456",
            title="DevOps Lead",
            company="Beta Inc",
            source="comeet",
        )
        cache.update([test_job])
        assert cache.get_by_id("comeet_456") is not None

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await delete_job(job_id="comeet_456", ctx=ctx)
        assert res["success"] is True

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        del_events = [l for l in lines if l.get("event") == "Dismissed job from cache"]
        assert len(del_events) >= 1
        event = del_events[-1]
        assert event["job_id"] == "comeet_456"
        assert event["title"] == "DevOps Lead"
        assert event["company"] == "Beta Inc"
        assert event["source"] == "comeet"

        # Verify job was dismissed in cache
        assert cache.get_by_id("comeet_456") is None
        assert "comeet_456" in cache.dismissed_ids

    @pytest.mark.asyncio
    async def test_delete_job_structured_logging_uncached(self, capfd):
        from unittest.mock import MagicMock
        from fastmcp import Context
        from job_mcp.core.api_client import JobCache
        from job_mcp.main import delete_job

        cache = JobCache(ttl_minutes=10)
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await delete_job(job_id="comeet_999", ctx=ctx)
        assert res["success"] is True

        captured = capfd.readouterr()
        lines = [json.loads(l) for l in captured.err.strip().split("\n") if l.strip()]
        del_events = [l for l in lines if l.get("event") == "Dismissed job from cache"]
        assert len(del_events) >= 1
        event = del_events[-1]
        assert event["job_id"] == "comeet_999"
        assert event["title"] is None
        assert event["company"] is None
        assert event["source"] is None

        assert "comeet_999" in cache.dismissed_ids


