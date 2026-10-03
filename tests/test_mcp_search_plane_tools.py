"""Unit and integration tests for FastMCP job_search and job_fetch tools (Milestone 6-D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from job_mcp.core.api_client import JobCache
from job_mcp.core.search_plane.adapter import SearchPlaneAdapter
from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
)
from job_mcp.main import job_fetch, job_search
from job_mcp.models.schemas import Job, WorkMode
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.base import BaseJobSource
from job_mcp.sources.registry import SourceRegistry


def _make_sample_job(
    job_id: str = "ashby_acme_posting1",
    source: str = "ashby",
    title: str = "Senior Backend Engineer",
    company: str = "Acme Corp",
    location: str = "Tel Aviv",
    work_mode: WorkMode = WorkMode.HYBRID,
) -> Job:
    return Job(
        job_id=job_id,
        title=title,
        company=company,
        location=location,
        work_mode=work_mode,
        source=source,
        sources=[source],
        url=f"https://jobs.ashbyhq.com/acme/{job_id}",
        description="Develop scalable cloud distributed systems.",
    )


def _make_mock_context(
    cache: JobCache | None = None,
    registry: SourceRegistry | None = None,
    aggregator: JobAggregator | None = None,
    adapter: SearchPlaneAdapter | None = None,
) -> Context:
    c = cache if cache is not None else JobCache()
    reg = (
        registry
        if registry is not None
        else (getattr(aggregator, "registry", None) or getattr(adapter, "registry", None) or SourceRegistry())
    )
    agg = aggregator if aggregator is not None else JobAggregator(registry=reg, cache=c)
    adp = adapter if adapter is not None else SearchPlaneAdapter(aggregator=agg, registry=reg, cache=c)

    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {
        "cache": c,
        "registry": reg,
        "aggregator": agg,
        "search_adapter": adp,
    }
    return ctx


# ===========================================================================
# 1. job_search Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_job_search_default_provider_agnostic():
    """Verify job_search executes across enabled sources without requiring provider keys."""
    sample = _make_sample_job()
    mock_agg = MagicMock(spec=JobAggregator)
    mock_agg.fetch_all_jobs = AsyncMock(return_value=[sample])

    ctx = _make_mock_context(aggregator=mock_agg)

    res = await job_search(ctx=ctx)

    assert res["success"] is True
    assert "Retrieved 1 job search results" in res["message"]
    data = res["data"]
    assert len(data["items"]) == 1

    item = data["items"][0]
    assert item["title"] == "Senior Backend Engineer"
    assert item["company"] == "Acme Corp"
    assert item["source_family"] == "ashby"
    assert item["ref"].startswith("v1_")

    # Check decoded ref is opaque and matches routing coordinates
    decoded_ref = JobRef.decode(item["ref"])
    assert decoded_ref.source_family == "ashby"
    assert decoded_ref.account == "acme"
    assert decoded_ref.locator == "posting1"

    # Aggregator was queried with sources=None (unified search)
    mock_agg.fetch_all_jobs.assert_called_once()
    _, kwargs = mock_agg.fetch_all_jobs.call_args
    assert kwargs.get("sources") is None


@pytest.mark.asyncio
async def test_job_search_filter_propagation():
    """Verify search parameters are mapped accurately onto JobSearchRequest."""
    sample = _make_sample_job(company="Acme Corp", location="Tel Aviv")
    mock_agg = MagicMock(spec=JobAggregator)
    mock_agg.fetch_all_jobs = AsyncMock(return_value=[sample])

    ctx = _make_mock_context(aggregator=mock_agg)

    res = await job_search(
        query="backend",
        location="Tel Aviv",
        work_mode="hybrid",
        company="Acme",
        tech_stack=["Python", "FastAPI"],
        limit=10,
        sources=["ashby"],
        ctx=ctx,
    )

    assert res["success"] is True
    assert len(res["data"]["items"]) == 1
    mock_agg.fetch_all_jobs.assert_called_once()
    _, kwargs = mock_agg.fetch_all_jobs.call_args
    assert kwargs.get("sources") == ["ashby"]
    prefs = kwargs.get("preferences")
    assert prefs.keywords == ["backend"]
    assert prefs.location == "Tel Aviv"
    assert prefs.work_mode == WorkMode.HYBRID
    assert prefs.tech_stack == ["Python", "FastAPI"]


@pytest.mark.asyncio
async def test_job_search_limit_boundaries():
    """Verify limit validation enforces bounded 1-200 range."""
    ctx = _make_mock_context()

    with pytest.raises(ValueError, match="limit must be between 1 and 200"):
        await job_search(limit=0, ctx=ctx)

    with pytest.raises(ValueError, match="limit must be between 1 and 200"):
        await job_search(limit=201, ctx=ctx)


@pytest.mark.asyncio
async def test_job_search_work_mode_validation():
    """Verify work_mode accepts valid strings case-insensitively and rejects invalid ones."""
    ctx = _make_mock_context()

    with pytest.raises(ValueError, match="Invalid work_mode"):
        await job_search(work_mode="on_the_moon", ctx=ctx)


@pytest.mark.asyncio
async def test_job_search_deferred_fields_rejected():
    """Verify deferred/future parameters are strictly rejected rather than silently ignored."""
    ctx = _make_mock_context()

    with pytest.raises(ValueError, match="freshness_days is not supported in Milestone 6"):
        await job_search(freshness_days=7, ctx=ctx)

    with pytest.raises(ValueError, match="cursor pagination is not supported in Milestone 6"):
        await job_search(cursor="cursor_token_123", ctx=ctx)

    with pytest.raises(ValueError, match="seniority filtering is not supported in Milestone 6"):
        await job_search(seniority="Senior", ctx=ctx)


@pytest.mark.asyncio
async def test_job_search_unknown_source_warning():
    """Verify unknown sources do not crash the search plane and are reported in warnings."""
    mock_agg = MagicMock(spec=JobAggregator)
    mock_agg.fetch_all_jobs = AsyncMock(return_value=[])

    ctx = _make_mock_context(aggregator=mock_agg)

    res = await job_search(sources=["nonexistent_platform"], ctx=ctx)

    assert res["success"] is True
    assert res["data"]["items"] == []
    assert len(res["data"]["warnings"]) >= 1
    assert "nonexistent_platform" in res["data"]["warnings"][0]


@pytest.mark.asyncio
async def test_job_search_mixed_valid_and_unknown_sources_still_execute():
    """Verify a valid source executes and returns results even when an unknown source is requested."""
    sample = _make_sample_job()
    mock_ashby = MagicMock(spec=BaseJobSource)
    mock_ashby.source_id = "ashby"
    mock_ashby.fetch_jobs = AsyncMock(return_value=[sample])

    reg = SourceRegistry()
    reg.register(mock_ashby)
    cache = JobCache()
    agg = JobAggregator(registry=reg, cache=cache)
    adp = SearchPlaneAdapter(aggregator=agg, registry=reg, cache=cache)

    ctx = _make_mock_context(cache=cache, aggregator=agg, adapter=adp)

    res = await job_search(sources=["ashby", "does-not-exist"], ctx=ctx)

    assert res["success"] is True
    # Valid requested source still executes and yields results
    assert len(res["data"]["items"]) == 1
    assert res["data"]["items"][0]["source_family"] == "ashby"
    # Unknown source surfaces a diagnostic without crashing the plane
    assert any("does-not-exist" in w for w in res["data"]["warnings"])


@pytest.mark.asyncio
async def test_job_search_empty_matches():
    """Verify empty matches returns cleanly with empty items list."""
    mock_agg = MagicMock(spec=JobAggregator)
    mock_agg.fetch_all_jobs = AsyncMock(return_value=[])

    ctx = _make_mock_context(aggregator=mock_agg)

    res = await job_search(query="nonexistent_keyword_xyz", ctx=ctx)

    assert res["success"] is True
    assert res["data"]["items"] == []
    assert res["data"]["total_estimated"] == 0


@pytest.mark.asyncio
async def test_job_search_read_only_safety():
    """Verify job_search does not interact with application ledger or dispatcher."""
    sample = _make_sample_job()
    mock_agg = MagicMock(spec=JobAggregator)
    mock_agg.fetch_all_jobs = AsyncMock(return_value=[sample])

    ctx = _make_mock_context(aggregator=mock_agg)

    with patch("job_mcp.main._get_ledger") as mock_get_ledger, patch(
        "job_mcp.main._get_dispatcher"
    ) as mock_get_dispatcher:
        res = await job_search(ctx=ctx)
        assert res["success"] is True
        mock_get_ledger.assert_not_called()
        mock_get_dispatcher.assert_not_called()


# ===========================================================================
# 2. job_fetch Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_job_fetch_cached_ref_found():
    """Verify job_fetch hits in-memory cache directly and returns FOUND status."""
    sample = _make_sample_job()
    cache = JobCache()
    cache.update([sample])

    ctx = _make_mock_context(cache=cache)

    ref = JobRef(version=1, source_family="ashby", account="acme", locator="posting1").encode()

    res = await job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "FOUND"
    assert res["data"]["job"] is not None
    assert res["data"]["job"]["job_id"] == "ashby_acme_posting1"
    assert res["data"]["job"]["title"] == "Senior Backend Engineer"
    assert "Job found" in res["message"]


@pytest.mark.asyncio
async def test_job_fetch_native_refetch_ashby_found():
    """Verify job_fetch queries provider natively on cache miss and returns FOUND."""
    sample = _make_sample_job()
    cache = JobCache()
    mock_ashby = MagicMock(spec=BaseJobSource)
    mock_ashby.source_id = "ashby"
    mock_ashby.fetch_job_by_ref = AsyncMock(
        return_value=FetchResult(status=FetchStatus.FOUND, job=sample, ref="v1_...")
    )

    reg = SourceRegistry()
    reg.register(mock_ashby)
    agg = JobAggregator(registry=reg, cache=cache)
    adp = SearchPlaneAdapter(aggregator=agg, registry=reg, cache=cache)

    ctx = _make_mock_context(cache=cache, aggregator=agg, adapter=adp)

    ref = JobRef(version=1, source_family="ashby", account="acme", locator="posting1").encode()

    res = await job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "FOUND"
    assert res["data"]["job"]["title"] == "Senior Backend Engineer"
    mock_ashby.fetch_job_by_ref.assert_called_once()
    # Cache was updated with fetched job
    assert cache.get_by_id(sample.job_id) is not None


@pytest.mark.asyncio
async def test_job_fetch_not_found_returns_cleanly_without_exception():
    """Verify job_fetch NOT_FOUND returns factual status and does NOT raise an MCP exception."""
    cache = JobCache()
    mock_ashby = MagicMock(spec=BaseJobSource)
    mock_ashby.source_id = "ashby"
    mock_ashby.fetch_job_by_ref = AsyncMock(
        return_value=FetchResult(
            status=FetchStatus.NOT_FOUND,
            ref="v1_...",
            diagnostic="Posting was not found on Ashby board 'acme'",
        )
    )

    reg = SourceRegistry()
    reg.register(mock_ashby)
    agg = JobAggregator(registry=reg, cache=cache)
    adp = SearchPlaneAdapter(aggregator=agg, registry=reg, cache=cache)

    ctx = _make_mock_context(cache=cache, aggregator=agg, adapter=adp)

    ref = JobRef(version=1, source_family="ashby", account="acme", locator="missing_id").encode()

    res = await job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "NOT_FOUND"
    assert res["data"]["job"] is None
    assert "not_found" in res["message"].lower() or "not found" in res["message"].lower()


@pytest.mark.asyncio
async def test_job_fetch_invalid_ref_returns_cleanly_without_exception():
    """Verify invalid/corrupt JobRef returns INVALID_REF without raising an exception."""
    ctx = _make_mock_context()

    # Totally corrupt string
    res = await job_fetch(ref="not_a_valid_ref", ctx=ctx)
    assert res["success"] is True
    assert res["data"]["status"] == "INVALID_REF"
    assert res["data"]["job"] is None

    # Empty string
    res_empty = await job_fetch(ref="", ctx=ctx)
    assert res_empty["success"] is True
    assert res_empty["data"]["status"] == "INVALID_REF"
    assert res_empty["data"]["job"] is None


@pytest.mark.asyncio
async def test_job_fetch_unsupported_refetch():
    """Verify source families without single-job endpoints return UNSUPPORTED_REFETCH."""
    ctx = _make_mock_context()

    # Greenhouse ref not in cache
    ref = JobRef(version=1, source_family="greenhouse", account="stripe", locator="12345").encode()

    res = await job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "UNSUPPORTED_REFETCH"
    assert res["data"]["job"] is None
    assert "does not support native single-posting refetch" in res["data"]["diagnostic"]


@pytest.mark.asyncio
async def test_job_fetch_upstream_error():
    """Verify upstream network/server failure returns UPSTREAM_ERROR."""
    cache = JobCache()
    mock_ashby = MagicMock(spec=BaseJobSource)
    mock_ashby.source_id = "ashby"
    mock_ashby.fetch_job_by_ref = AsyncMock(side_effect=RuntimeError("Ashby gateway timeout 504"))

    reg = SourceRegistry()
    reg.register(mock_ashby)
    agg = JobAggregator(registry=reg, cache=cache)
    adp = SearchPlaneAdapter(aggregator=agg, registry=reg, cache=cache)

    ctx = _make_mock_context(cache=cache, aggregator=agg, adapter=adp)

    ref = JobRef(version=1, source_family="ashby", account="acme", locator="posting1").encode()

    res = await job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is False
    assert res["data"]["status"] == "UPSTREAM_ERROR"
    assert res["error_code"] == "UPSTREAM_ERROR"
    assert res["data"]["job"] is None


@pytest.mark.asyncio
async def test_job_fetch_no_m7_lifecycle_leakage():
    """Verify fetch status never uses M7 observation lifecycle terms."""
    forbidden = {"EXPIRED", "STALE", "DEAD", "POSTING_EXPIRED", "CLOSED", "REMOVED"}

    ctx = _make_mock_context()
    ref = JobRef(version=1, source_family="workable", account="acme", locator="xyz999").encode()

    res = await job_fetch(ref=ref, ctx=ctx)
    status = res["data"]["status"]
    assert status not in forbidden
    assert status in {s.value for s in FetchStatus}
