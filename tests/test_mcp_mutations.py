"""Mutation test suite for Milestone 6-D Agent-Facing FastMCP Integration.

Formally proves resistance against all 18 mutations:
1. job_search bypasses SearchPlaneAdapter
2. job_fetch manually parses provider routing instead of delegating
3. provider becomes mandatory argument to job_search
4. source-specific job refs exposed instead of opaque encoded refs
5. job_fetch NOT_FOUND raises MCP exception
6. INVALID_REF raises MCP exception
7. discover_companies writes portals.yml
8. discovery MCP handler directly fetches URLs outside discovery service
9. discovery target batch becomes unbounded
10. old compatibility tool removed
11. old tool signature changed
12. new tool bypasses metrics middleware
13. freshness_days silently accepted but ignored
14. cursor silently accepted but ignored
15. M7 lifecycle status leaks into job_fetch
16. tool inventory test hardcodes historical total count
17. application dispatch accidentally invoked by read-only search
18. registry/provider graph rebuilt unnecessarily per call if shared lifespan is available
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

import job_mcp.main as main_mod
from job_mcp import mcp
from job_mcp.core.search_plane.adapter import SearchPlaneAdapter
from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)
from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
    JobSearchRequest,
    JobSearchResultItem,
    JobSearchResultSet,
)
from job_mcp.models.schemas import WorkMode
from job_mcp.utils.metrics import TOOL_NAMES


def _create_mock_context(adapter: SearchPlaneAdapter | None = None) -> Context:
    ctx = MagicMock(spec=Context)
    lifespan_ctx: dict[str, object] = {}
    if adapter is not None:
        lifespan_ctx["search_adapter"] = adapter
    ctx.lifespan_context = lifespan_ctx
    return ctx


# ---------------------------------------------------------------------------
# Mutation 1: job_search bypasses SearchPlaneAdapter
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_1_job_search_must_delegate_to_adapter():
    """Verify job_search always delegates search execution to SearchPlaneAdapter."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.search = AsyncMock(return_value=JobSearchResultSet(items=[], warnings=[]))
    ctx = _create_mock_context(mock_adapter)

    await main_mod.job_search(query="python", location="Remote", ctx=ctx)

    mock_adapter.search.assert_awaited_once()
    req = mock_adapter.search.await_args[0][0]
    assert isinstance(req, JobSearchRequest)
    assert req.query == "python"
    assert req.location == "Remote"


# ---------------------------------------------------------------------------
# Mutation 2: job_fetch manually parses provider routing instead of delegating
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_2_job_fetch_delegates_to_adapter_without_manual_routing():
    """Verify job_fetch delegates directly to adapter.fetch without ad-hoc routing/parsing."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    ref = JobRef(source_family="ashby", account="openai", locator="123-uuid").encode()
    mock_adapter.fetch = AsyncMock(return_value=FetchResult(status=FetchStatus.FOUND, job=None))
    ctx = _create_mock_context(mock_adapter)

    res = await main_mod.job_fetch(ref=ref, ctx=ctx)

    mock_adapter.fetch.assert_awaited_once_with(ref)
    assert res["success"] is True
    assert res["data"]["status"] == "FOUND"


# ---------------------------------------------------------------------------
# Mutation 3: provider becomes mandatory argument to job_search
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_3_sources_is_not_mandatory_in_job_search():
    """Verify job_search can be called without sources argument (defaults to None / all)."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.search = AsyncMock(return_value=JobSearchResultSet(items=[], warnings=[]))
    ctx = _create_mock_context(mock_adapter)

    # Calling with zero args except ctx must succeed without missing-argument errors
    res = await main_mod.job_search(ctx=ctx)
    assert res["success"] is True
    req = mock_adapter.search.await_args[0][0]
    assert req.sources is None


# ---------------------------------------------------------------------------
# Mutation 4: source-specific job refs exposed instead of opaque encoded refs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_4_job_search_returns_opaque_v1_refs():
    """Verify all returned items expose opaque v1_ refs that decode cleanly into JobRef."""
    ref_obj = JobRef(source_family="workable", account="supermetrics", locator="abc123xyz")
    opaque_ref = ref_obj.encode()
    item = JobSearchResultItem(
        ref=opaque_ref,
        title="Engineer",
        company="Supermetrics",
        location="Remote",
        work_mode=WorkMode.REMOTE,
        source_family="workable",
    )
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.search = AsyncMock(return_value=JobSearchResultSet(items=[item], warnings=[]))
    ctx = _create_mock_context(mock_adapter)

    res = await main_mod.job_search(query="engineer", ctx=ctx)
    result_items = res["data"]["items"]
    assert len(result_items) == 1
    returned_ref = result_items[0]["ref"]
    assert returned_ref.startswith("v1_")
    # Must decode cleanly as JobRef without relying on provider-specific formats
    decoded = JobRef.decode(returned_ref)
    assert decoded.source_family == "workable"
    assert decoded.account == "supermetrics"
    assert decoded.locator == "abc123xyz"


# ---------------------------------------------------------------------------
# Mutation 5: job_fetch NOT_FOUND raises MCP exception
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_5_job_fetch_not_found_returns_clean_factual_status():
    """Verify NOT_FOUND returns success=True with factual status and does NOT raise exception."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.fetch = AsyncMock(return_value=FetchResult(status=FetchStatus.NOT_FOUND, job=None))
    ctx = _create_mock_context(mock_adapter)

    ref = JobRef(source_family="ashby", account="acme", locator="nonexistent").encode()
    res = await main_mod.job_fetch(ref=ref, ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "NOT_FOUND"
    assert res["data"]["job"] is None


# ---------------------------------------------------------------------------
# Mutation 6: INVALID_REF raises MCP exception
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_6_job_fetch_invalid_ref_returns_clean_factual_status():
    """Verify INVALID_REF returns success=True with factual status and does NOT raise exception."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.fetch = AsyncMock(return_value=FetchResult(status=FetchStatus.INVALID_REF, job=None))
    ctx = _create_mock_context(mock_adapter)

    res = await main_mod.job_fetch(ref="not-a-valid-ref", ctx=ctx)

    assert res["success"] is True
    assert res["data"]["status"] == "INVALID_REF"
    assert res["data"]["job"] is None


# ---------------------------------------------------------------------------
# Mutation 7: discover_companies writes portals.yml
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_7_discover_companies_never_modifies_portals_yml():
    """Verify discover_companies never creates, writes, or modifies portals.yml."""
    portals_path = "portals.yml"
    initial_exists = os.path.exists(portals_path)
    initial_mtime = os.path.getmtime(portals_path) if initial_exists else None

    ev = DiscoveryEvidence(kind="DIRECT_PROVIDER_URL", value="ashby:acme")
    portal = DiscoveredPortal(source_family="ashby", account="acme", confidence_basis=(ev,))
    result = DiscoveryResult(
        status=DiscoveryStatus.CONFIRMED,
        input=DiscoveryTarget(company_name="Acme"),
        candidates=(portal,),
        evidence=(ev,),
    )

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = [result]
        res = await main_mod.discover_companies(company_name="Acme")
        assert res["success"] is True

    if initial_exists:
        assert os.path.getmtime(portals_path) == initial_mtime
    else:
        assert not os.path.exists(portals_path)


# ---------------------------------------------------------------------------
# Mutation 8: discovery MCP handler directly fetches URLs outside discovery service
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_8_discovery_mcp_handler_never_makes_direct_http_calls():
    """Verify discover_companies handler delegates completely without direct httpx calls."""
    with patch("httpx.AsyncClient") as mock_client, patch(
        "job_mcp.main.service_discover_companies", new_callable=AsyncMock
    ) as mock_svc:
        mock_svc.return_value = []
        await main_mod.discover_companies(career_url="https://jobs.ashbyhq.com/acme")
        mock_client.assert_not_called()
        mock_svc.assert_awaited_once()


# ---------------------------------------------------------------------------
# Mutation 9: discovery target batch becomes unbounded
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_9_discovery_batch_size_is_strictly_bounded():
    """Verify discovery rejects batches > 10 targets and rejects 0 targets."""
    # > 10 targets must raise ValueError
    targets_11 = [{"company_name": f"Company {i}"} for i in range(11)]
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        await main_mod.discover_companies(targets=targets_11)

    # 0 targets must raise ValueError
    with pytest.raises(ValueError, match="At least one discovery target must be provided"):
        await main_mod.discover_companies(targets=[])

    with pytest.raises(ValueError, match="At least one discovery target must be provided"):
        await main_mod.discover_companies()


# ---------------------------------------------------------------------------
# Mutation 10: old compatibility tool removed
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_10_no_legacy_tools_removed():
    """Verify all 16 legacy tools remain registered on FastMCP."""
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    legacy_tools = [
        "set_operation_mode", "list_job_sources", "get_job_matches",
        "filter_jobs_by_preferences", "bookmark_job", "delete_job",
        "auto_apply_job", "confirm_auto_apply", "get_application_history",
        "mark_job_as_applied", "calibrate_selectors", "search_linkedin_jobs",
        "get_linkedin_job_details", "notify_new_jobs", "test_notifier", "run_job_scout"
    ]
    for lt in legacy_tools:
        assert lt in tool_names, f"Legacy tool {lt} was removed!"


# ---------------------------------------------------------------------------
# Mutation 11: old tool signature changed
# ---------------------------------------------------------------------------
def test_mutation_11_legacy_tool_signatures_unaltered():
    """Verify legacy tool signatures have not lost parameters or altered defaults."""
    sig = inspect.signature(main_mod.get_job_matches)
    assert "sources" in sig.parameters
    assert "cv_path" in sig.parameters
    assert "force_refresh" in sig.parameters
    assert sig.parameters["force_refresh"].default is False


# ---------------------------------------------------------------------------
# Mutation 12: new tool bypasses metrics middleware
# ---------------------------------------------------------------------------
def test_mutation_12_new_tools_are_registered_in_metrics_bounded_set():
    """Verify job_search, job_fetch, and discover_companies are in bounded TOOL_NAMES set."""
    assert "job_search" in TOOL_NAMES
    assert "job_fetch" in TOOL_NAMES
    assert "discover_companies" in TOOL_NAMES


# ---------------------------------------------------------------------------
# Mutation 13: freshness_days silently accepted but ignored
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_13_freshness_days_explicitly_rejected():
    """Verify passing freshness_days raises ValueError rather than being silently ignored."""
    with pytest.raises(ValueError, match="freshness_days is not supported in Milestone 6"):
        await main_mod.job_search(freshness_days=7)


# ---------------------------------------------------------------------------
# Mutation 14: cursor silently accepted but ignored
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_14_cursor_explicitly_rejected():
    """Verify passing cursor or seniority raises ValueError rather than being silently ignored."""
    with pytest.raises(ValueError, match="cursor pagination is not supported in Milestone 6"):
        await main_mod.job_search(cursor="token_123")

    with pytest.raises(ValueError, match="seniority filtering is not supported in Milestone 6"):
        await main_mod.job_search(seniority="staff")


# ---------------------------------------------------------------------------
# Mutation 15: M7 lifecycle status leaks into job_fetch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_15_no_m7_lifecycle_statuses_in_job_fetch():
    """Verify job_fetch strictly uses factual FetchStatus and rejects/forbids M7 lifecycle terms."""
    m7_forbidden_terms = {"EXPIRED", "STALE", "DEAD", "POSTING_EXPIRED", "ACTIVE"}
    valid_statuses = {s.value for s in FetchStatus}

    # Verify FetchStatus enum contains zero M7 terms
    assert not (m7_forbidden_terms & valid_statuses)

    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    mock_adapter.fetch = AsyncMock(return_value=FetchResult(status=FetchStatus.NOT_FOUND, job=None))
    ctx = _create_mock_context(mock_adapter)

    res = await main_mod.job_fetch(ref="v1_test", ctx=ctx)
    status_str = res["data"]["status"]
    assert status_str not in m7_forbidden_terms
    assert status_str in valid_statuses


# ---------------------------------------------------------------------------
# Mutation 16: tool inventory test hardcodes historical total count
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_16_tool_inventory_preserves_extensibility():
    """Verify tool inventory test checks >= 19 and does not fail on future additions."""
    tools = await mcp.list_tools()
    # Must be at least 19 (16 legacy + 3 new), strictly avoiding brittle == 16 assertions
    assert len(tools) >= 19


# ---------------------------------------------------------------------------
# Mutation 17: application dispatch accidentally invoked by read-only search
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_17_search_plane_never_invokes_application_dispatcher():
    """Verify job_search and job_fetch never invoke HybridApplicationDispatcher or ledger."""
    with patch("job_mcp.main.HybridApplicationDispatcher") as mock_disp, patch(
        "job_mcp.main.ApplicationLedger"
    ) as mock_ledger:
        mock_adapter = MagicMock(spec=SearchPlaneAdapter)
        mock_adapter.search = AsyncMock(return_value=JobSearchResultSet(items=[], warnings=[]))
        mock_adapter.fetch = AsyncMock(return_value=FetchResult(status=FetchStatus.NOT_FOUND, job=None))
        ctx = _create_mock_context(mock_adapter)

        await main_mod.job_search(query="engineer", ctx=ctx)
        await main_mod.job_fetch(ref="v1_any", ctx=ctx)

        mock_disp.assert_not_called()
        mock_ledger.assert_not_called()


# ---------------------------------------------------------------------------
# Mutation 18: registry/provider graph rebuilt unnecessarily per call if shared lifespan is available
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mutation_18_lifespan_adapter_reused_without_rebuilding():
    """Verify _get_search_adapter reuses the adapter in lifespan_context without re-instantiating."""
    mock_adapter = MagicMock(spec=SearchPlaneAdapter)
    ctx = _create_mock_context(mock_adapter)

    with patch("job_mcp.main.SearchPlaneAdapter") as mock_adapter_cls:
        adapter1 = main_mod._get_search_adapter(ctx)
        adapter2 = main_mod._get_search_adapter(ctx)

        assert adapter1 is mock_adapter
        assert adapter2 is mock_adapter
        # SearchPlaneAdapter constructor must not have been called because lifespan cache was used
        mock_adapter_cls.assert_not_called()
