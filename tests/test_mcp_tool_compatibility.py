"""Compatibility tests for FastMCP tool surface (Milestone 6-D).

Verifies that all 16 legacy MCP tools remain registered with unchanged signatures,
that the 3 new M6-D tools (job_search, job_fetch, discover_companies) are properly
exposed, and that backward compatibility is fully preserved.
"""

from __future__ import annotations

import inspect

import pytest

import job_mcp.main as main_mod
from job_mcp import mcp

LEGACY_TOOL_NAMES = [
    "set_operation_mode",
    "list_job_sources",
    "get_job_matches",
    "filter_jobs_by_preferences",
    "bookmark_job",
    "delete_job",
    "auto_apply_job",
    "confirm_auto_apply",
    "get_application_history",
    "mark_job_as_applied",
    "calibrate_selectors",
    "search_linkedin_jobs",
    "get_linkedin_job_details",
    "notify_new_jobs",
    "test_notifier",
    "run_job_scout",
]

M6D_TOOL_NAMES = [
    "job_search",
    "job_fetch",
    "discover_companies",
]


@pytest.mark.asyncio
async def test_tool_inventory_contains_all_legacy_and_new_tools():
    """Verify that all 16 legacy tools and 3 new M6-D tools are registered on the FastMCP instance."""
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}

    assert len(tool_names) >= 19

    for name in LEGACY_TOOL_NAMES:
        assert name in tool_names, f"Legacy tool {name} is missing from FastMCP registry"

    for name in M6D_TOOL_NAMES:
        assert name in tool_names, f"New M6-D tool {name} is missing from FastMCP registry"


def test_legacy_tool_signatures_remain_unchanged():
    """Verify parameter names and signatures of legacy tools are preserved exactly."""
    # 1. list_job_sources
    sig = inspect.signature(main_mod.list_job_sources)
    assert "category" in sig.parameters

    # 2. get_job_matches
    sig = inspect.signature(main_mod.get_job_matches)
    for param in ["sources", "force_refresh", "cv_path", "tech_stack", "work_mode", "location"]:
        assert param in sig.parameters

    # 3. filter_jobs_by_preferences
    sig = inspect.signature(main_mod.filter_jobs_by_preferences)
    for param in ["tech_stack", "work_mode", "location", "min_salary", "cv_path", "limit"]:
        assert param in sig.parameters

    # 4. bookmark_job
    sig = inspect.signature(main_mod.bookmark_job)
    assert "job_id" in sig.parameters

    # 5. delete_job
    sig = inspect.signature(main_mod.delete_job)
    assert "job_id" in sig.parameters

    # 6. auto_apply_job
    sig = inspect.signature(main_mod.auto_apply_job)
    assert "job_id" in sig.parameters
    assert "cv_path" in sig.parameters

    # 7. confirm_auto_apply
    sig = inspect.signature(main_mod.confirm_auto_apply)
    assert "job_id" in sig.parameters
    assert "force" in sig.parameters

    # 8. get_application_history
    sig = inspect.signature(main_mod.get_application_history)
    assert "limit" in sig.parameters
    assert "status" in sig.parameters

    # 9. mark_job_as_applied
    sig = inspect.signature(main_mod.mark_job_as_applied)
    assert "job_id" in sig.parameters
    assert "company" in sig.parameters
    assert "job_title" in sig.parameters

    # 10. calibrate_selectors
    sig = inspect.signature(main_mod.calibrate_selectors)
    assert "force_recalibrate" in sig.parameters

    # 11. search_linkedin_jobs
    sig = inspect.signature(main_mod.search_linkedin_jobs)
    for param in ["keywords", "location", "limit", "start", "work_mode"]:
        assert param in sig.parameters

    # 12. get_linkedin_job_details
    sig = inspect.signature(main_mod.get_linkedin_job_details)
    assert "job_id" in sig.parameters

    # 13. notify_new_jobs
    sig = inspect.signature(main_mod.notify_new_jobs)
    for param in ["channel", "sources", "force_refresh", "auto_mark_seen"]:
        assert param in sig.parameters

    # 14. test_notifier
    sig = inspect.signature(main_mod.test_notifier)
    assert "channel" in sig.parameters

    # 15. run_job_scout
    sig = inspect.signature(main_mod.run_job_scout)
    for param in ["cv_path", "sources", "auto_apply", "max_applications", "action_mode"]:
        assert param in sig.parameters

    # 16. set_operation_mode
    sig = inspect.signature(main_mod.set_operation_mode)
    assert "mode" in sig.parameters


def test_m6d_tool_signatures_match_specification():
    """Verify parameter names and signatures of M6-D search plane tools."""
    # job_search
    sig = inspect.signature(main_mod.job_search)
    expected_search_params = [
        "query",
        "location",
        "work_mode",
        "company",
        "tech_stack",
        "limit",
        "sources",
        "freshness_days",
        "cursor",
        "seniority",
        "ctx",
    ]
    for param in expected_search_params:
        assert param in sig.parameters, f"job_search missing parameter {param}"

    # job_fetch
    sig = inspect.signature(main_mod.job_fetch)
    assert "ref" in sig.parameters
    assert "ctx" in sig.parameters

    # discover_companies
    sig = inspect.signature(main_mod.discover_companies)
    assert "company_name" in sig.parameters
    assert "career_url" in sig.parameters
    assert "targets" in sig.parameters
    assert "ctx" in sig.parameters


@pytest.mark.asyncio
async def test_tool_descriptions_meet_guidelines():
    """Verify tool docstrings/descriptions guide agents on opaque refs and factual fetch."""
    tools = await mcp.list_tools()
    tool_map = {t.name: t for t in tools}

    search_desc = tool_map["job_search"].description
    assert "opaque ref" in search_desc
    assert "job_fetch" in search_desc

    fetch_desc = tool_map["job_fetch"].description
    assert "opaque ref" in fetch_desc
    assert "factual" in fetch_desc

    disc_desc = tool_map["discover_companies"].description
    assert "ATS portals" in disc_desc
    assert "Does not persist" in disc_desc
