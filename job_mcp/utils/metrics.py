"""Bounded Prometheus instrumentation for MCP operations."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from fastmcp.server.middleware import Middleware
from prometheus_client import Counter, Histogram

TOOL_NAMES = frozenset({
    "list_job_sources", "get_job_matches", "filter_jobs_by_preferences",
    "bookmark_job", "delete_job", "auto_apply_job", "confirm_auto_apply",
    "calibrate_selectors", "set_operation_mode", "search_linkedin_jobs",
    "get_linkedin_job_details", "notify_new_jobs", "test_notifier",
    "run_job_scout", "get_application_history", "mark_job_as_applied",
})
SOURCE_NAMES = frozenset({
    "hiremetech", "comeet", "alljobs", "workday", "eightfold", "direct_tech",
    "linkedin", "jobify", "greenhouse", "lever", "gotfriends",
})

TOOL_EXECUTION_DURATION = Histogram(
    "mcp_tool_execution_duration_seconds",
    "MCP tool execution duration.",
    ["tool_name", "status"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0],
)
SYSTEM1_TRIAGE_TOTAL = Counter(
    "mcp_system1_triage_total", "System 1 triage decisions.", ["decision"]
)
SOURCE_FETCH_DURATION = Histogram(
    "mcp_source_fetch_duration_seconds", "Source fetch duration.", ["source_name"]
)
ESTIMATED_COST_SAVED_USD = Counter(
    "mcp_estimated_cost_saved_usd", "Estimated LLM API cost avoided in USD."
)


def canonical_tool_name(name: object) -> str:
    """Return the bounded tool label for an incoming tool name."""
    return name if isinstance(name, str) and name in TOOL_NAMES else "unknown"


def canonical_source_name(name: object) -> str:
    """Return the bounded source label for a configured source."""
    return name if isinstance(name, str) and name in SOURCE_NAMES else "other"


def _result_failed(result: Any) -> bool:
    if getattr(result, "is_error", False):
        return True
    structured = getattr(result, "structured_content", None)
    return isinstance(structured, dict) and structured.get("success") is False


class ToolMetricsMiddleware(Middleware):
    """Records exactly one bounded observation for every MCP tool call."""

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        tool_name = canonical_tool_name(getattr(context.message, "name", None))
        started = perf_counter()
        try:
            result = await call_next(context)
        except Exception:
            TOOL_EXECUTION_DURATION.labels(tool_name=tool_name, status="error").observe(perf_counter() - started)
            raise
        status = "error" if _result_failed(result) else "success"
        TOOL_EXECUTION_DURATION.labels(tool_name=tool_name, status=status).observe(perf_counter() - started)
        return result


def record_local_accept_cost_saving() -> None:
    """Increment only the configured, explicitly estimated local-triage saving."""
    try:
        amount = float(os.getenv("SYSTEM1_LOCAL_ACCEPT_COST_SAVED_USD", "0"))
    except ValueError:
        return
    if amount > 0:
        ESTIMATED_COST_SAVED_USD.inc(amount)
