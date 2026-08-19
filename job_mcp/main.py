"""FastMCP Server for HireMeTech job search, matching, and application automation."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastmcp import Context, FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from job_mcp.core.api_client import (
    JobCache,
    extract_candidate_profile,
    fetch_jobs_via_api,
    fetch_user_resume_profile,
    filter_jobs,
)
from job_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    SessionManager,
)
from job_mcp.core.browser import (
    bookmark_job as browser_bookmark_job,
    delete_job as browser_delete_job,
    dynamic_registry,
    execute_application as browser_execute_application,
    extract_jobs as browser_extract_jobs,
    preview_application as browser_preview_application,
)
from job_mcp.core.discovery import calibrate_all_selectors
from job_mcp.models.schemas import (
    CandidateProfile,
    Job,
    JobPreferences,
    OperationMode,
    ToolResponse,
    WorkMode,
)
from job_mcp.notifiers.base import BaseNotifier
from job_mcp.notifiers.telegram import TelegramNotifier
from job_mcp.notifiers.tracker import JobTracker
from job_mcp.sources import (
    BaseJobSource,
    JobAggregator,
    SourceRegistry,
    create_default_registry,
)
from job_mcp.sources.hiremetech import HireMeTechSource
from job_mcp.sources.linkedin import (
    LinkedInSource,
    search_linkedin_jobs_api,
)
from job_mcp.utils.logger import generate_trace_id, get_logger

logger = get_logger(__name__)


def _response(
    success: bool,
    message: str,
    data: Any = None,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    """Build a ToolResponse dict with auto-generated trace_id."""
    return ToolResponse(
        success=success,
        message=message,
        data=data,
        error_code=error_code,
        trace_id=generate_trace_id(),
    ).model_dump()


# Server system instructions
SERVER_INSTRUCTIONS = """
HireMeTech MCP Server enables intelligent job matching, filtering, bookmarking, notification alerts, and automated job applications across multiple sources (HireMeTech, Comeet, AllJobs, Workday, Eightfold, DirectTech, LinkedIn).

## Operation Modes

The server supports two operation modes, controlled via `set_operation_mode`:

### Supervised Mode (default)
- Present each tool call to the user for confirmation before executing.
- Standard behavior matching typical MCP tool usage.

### Autonomous Mode
- Execute read-only and safe-action tools (`list_job_sources`, `get_job_matches`, `filter_jobs_by_preferences`, `bookmark_job`, `delete_job`, `calibrate_selectors`, `search_linkedin_jobs`, `get_linkedin_job_details`, `notify_new_jobs`, `test_notifier`) WITHOUT asking the user for per-tool confirmation.
- Chain operations freely: list sources -> scan -> filter -> bookmark matching jobs -> notify alerts -> report results.
- The ONLY action that ALWAYS requires explicit user confirmation is `confirm_auto_apply` — actual job application submission. This is a safety-critical action.

## Available Tools
1. `list_job_sources`: List all registered job sources, their capabilities, and current health status.
2. `get_job_matches`: Fetch matched job listings across all or specified job sources. Uses caching for high performance.
3. `filter_jobs_by_preferences`: Filter cached job listings based on tech stack, work mode (remote/hybrid/onsite), location, minimum salary, keywords, exclusion criteria, or CV keyword extraction.
4. `bookmark_job`: Save/favorite a specific job listing by its ID.
5. `delete_job`: Dismiss or hide a job listing from view by its ID.
6. `auto_apply_job`: Step 1 of safe auto-application. Inspects the job application form, generates a preview of fields to submit and warnings, and stages the application.
7. `confirm_auto_apply`: Step 2 of safe auto-application. Submits the staged job application after user confirmation. ALWAYS requires explicit user confirmation, even in autonomous mode.
8. `calibrate_selectors`: Test, discover, and calibrate DOM selectors against the live HireMeTech page with self-healing heuristics.
9. `set_operation_mode`: Switch between 'supervised' and 'autonomous' operation modes.
10. `search_linkedin_jobs`: Search LinkedIn guest jobs directly with keywords, location, workplace type, and filters.
11. `get_linkedin_job_details`: Retrieve full detailed job description, criteria, and direct apply link for a LinkedIn job.
12. `notify_new_jobs`: Aggregate jobs across sources, filter unseen postings via JobTracker, and dispatch alerts to Telegram or other channels.
13. `test_notifier`: Test connectivity, health, and message delivery for notification channels.

## Safety Rules
- Never apply to a job without first inspecting details using `auto_apply_job` (Step 1) and receiving explicit user confirmation before calling `confirm_auto_apply` (Step 2). This rule applies in ALL modes.
- When a user provides a CV path, use `filter_jobs_by_preferences(cv_path=...)` to automatically parse and score matching jobs against the CV.
- In autonomous mode, proceed with read/filter/bookmark/notify operations without waiting for per-tool confirmation. Report a summary of all actions taken at the end.
"""

# Global singletons for direct tool execution or when context is omitted in tests
_default_session: Optional[SessionManager] = None
_default_cache: Optional[JobCache] = None
_default_registry: Optional[SourceRegistry] = None
_default_aggregator: Optional[JobAggregator] = None
_default_tracker: Optional[JobTracker] = None
_default_notifier: Optional[BaseNotifier] = None

# Staged pending applications store: job_id -> application preview dict
_pending_applications: dict[str, dict[str, Any]] = {}

# Current operation mode
_operation_mode: OperationMode = OperationMode.SUPERVISED

# Timeout constant for live scraping operations to avoid proxy / tunnel timeouts (e.g. DevTunnel 10s limit)
_SCRAPE_TIMEOUT_SECONDS: float = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "10.0"))
_WARMUP_TIMEOUT_SECONDS: float = 30.0
_SESSION_COOLDOWN_SECONDS: float = float(os.getenv("SESSION_COOLDOWN_SECONDS", "15.0"))
_last_session_failure_time: float = 0.0


async def _warm_cache(
    session: Optional[SessionManager] = None,
    cache: Optional[JobCache] = None,
    aggregator: Optional[JobAggregator] = None,
) -> None:
    """Warm up the job cache asynchronously in the background during startup.

    Catches all exceptions gracefully so background warmup never crashes the server.
    """
    try:
        if aggregator is not None:
            agg = aggregator
        elif session is not None:
            reg = SourceRegistry()
            reg.register(HireMeTechSource(session_manager=session))
            agg = JobAggregator(registry=reg, cache=cache)
        else:
            agg = _get_aggregator()

        if cache is not None:
            agg.cache = cache

        jobs = await agg.fetch_all_jobs(force_refresh=True)
        if cache is not None and jobs:
            cache.update(jobs)
        logger.info("Cache warmup completed with %d jobs across sources.", len(jobs))
    except asyncio.CancelledError:
        logger.debug("Cache warmup task cancelled.")
        raise
    except Exception as exc:
        logger.warning("Cache warmup failed: %s", exc)


@asynccontextmanager
async def browser_lifespan(server: FastMCP):
    """Lifespan context manager to manage browser session and job cache across server lifecycle."""
    global _default_session, _default_cache, _default_registry, _default_aggregator, _default_tracker, _default_notifier
    logger.info("Starting HireMeTech FastMCP lifespan...")

    session_mgr = SessionManager()
    job_cache = JobCache()
    registry = create_default_registry(session_manager=session_mgr)
    aggregator = JobAggregator(registry=registry, cache=job_cache)
    tracker = JobTracker(storage_path=os.getenv("JOB_TRACKER_PATH"))
    telegram_notifier = TelegramNotifier()

    _default_session = session_mgr
    _default_cache = job_cache
    _default_registry = registry
    _default_aggregator = aggregator
    _default_tracker = tracker
    _default_notifier = telegram_notifier

    warmup_task = asyncio.create_task(_warm_cache(session_mgr, job_cache, aggregator=aggregator))

    try:
        yield {
            "session": session_mgr,
            "cache": job_cache,
            "registry": registry,
            "aggregator": aggregator,
            "tracker": tracker,
            "notifier": telegram_notifier,
        }
    finally:
        logger.info("Shutting down HireMeTech FastMCP lifespan...")
        if warmup_task and not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except (asyncio.CancelledError, Exception):
                pass

        if session_mgr.is_running:
            try:
                await session_mgr.shutdown()
            except Exception as exc:
                logger.error("Error during session shutdown: %s", exc)


# Initialize FastMCP Server
mcp = FastMCP(
    name="HireMeTech",
    instructions=SERVER_INSTRUCTIONS.strip(),
    lifespan=browser_lifespan,
)


class GeminiProbeMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware to ensure preliminary probes from Gemini Spark and other clients succeed."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method

        # Handle CORS OPTIONS preflight
        if method == "OPTIONS":
            return Response(
                b"",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle HEAD requests on any MCP endpoint
        if method == "HEAD":
            return Response(
                b"",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle POST probes on /sse that are sent by clients checking endpoint availability
        if method == "POST" and path == "/sse":
            return Response(
                b"SSE Endpoint Ready",
                status_code=200,
                headers={
                    "Content-Type": "text/plain",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle GET probes on /mcp or /sse when no active session ID is provided
        has_session = bool(request.headers.get("mcp-session-id") or request.query_params.get("session_id"))
        if method == "GET" and path in ("/mcp", "/sse") and not has_session:
            return Response(
                b"MCP Server Active",
                status_code=200,
                headers={
                    "Content-Type": "text/plain",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle DELETE probes on MCP endpoints
        if method == "DELETE" and path in ("/mcp", "/sse"):
            return Response(
                b"",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle OAuth discovery probes
        if (
            path.startswith("/.well-known/oauth-protected-resource")
            or path.startswith("/.well-known/oauth-authorization-server")
        ):
            oauth_metadata = {
                "resource": str(request.base_url).rstrip("/"),
                "authorization_servers": [],
                "scopes_supported": ["mcp"],
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
            }
            return JSONResponse(
                oauth_metadata,
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        response = await call_next(request)

        # Ensure CORS headers are on every response
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"

        # Intercept downstream 409 Conflict responses on MCP/SSE endpoints and map to 200 OK
        if response.status_code == 409 and path in ("/mcp", "/sse"):
            logger.info("Intercepted 409 Conflict on %s %s, returning clean 200 OK.", method, path)
            return Response(
                b"MCP Session Reset",
                status_code=200,
                headers={
                    "Content-Type": "text/plain",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        return response



@mcp.custom_route("/health", methods=["GET", "HEAD"])
async def health_check(request):
    """Health check endpoint for containers and reverse proxies."""
    return JSONResponse({"status": "ok", "server": "HireMeTech MCP"})


@mcp.custom_route("/", methods=["GET", "HEAD", "OPTIONS"])
async def root_endpoint(request):
    """Root status endpoint directing to /mcp."""
    return JSONResponse({
        "status": "ok",
        "server": "HireMeTech FastMCP Server",
        "mcp_endpoint": "/mcp",
        "sse_endpoint": "/sse",
    })


def _get_cache(ctx: Optional[Context] = None) -> JobCache:
    """Retrieve JobCache instance from Context lifespan state or global default."""
    global _default_cache
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "cache" in lifespan_ctx:
            return lifespan_ctx["cache"]
    if _default_cache is None:
        _default_cache = JobCache()
    return _default_cache


def _get_registry(ctx: Optional[Context] = None) -> SourceRegistry:
    """Retrieve SourceRegistry instance from Context lifespan state or global default."""
    global _default_registry
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict):
            if "registry" in lifespan_ctx:
                return lifespan_ctx["registry"]
            if "session" in lifespan_ctx and "aggregator" not in lifespan_ctx and _default_registry is None:
                reg = SourceRegistry()
                reg.register(HireMeTechSource(session_manager=lifespan_ctx["session"]))
                return reg
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry


def _get_aggregator(ctx: Optional[Context] = None) -> JobAggregator:
    """Retrieve JobAggregator instance from Context lifespan state or global default."""
    global _default_aggregator
    cache = _get_cache(ctx)
    registry = _get_registry(ctx)

    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict):
            if "session" in lifespan_ctx:
                hmt = registry.get("hiremetech")
                if isinstance(hmt, HireMeTechSource):
                    hmt.session_manager = lifespan_ctx["session"]
            if "aggregator" in lifespan_ctx:
                agg = lifespan_ctx["aggregator"]
                agg.cache = cache
                return agg

    if _default_aggregator is None:
        _default_aggregator = JobAggregator(registry=registry, cache=cache)
    else:
        _default_aggregator.cache = cache
        _default_aggregator.registry = registry

    return _default_aggregator


def _get_tracker(ctx: Optional[Context] = None) -> JobTracker:
    """Retrieve JobTracker instance from Context lifespan state or global default."""
    global _default_tracker
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "tracker" in lifespan_ctx:
            return lifespan_ctx["tracker"]
    if _default_tracker is None:
        tracker_path = os.getenv("JOB_TRACKER_PATH")
        _default_tracker = JobTracker(storage_path=tracker_path)
    return _default_tracker


def _get_notifier(channel: str = "telegram", ctx: Optional[Context] = None) -> Optional[BaseNotifier]:
    """Retrieve BaseNotifier instance for specified channel from Context or default."""
    global _default_notifier
    channel_lower = channel.strip().lower()
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict):
            if f"{channel_lower}_notifier" in lifespan_ctx:
                return lifespan_ctx[f"{channel_lower}_notifier"]
            if "notifier" in lifespan_ctx and channel_lower == "telegram":
                return lifespan_ctx["notifier"]

    if channel_lower == "telegram":
        if _default_notifier is None or not isinstance(_default_notifier, TelegramNotifier):
            _default_notifier = TelegramNotifier()
        return _default_notifier
    return None


def _get_linkedin_source(ctx: Optional[Context] = None) -> LinkedInSource:
    """Retrieve LinkedInSource instance from registry or create a standalone instance."""
    reg = _get_registry(ctx)
    src = reg.get("linkedin")
    if isinstance(src, LinkedInSource):
        return src

    session_mgr = None
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "session" in lifespan_ctx:
            session_mgr = lifespan_ctx["session"]

    return LinkedInSource(session_manager=session_mgr)


def _get_session(ctx: Optional[Context] = None) -> Optional[SessionManager]:
    """Retrieve SessionManager instance from Context lifespan state or global default."""
    global _default_session
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "session" in lifespan_ctx:
            return lifespan_ctx["session"]
    return _default_session


async def _is_session_authenticated(session: Optional[SessionManager]) -> bool:
    """Fast non-blocking check to determine if a browser session is active and authenticated.

    Does NOT invoke aggressive browser launch recovery if unauthenticated or not running.
    """
    if session is None:
        return False

    # Check is_authenticated attribute / property
    if hasattr(session, "is_authenticated"):
        is_auth = getattr(session, "is_authenticated")
        if isinstance(is_auth, bool) and not is_auth:
            return False
        if callable(is_auth) and not hasattr(is_auth, "assert_called"):
            try:
                res = is_auth()
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, bool) and not res:
                    return False
            except Exception:
                return False

    # Check if session is running
    if hasattr(session, "is_running"):
        is_running = getattr(session, "is_running")
        if isinstance(is_running, bool) and not is_running:
            return False
        if callable(is_running) and not hasattr(is_running, "assert_called"):
            try:
                res = is_running()
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, bool) and not res:
                    return False
            except Exception:
                return False

    # If running, do a fast session health check
    if hasattr(session, "check_session_health") and callable(session.check_session_health):
        try:
            is_healthy = session.check_session_health()
            if asyncio.iscoroutine(is_healthy):
                is_healthy = await is_healthy
            return bool(is_healthy)
        except Exception as exc:
            logger.debug("Fast session health check failed: %s", exc)
            return False

    return False


async def _ensure_session(ctx: Optional[Context] = None) -> tuple[SessionManager, bool]:
    """Retrieve SessionManager and verify authentication health status with cooldown throttling.

    Uses ensure_ready() for automatic retry/recovery on browser failures.
    Applies backoff cooldown after failed attempts to prevent browser launch storms.

    Args:
        ctx: Optional FastMCP Context.

    Returns:
        tuple[SessionManager, bool]: Tuple of (session_manager, is_authenticated_and_healthy).
    """
    global _default_session, _last_session_failure_time
    session = _get_session(ctx)
    if session is None:
        if _default_session is None:
            _default_session = SessionManager()
        session = _default_session

    now = time.monotonic()
    last_fail = getattr(session, "_last_failure_time", None)
    if not isinstance(last_fail, (int, float)):
        last_fail = _last_session_failure_time if session is _default_session else 0.0

    if now - last_fail < _SESSION_COOLDOWN_SECONDS:
        logger.debug(
            "Session recovery cooldown active (%.1fs remaining); skipping browser ensure_ready.",
            _SESSION_COOLDOWN_SECONDS - (now - last_fail),
        )
        return session, False

    try:
        await session.ensure_ready(max_retries=3)
        try:
            session._last_failure_time = 0.0
        except Exception:
            pass
        _last_session_failure_time = 0.0
        return session, True
    except (RuntimeError, Exception) as exc:
        now = time.monotonic()
        try:
            session._last_failure_time = now
        except Exception:
            pass
        _last_session_failure_time = now
        logger.warning("Session could not be made ready after retries: %s", exc)
        return session, False


@mcp.tool()
async def set_operation_mode(
    mode: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Switch between supervised and autonomous operation modes.

    In autonomous mode, the server signals the LLM client that it may chain
    read-only and safe-action tools without per-tool user confirmation.
    Only `confirm_auto_apply` (actual job submission) always requires explicit
    user confirmation.

    Args:
        mode: Operation mode — 'supervised' or 'autonomous'.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse confirming the mode switch.
    """
    global _operation_mode

    mode_clean = mode.strip().lower()
    try:
        new_mode = OperationMode(mode_clean)
    except ValueError:
        return _response(
            success=False,
            message=f"Invalid operation mode '{mode}'. Valid modes: 'supervised', 'autonomous'.",
            error_code="INVALID_MODE",
        )

    _operation_mode = new_mode
    logger.info("Operation mode changed to: %s", new_mode.value)

    return _response(
        success=True,
        message=(
            f"Operation mode set to '{new_mode.value}'. "
            + (
                "The server will now execute read/filter/bookmark tools autonomously. "
                "Only job application submission (confirm_auto_apply) requires explicit user confirmation."
                if new_mode == OperationMode.AUTONOMOUS
                else "All tool calls will be presented for user confirmation."
            )
        ),
        data={"mode": new_mode.value},
    )


@mcp.tool()
async def list_job_sources(
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """List all registered job sources, their capabilities, and current health status.

    Args:
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with registered sources metadata and real-time health checks.
    """
    registry = _get_registry(ctx)
    aggregator = _get_aggregator(ctx)

    try:
        sources_meta = [m.model_dump() for m in registry.list_sources()]
        health_status = await aggregator.check_all_health()
        return _response(
            success=True,
            message=f"Retrieved {len(sources_meta)} registered job sources.",
            data={
                "sources": sources_meta,
                "health": health_status,
            },
        )
    except Exception as exc:
        logger.exception("Error in list_job_sources: %s", exc)
        return _response(
            success=False,
            message=f"Failed to list job sources: {exc}",
            error_code="SOURCES_ERROR",
        )


@mcp.tool()
async def get_job_matches(
    sources: Optional[list[str]] = None,
    tech_stack: Optional[list[str]] = None,
    work_mode: Optional[str] = None,
    location: Optional[str] = None,
    min_salary: Optional[int] = None,
    keywords: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
    cv_path: Optional[str] = None,
    force_refresh: bool = False,
    limit: int = 50,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Fetch matched job listings from registered sources (HireMeTech, Comeet, AllJobs, LinkedIn, Workday, Eightfold, DirectTech).

    Uses cached job listings unless cache is stale or force_refresh is True.
    Optionally filters by specific source IDs (e.g. ['hiremetech', 'comeet', 'alljobs']).
    When cv_path is provided, extracts candidate profile and dynamically populates queries, seniority exclusions, and scores.

    Args:
        sources: Optional list of source IDs to query (e.g. ['hiremetech', 'linkedin', 'workday']).
                 If None, queries all active registered sources.
        tech_stack: Preferred technologies (e.g. ['Python', 'FastAPI']).
        work_mode: Preferred work mode: 'remote', 'hybrid', or 'onsite'.
        location: Target location (e.g. 'Tel Aviv', 'Remote').
        min_salary: Minimum desired annual salary.
        keywords: Additional keywords to require or match.
        exclude_keywords: Keywords to exclude (e.g. ['Junior', 'PHP']).
        cv_path: Path to resume/CV file (or raw CV text) for automatic skill extraction and matching.
        force_refresh: Force a live scrape/fetch from sources even if cache is fresh.
        limit: Maximum number of jobs to return (default: 50).
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with list of job listings.
    """
    cache = _get_cache(ctx)

    # 1. Profile and Preferences Resolution
    effective_cv_path = cv_path or os.getenv("DEFAULT_CV_PATH")
    profile: Optional[CandidateProfile] = None
    if effective_cv_path:
        profile = extract_candidate_profile(effective_cv_path)

    parsed_work_mode: Optional[WorkMode] = None
    if work_mode:
        try:
            parsed_work_mode = WorkMode(work_mode.strip().lower())
        except ValueError:
            logger.debug("Unknown work mode for get_job_matches: '%s'", work_mode)

    effective_tech_stack = list(tech_stack or [])
    effective_keywords = list(keywords or [])
    effective_exclude = list(exclude_keywords or [])

    if profile is not None:
        if not effective_tech_stack:
            effective_tech_stack = list(profile.primary_stack or profile.top_skills or profile.skills)
        if not effective_keywords and profile.search_queries:
            effective_keywords = list(profile.search_queries[:3])
        if not effective_exclude and profile.suggested_exclusions:
            effective_exclude = list(profile.suggested_exclusions)

    prefs: Optional[JobPreferences] = None
    if (
        effective_tech_stack
        or parsed_work_mode
        or location
        or min_salary
        or effective_keywords
        or effective_exclude
        or effective_cv_path
        or profile is not None
    ):
        prefs = JobPreferences(
            tech_stack=effective_tech_stack,
            work_mode=parsed_work_mode,
            location=location,
            min_salary=min_salary,
            keywords=effective_keywords,
            exclude_keywords=effective_exclude,
            cv_path=effective_cv_path,
        )

    # 2. Check Cache
    if not force_refresh:
        cached_jobs = cache.get_all()
        if cached_jobs:
            if sources is not None:
                source_set = set(sources)
                cached_jobs = [
                    j
                    for j in cached_jobs
                    if j.source in source_set or any(s in source_set for s in getattr(j, "sources", []))
                ]
            if prefs is not None or profile is not None:
                cached_jobs = filter_jobs(cached_jobs, prefs or JobPreferences(), profile=profile)
            if limit and len(cached_jobs) > limit:
                cached_jobs = cached_jobs[:limit]
            logger.info("Returning %d jobs from cache.", len(cached_jobs))
            return _response(
                success=True,
                message=f"Retrieved {len(cached_jobs)} cached job matches.",
                data=[job.model_dump() for job in cached_jobs],
            )

    aggregator = _get_aggregator(ctx)
    aggregator.cache = cache

    # Ensure HireMeTechSource has current session if provided
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "session" in lifespan_ctx:
            hmt = aggregator.registry.get("hiremetech")
            if isinstance(hmt, HireMeTechSource):
                hmt.session_manager = lifespan_ctx["session"]

    # If single-source HireMeTech is requested or only HireMeTech is in registry, verify session auth
    active = aggregator.registry.get_active(sources)
    if len(active) == 1 and active[0].source_id == "hiremetech":
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return _response(
                success=False,
                message=(
                    "Browser session is not authenticated or not logged into HireMeTech. "
                    "Please run 'python -m hireme_mcp.setup' to authenticate."
                ),
                error_code="UNAUTHENTICATED",
            )

    try:
        jobs = await asyncio.wait_for(
            aggregator.fetch_all_jobs(
                sources=sources,
                preferences=prefs,
                force_refresh=force_refresh,
                profile=profile,
            ),
            timeout=_SCRAPE_TIMEOUT_SECONDS,
        )
        if limit and len(jobs) > limit:
            jobs = jobs[:limit]

        return _response(
            success=True,
            message=f"Successfully fetched {len(jobs)} live job matches.",
            data=[job.model_dump() for job in jobs],
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("Scraping jobs timed out after %.1fs.", _SCRAPE_TIMEOUT_SECONDS)
        return _response(
            success=False,
            message=f"Scraping job matches timed out after {_SCRAPE_TIMEOUT_SECONDS:.0f} seconds.",
            error_code="FETCH_ERROR",
        )
    except Exception as exc:
        logger.exception("Error in get_job_matches: %s", exc)
        return _response(
            success=False,
            message=f"Failed to fetch job matches: {exc}",
            error_code="FETCH_ERROR",
        )


@mcp.tool()
async def filter_jobs_by_preferences(
    tech_stack: Optional[list[str]] = None,
    work_mode: Optional[str] = None,
    location: Optional[str] = None,
    min_salary: Optional[int] = None,
    keywords: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
    cv_path: Optional[str] = None,
    limit: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Filter and rank cached job listings based on tech stack, location, salary, or CV.

    Args:
        tech_stack: Preferred technologies (e.g. ['Python', 'FastAPI', 'React']).
        work_mode: Preferred work mode: 'remote', 'hybrid', or 'onsite'.
        location: Target location (e.g. 'New York', 'London', 'Remote').
        min_salary: Minimum desired annual salary in USD.
        keywords: Additional keywords to require or match in job descriptions.
        exclude_keywords: Keywords to exclude (e.g. ['Junior', 'PHP', 'Contract']).
        cv_path: Path to resume/CV file (PDF, DOCX, TXT) for automatic skill extraction and matching.
        limit: Optional maximum number of jobs to return.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with filtered and match-scored job listings.
    """
    cache = _get_cache(ctx)
    cached_jobs = cache.get_all()

    if not cached_jobs:
        return _response(
            success=False,
            message="No job listings found in cache. Please call 'get_job_matches' first to load jobs.",
            error_code="NO_CACHED_JOBS",
        )

    parsed_work_mode: Optional[WorkMode] = None
    if work_mode:
        mode_clean = work_mode.strip().lower()
        try:
            parsed_work_mode = WorkMode(mode_clean)
        except ValueError:
            logger.debug("Unknown work mode: '%s'. Proceeding with string match.", work_mode)

    effective_cv_path = cv_path or os.getenv("DEFAULT_CV_PATH")
    effective_tech_stack = list(tech_stack or [])
    effective_keywords = list(keywords or [])
    effective_exclude = list(exclude_keywords or [])

    profile: Optional[CandidateProfile] = None
    if effective_cv_path:
        profile = extract_candidate_profile(effective_cv_path)
        if not effective_tech_stack:
            effective_tech_stack = list(profile.primary_stack or profile.top_skills or profile.skills)
        if not effective_keywords and profile.search_queries:
            effective_keywords = list(profile.search_queries[:3])
        if not effective_exclude and profile.suggested_exclusions:
            effective_exclude = list(profile.suggested_exclusions)

    # If no local CV path is provided, attempt to supplement skills from online user resume profile
    if not effective_cv_path and not effective_tech_stack and not effective_keywords:
        try:
            session, is_healthy = await _ensure_session(ctx)
            if is_healthy:
                page = await session.get_page()
                online_profile = await asyncio.wait_for(fetch_user_resume_profile(page.request), timeout=3.0)
                if isinstance(online_profile, dict):
                    skills = online_profile.get("technical_skills") or online_profile.get("skills") or []
                    if isinstance(skills, list) and skills:
                        effective_tech_stack = [s for s in skills if isinstance(s, str) and s.strip()]
        except Exception as exc:
            logger.debug("Optional resume profile fetch skipped or failed: %s", exc)

    prefs = JobPreferences(
        tech_stack=effective_tech_stack,
        work_mode=parsed_work_mode,
        location=location,
        min_salary=min_salary,
        keywords=effective_keywords,
        exclude_keywords=effective_exclude,
        cv_path=effective_cv_path,
    )

    try:
        filtered = filter_jobs(cached_jobs, prefs, profile=profile)
        if limit and len(filtered) > limit:
            filtered = filtered[:limit]
        return _response(
            success=True,
            message=f"Found {len(filtered)} matching jobs (out of {len(cached_jobs)} total).",
            data=[job.model_dump() for job in filtered],
        )
    except Exception as exc:
        logger.exception("Error in filter_jobs_by_preferences: %s", exc)
        return _response(
            success=False,
            message=f"Failed to filter jobs: {exc}",
            error_code="FILTER_ERROR",
        )


@mcp.tool()
async def bookmark_job(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Save or bookmark a job listing by its ID.

    Args:
        job_id: Unique ID of the job listing.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse indicating bookmark success or failure.
    """
    try:
        cache = _get_cache(ctx)
        cached_job = cache.get_by_id(job_id)
        title = cached_job.title if cached_job else None
        company = cached_job.company if cached_job else None
        source = cached_job.source if cached_job else None
        logger.info(
            "Bookmarked job",
            job_id=job_id,
            title=title,
            company=company,
            source=source,
        )

        # Update cache if job present
        if cached_job:
            cached_job.is_bookmarked = True

        # Handle external ATS job sources (Comeet, AllJobs, Workday, Eightfold, DirectTech, LinkedIn, etc.)
        if (cached_job is not None and cached_job.source != "hiremetech") or job_id.startswith(
            ("comeet_", "alljobs_", "workday_", "eightfold_", "direct_", "linkedin_")
        ):
            return _response(
                success=True,
                message=f"Successfully bookmarked external job '{job_id}' in cache.",
                data={"job_id": job_id, "is_bookmarked": True},
            )

        session = _get_session(ctx)
        is_healthy = await _is_session_authenticated(session)
        if not is_healthy or session is None:
            return _response(
                success=True,
                message=f"Successfully bookmarked job '{job_id}' in cache (portal bookmark skipped: unauthenticated session).",
                data={"job_id": job_id, "is_bookmarked": True, "portal_bookmarked": False},
            )

        page = await session.get_page()
        await browser_bookmark_job(page, job_id)

        return _response(
            success=True,
            message=f"Successfully bookmarked job '{job_id}'.",
            data={"job_id": job_id, "is_bookmarked": True, "portal_bookmarked": True},
        )

    except Exception as exc:
        logger.exception("Error in bookmark_job for '%s': %s", job_id, exc)
        return _response(
            success=False,
            message=f"Failed to bookmark job '{job_id}': {exc}",
            error_code="BOOKMARK_ERROR",
        )


@mcp.tool()
async def delete_job(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Dismiss or hide a job listing from view.

    Args:
        job_id: Unique ID of the job listing.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse indicating dismissal success or failure.
    """
    try:
        cache = _get_cache(ctx)
        cached_job = cache.get_by_id(job_id)
        title = cached_job.title if cached_job else None
        company = cached_job.company if cached_job else None
        source = cached_job.source if cached_job else None
        logger.info(
            "Dismissed job from cache",
            job_id=job_id,
            title=title,
            company=company,
            source=source,
        )
        cache.dismiss(job_id)

        # Handle external ATS job sources (Comeet, AllJobs, Workday, Eightfold, DirectTech, LinkedIn, etc.)
        if (cached_job is not None and cached_job.source != "hiremetech") or job_id.startswith(
            ("comeet_", "alljobs_", "workday_", "eightfold_", "direct_", "linkedin_")
        ):
            return _response(
                success=True,
                message=f"Successfully dismissed external job '{job_id}' from view and cache.",
                data={"job_id": job_id},
            )

        session = _get_session(ctx)
        is_healthy = await _is_session_authenticated(session)
        if not is_healthy or session is None:
            return _response(
                success=True,
                message=f"Successfully dismissed job '{job_id}' from cache (portal dismissal skipped: unauthenticated session).",
                data={"job_id": job_id, "portal_deleted": False},
            )

        page = await session.get_page()
        await browser_delete_job(page, job_id)

        return _response(
            success=True,
            message=f"Successfully dismissed/deleted job '{job_id}'.",
            data={"job_id": job_id, "portal_deleted": True},
        )

    except Exception as exc:
        logger.exception("Error in delete_job for '%s': %s", job_id, exc)
        return _response(
            success=False,
            message=f"Failed to delete job '{job_id}': {exc}",
            error_code="DELETE_ERROR",
        )


@mcp.tool()
async def auto_apply_job(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Step 1: Inspect job application form and generate preview without submitting.

    Stores the preview in pending applications for subsequent confirmation via `confirm_auto_apply`.

    Args:
        job_id: Unique ID of the job listing.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse containing application preview, fields, and confirmation instructions.
    """
    try:
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return _response(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            )

        page = await session.get_page()
        preview = await browser_preview_application(page, job_id)

        # Store preview in pending applications store
        _pending_applications[job_id] = preview.model_dump()

        return _response(
            success=True,
            message=(
                f"Application preview generated for job '{job_id}' ({preview.job_title} at {preview.company}). "
                f"Please review the application fields and warnings carefully. "
                f"To submit the application, call 'confirm_auto_apply(job_id=\"{job_id}\")'."
            ),
            data=preview.model_dump(),
        )

    except Exception as exc:
        logger.exception("Error in auto_apply_job for '%s': %s", job_id, exc)
        return _response(
            success=False,
            message=f"Failed to generate application preview for job '{job_id}': {exc}",
            error_code="PREVIEW_ERROR",
        )


@mcp.tool()
async def confirm_auto_apply(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Step 2: Confirm and execute job application submission.

    Requires `auto_apply_job` to have been called first for this job_id.

    Args:
        job_id: Unique ID of the job listing previously previewed.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with submission status and details.
    """
    if job_id not in _pending_applications:
        return _response(
            success=False,
            message=(
                f"No pending application preview found for job '{job_id}'. "
                f"You MUST call 'auto_apply_job(job_id=\"{job_id}\")' first to preview and verify before confirming."
            ),
            error_code="NO_PENDING_PREVIEW",
        )

    try:
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return _response(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            )

        page = await session.get_page()
        await browser_execute_application(page, job_id)

        preview_details = _pending_applications.pop(job_id)

        return _response(
            success=True,
            message=(
                f"Successfully submitted application for job '{job_id}' "
                f"({preview_details.get('job_title', 'Position')} at {preview_details.get('company', 'Employer')})."
            ),
            data={
                "job_id": job_id,
                "submitted": True,
                "application_details": preview_details,
            },
        )

    except Exception as exc:
        logger.exception("Error in confirm_auto_apply for '%s': %s", job_id, exc)
        return _response(
            success=False,
            message=f"Failed to submit application for job '{job_id}': {exc}",
            error_code="APPLY_EXECUTION_ERROR",
        )


@mcp.tool()
async def calibrate_selectors(
    force_recalibrate: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Test, discover, and calibrate DOM selectors against the live page.

    Tests all registered selectors against the live DOM, heuristically discovers
    alternative selectors if layout changes have occurred, and persists verified
    selectors into the dynamic selector registry.

    Args:
        force_recalibrate: If True, clears existing dynamic registry cache before calibration.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse containing calibration results and status per selector key.
    """
    try:
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return _response(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            )

        page = await session.get_page()
        # Navigate to dashboard if not already there
        if DASHBOARD_PATH not in (page.url or ""):
            await page.goto(f"{BASE_URL}{DASHBOARD_PATH}", wait_until="commit", timeout=20000)
            if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                t = page.wait_for_timeout(2500)
                if asyncio.iscoroutine(t):
                    await t

        if force_recalibrate:
            dynamic_registry.clear()

        results = await calibrate_all_selectors(page, dynamic_registry)

        matched_count = sum(1 for v in results.values() if v.get("status") != "failed")
        total_count = len(results)

        return _response(
            success=True,
            message=f"Calibrated {matched_count}/{total_count} selectors successfully.",
            data={
                "results": results,
                "matched_count": matched_count,
                "total_count": total_count,
            },
        )

    except Exception as exc:
        logger.exception("Error during selector calibration: %s", exc)
        return _response(
            success=False,
            message=f"Selector calibration failed: {exc}",
            error_code="CALIBRATION_ERROR",
        )


@mcp.tool()
async def search_linkedin_jobs(
    keywords: str = "",
    location: str = "Israel",
    start: int = 0,
    work_mode: Optional[str] = None,
    f_WT: Optional[str] = None,
    f_TPR: Optional[str] = None,
    f_AL: Optional[bool] = None,
    f_E: Optional[str] = None,
    sort_by: Optional[str] = None,
    cv_path: Optional[str] = None,
    limit: int = 25,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Search LinkedIn jobs using the high-speed guest seeMoreJobPostings endpoint.

    Args:
        keywords: Search query / keywords (e.g. 'Software Engineer', 'Python').
        location: Target geographic location (default: 'Israel').
        start: Pagination start offset index.
        work_mode: Optional work mode filter: 'onsite', 'remote', or 'hybrid'.
        f_WT: Raw LinkedIn workplace type parameter ('1'=onsite, '2'=remote, '3'=hybrid).
        f_TPR: Time posted range (e.g. 'r86400' for past 24h, 'r604800' for past week).
        f_AL: Filter for Easy Apply jobs.
        f_E: Experience level filter (e.g. '1'=internship, '2'=entry, '3'=associate, '4'=mid-senior).
        sort_by: Sorting parameter ('R'=relevant, 'DD'=date posted).
        cv_path: Optional path to resume/CV file (or raw CV text) to dynamically derive search keywords if keywords is empty.
        limit: Maximum number of jobs to return (default: 25).
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with list of standardized LinkedIn job listings.
    """
    try:
        parsed_work_mode: Optional[WorkMode] = None
        if work_mode:
            try:
                parsed_work_mode = WorkMode(work_mode.strip().lower())
            except ValueError:
                logger.debug("Unknown work mode for LinkedIn search: '%s'", work_mode)

        effective_keywords = (keywords or "").strip()
        effective_cv_path = cv_path or os.getenv("DEFAULT_CV_PATH")
        if not effective_keywords and effective_cv_path:
            profile = extract_candidate_profile(effective_cv_path)
            if profile:
                if profile.search_queries:
                    effective_keywords = profile.search_queries[0]
                elif profile.top_skills:
                    effective_keywords = " ".join(profile.top_skills[:2])
                elif profile.skills:
                    effective_keywords = " ".join(profile.skills[:2])

        jobs = await search_linkedin_jobs_api(
            keywords=effective_keywords,
            location=location,
            start=start,
            work_mode=parsed_work_mode,
            f_WT=f_WT,
            f_TPR=f_TPR,
            f_AL=f_AL,
            f_E=f_E,
            sort_by=sort_by,
        )

        if limit and len(jobs) > limit:
            jobs = jobs[:limit]

        cache = _get_cache(ctx)
        if jobs:
            cache.update(jobs)

        return _response(
            success=True,
            message=f"Successfully fetched {len(jobs)} LinkedIn job matches.",
            data=[j.model_dump() for j in jobs],
        )
    except Exception as exc:
        logger.exception("Error in search_linkedin_jobs: %s", exc)
        return _response(
            success=False,
            message=f"Failed to search LinkedIn jobs: {exc}",
            error_code="LINKEDIN_SEARCH_ERROR",
        )


@mcp.tool()
async def get_linkedin_job_details(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Retrieve full job details for a specific LinkedIn job posting.

    Args:
        job_id: LinkedIn job ID (e.g. 'linkedin_4152839402' or '4152839402').
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with detailed fields including full description, criteria, apply URL, etc.
    """
    try:
        linkedin_source = _get_linkedin_source(ctx)
        details = await linkedin_source.fetch_job_details(job_id)

        if not details:
            return _response(
                success=False,
                message=f"Could not retrieve details for LinkedIn job '{job_id}'.",
                error_code="JOB_NOT_FOUND",
            )

        serializable_details = dict(details)
        if "work_mode" in serializable_details and hasattr(serializable_details["work_mode"], "value"):
            serializable_details["work_mode"] = serializable_details["work_mode"].value

        return _response(
            success=True,
            message=f"Successfully retrieved details for LinkedIn job '{job_id}'.",
            data=serializable_details,
        )
    except Exception as exc:
        logger.exception("Error in get_linkedin_job_details for '%s': %s", job_id, exc)
        return _response(
            success=False,
            message=f"Failed to get LinkedIn job details: {exc}",
            error_code="LINKEDIN_DETAILS_ERROR",
        )


@mcp.tool()
async def notify_new_jobs(
    channel: str = "telegram",
    sources: Optional[list[str]] = None,
    min_score: Optional[float] = None,
    tech_stack: Optional[list[str]] = None,
    work_mode: Optional[str] = None,
    location: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    force_refresh: bool = False,
    auto_mark_seen: bool = True,
    limit: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Scan aggregated jobs, filter unseen postings via JobTracker, and dispatch alerts.

    Args:
        channel: Notification channel (default: 'telegram').
        sources: Optional list of source IDs to fetch from (e.g. ['linkedin', 'comeet', 'workday']).
        min_score: Minimum match score (0-100) threshold for alerting.
        tech_stack: Preferred tech stack list for ranking/filtering.
        work_mode: Preferred work mode: 'remote', 'hybrid', or 'onsite'.
        location: Geographic location filter.
        keywords: Additional keywords to require or match.
        force_refresh: Force live fetch from sources bypassing cache.
        auto_mark_seen: Automatically record notified jobs in tracker to avoid duplicate alerts.
        limit: Optional maximum number of unseen jobs to notify in one batch.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with notification dispatch summary and notified jobs.
    """
    try:
        notifier = _get_notifier(channel=channel, ctx=ctx)
        if notifier is None:
            return _response(
                success=False,
                message=f"Unsupported notification channel '{channel}'. Supported: 'telegram'.",
                error_code="UNSUPPORTED_CHANNEL",
            )

        if not getattr(notifier, "is_configured", True):
            return _response(
                success=False,
                message="Telegram notifier is not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
                error_code="NOTIFIER_NOT_CONFIGURED",
            )

        parsed_work_mode: Optional[WorkMode] = None
        if work_mode:
            try:
                parsed_work_mode = WorkMode(work_mode.strip().lower())
            except ValueError:
                logger.debug("Unknown work mode: '%s'", work_mode)

        prefs: Optional[JobPreferences] = None
        if tech_stack or parsed_work_mode or location or keywords or min_score:
            prefs = JobPreferences(
                tech_stack=list(tech_stack or []),
                work_mode=parsed_work_mode,
                location=location,
                keywords=list(keywords or []),
            )

        aggregator = _get_aggregator(ctx)
        jobs = await aggregator.fetch_all_jobs(
            sources=sources,
            preferences=prefs,
            force_refresh=force_refresh,
        )

        if min_score is not None:
            jobs = [j for j in jobs if (j.match_score or 0) >= min_score]

        tracker = _get_tracker(ctx)
        unseen_jobs = tracker.filter_unseen(jobs, auto_mark=False)

        if limit and len(unseen_jobs) > limit:
            unseen_jobs = unseen_jobs[:limit]

        if not unseen_jobs:
            return _response(
                success=True,
                message="No new unseen jobs to notify.",
                data={
                    "channel": channel,
                    "notified_count": 0,
                    "total_matched": len(jobs),
                    "jobs": [],
                },
            )

        alert_sent = await notifier.send_alert(
            unseen_jobs,
            title=f"🎯 New Job Matches ({len(unseen_jobs)})",
        )

        if not alert_sent:
            return _response(
                success=False,
                message=f"Failed to dispatch alert for {len(unseen_jobs)} jobs via {channel}.",
                error_code="NOTIFICATION_FAILED",
            )

        if auto_mark_seen:
            tracker.mark_many_seen(unseen_jobs)

        return _response(
            success=True,
            message=f"Successfully dispatched alert for {len(unseen_jobs)} new jobs via {channel}.",
            data={
                "channel": channel,
                "notified_count": len(unseen_jobs),
                "total_matched": len(jobs),
                "jobs": [j.model_dump() for j in unseen_jobs],
            },
        )

    except Exception as exc:
        logger.exception("Error in notify_new_jobs: %s", exc)
        return _response(
            success=False,
            message=f"Notification dispatch failed: {exc}",
            error_code="NOTIFY_ERROR",
        )


@mcp.tool()
async def test_notifier(
    channel: str = "telegram",
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Verify notifier connectivity, health check, and dispatch a test alert message.

    Args:
        channel: Notification channel to test (default: 'telegram').
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with health status and test message delivery result.
    """
    try:
        notifier = _get_notifier(channel=channel, ctx=ctx)
        if notifier is None:
            return _response(
                success=False,
                message=f"Unsupported notification channel '{channel}'. Supported: 'telegram'.",
                error_code="UNSUPPORTED_CHANNEL",
            )

        if not getattr(notifier, "is_configured", True):
            return _response(
                success=False,
                message="Telegram notifier is not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
                error_code="NOTIFIER_NOT_CONFIGURED",
            )

        is_healthy = await notifier.check_health()
        if not is_healthy:
            return _response(
                success=False,
                message=f"Health check failed for {channel} notifier (invalid credentials or unreachable endpoint).",
                error_code="HEALTH_CHECK_FAILED",
            )

        test_job = Job(
            job_id="test_notifier_job",
            title="Senior Test Engineer",
            company="HireMeTech Verification",
            location="Remote",
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastMCP", "Telegram"],
            description="Test alert verifying notification delivery pipeline.",
            source="system",
            sources=["system"],
            url="https://github.com",
            apply_url="https://github.com",
        )

        sent = await notifier.send_alert(
            [test_job],
            title="🧪 HireMeTech MCP Notifier Test",
        )

        if not sent:
            return _response(
                success=False,
                message=f"Health check passed but failed to send test alert via {channel}.",
                error_code="DELIVERY_FAILED",
            )

        return _response(
            success=True,
            message=f"Test notification successfully verified and dispatched via {channel}.",
            data={
                "channel": channel,
                "healthy": True,
                "delivered": True,
            },
        )

    except Exception as exc:
        logger.exception("Error in test_notifier: %s", exc)
        return _response(
            success=False,
            message=f"Failed to test notifier: {exc}",
            error_code="TEST_NOTIFIER_ERROR",
        )


