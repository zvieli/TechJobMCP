"""FastMCP Server for HireMeTech job search, matching, and application automation."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastmcp import Context, FastMCP

from hireme_mcp.core.api_client import JobCache, filter_jobs
from hireme_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    SessionManager,
)
from hireme_mcp.core.browser import (
    bookmark_job as browser_bookmark_job,
    delete_job as browser_delete_job,
    dynamic_registry,
    execute_application as browser_execute_application,
    extract_jobs as browser_extract_jobs,
    preview_application as browser_preview_application,
)
from hireme_mcp.core.discovery import calibrate_all_selectors
from hireme_mcp.models.schemas import (
    Job,
    JobPreferences,
    ToolResponse,
    WorkMode,
)
from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)

# Server system instructions
SERVER_INSTRUCTIONS = """
HireMeTech MCP Server enables intelligent job matching, filtering, bookmarking, and automated job applications on HireMeTech.

Available Tools:
1. `get_job_matches`: Fetch matched job listings from your HireMeTech dashboard. Uses caching for high performance.
2. `filter_jobs_by_preferences`: Filter cached job listings based on tech stack, work mode (remote/hybrid/onsite), location, minimum salary, keywords, exclusion criteria, or CV keyword extraction.
3. `bookmark_job`: Save/favorite a specific job listing by its ID.
4. `delete_job`: Dismiss or hide a job listing from view by its ID.
5. `auto_apply_job`: Step 1 of safe auto-application. Inspects the job application form, generates a preview of fields to submit and warnings, and stages the application.
6. `confirm_auto_apply`: Step 2 of safe auto-application. Submits the staged job application after user confirmation.
7. `calibrate_selectors`: Test, discover, and calibrate DOM selectors against the live HireMeTech page with self-healing heuristics.

Safety Rules:
- Never apply to a job without first inspecting details using `auto_apply_job` (Step 1) and receiving explicit user confirmation before calling `confirm_auto_apply` (Step 2).
- When a user provides a CV path, use `filter_jobs_by_preferences(cv_path=...)` to automatically parse and score matching jobs against the CV.
"""

# Global singletons for direct tool execution or when context is omitted in tests
_default_session: Optional[SessionManager] = None
_default_cache: Optional[JobCache] = None

# Staged pending applications store: job_id -> application preview dict
_pending_applications: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def browser_lifespan(server: FastMCP):
    """Lifespan context manager to manage browser session and job cache across server lifecycle."""
    global _default_session, _default_cache
    logger.info("Starting HireMeTech FastMCP lifespan...")

    session_mgr = SessionManager()
    job_cache = JobCache()
    _default_session = session_mgr
    _default_cache = job_cache

    try:
        await session_mgr.initialize()
    except Exception as exc:
        logger.warning("SessionManager initial start notice: %s", exc)

    try:
        yield {"session": session_mgr, "cache": job_cache}
    finally:
        logger.info("Shutting down HireMeTech FastMCP lifespan...")
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


from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


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

        # Handle GET probes on /mcp or /sse when not establishing an active SSE stream
        if method == "GET" and path in ("/mcp", "/sse"):
            accept_header = request.headers.get("accept", "")
            if "text/event-stream" not in accept_header:
                return Response(
                    b"MCP Server Active",
                    status_code=200,
                    headers={
                        "Content-Type": "text/plain",
                        "Access-Control-Allow-Origin": "*",
                    },
                )

        # Handle DELETE probes on MCP endpoints
        if method == "DELETE" and path in ("/mcp", "/sse"):
            return Response(
                b"",
                status_code=200,
                headers={"Access-Control-Allow-Origin": "*"},
            )

        # Handle OAuth discovery probes
        if path.startswith("/.well-known/oauth-protected-resource"):
            return JSONResponse({}, status_code=200, headers={"Access-Control-Allow-Origin": "*"})

        return await call_next(request)



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


async def _ensure_session(ctx: Optional[Context] = None) -> tuple[SessionManager, bool]:
    """Retrieve SessionManager and verify authentication health status.

    Args:
        ctx: Optional FastMCP Context.

    Returns:
        tuple[SessionManager, bool]: Tuple of (session_manager, is_authenticated_and_healthy).
    """
    global _default_session
    session: Optional[SessionManager] = None
    if ctx is not None:
        lifespan_ctx = getattr(ctx, "lifespan_context", None)
        if isinstance(lifespan_ctx, dict) and "session" in lifespan_ctx:
            session = lifespan_ctx["session"]

    if session is None:
        if _default_session is None:
            _default_session = SessionManager()
        session = _default_session

    if not session._initialized:
        await session.initialize()

    is_healthy = await session.check_session_health()
    return session, is_healthy


@mcp.tool()
async def get_job_matches(
    force_refresh: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Fetch matched job listings from HireMeTech dashboard.

    Uses cached job listings unless cache is stale or force_refresh is True.

    Args:
        force_refresh: Force a live scrape from the browser even if cache is fresh.
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with list of job listings.
    """
    cache = _get_cache(ctx)

    if not force_refresh and not cache.is_stale:
        cached_jobs = cache.get_all()
        if cached_jobs:
            logger.info("Returning %d jobs from cache.", len(cached_jobs))
            return ToolResponse(
                success=True,
                message=f"Retrieved {len(cached_jobs)} cached job matches.",
                data=[job.model_dump() for job in cached_jobs],
            ).model_dump()

    try:
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return ToolResponse(
                success=False,
                message=(
                    "Browser session is not authenticated or not logged into HireMeTech. "
                    "Please run 'python -m hireme_mcp.setup' to authenticate."
                ),
                error_code="UNAUTHENTICATED",
            ).model_dump()

        page = await session.get_page()
        target_url = f"{BASE_URL}{DASHBOARD_PATH}"
        if DASHBOARD_PATH not in (page.url or ""):
            await page.goto(target_url, wait_until="commit", timeout=20000)
            if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                t = page.wait_for_timeout(2500)
                if asyncio.iscoroutine(t):
                    await t

        jobs = await browser_extract_jobs(page)
        cache.update(jobs)

        return ToolResponse(
            success=True,
            message=f"Successfully fetched {len(jobs)} live job matches.",
            data=[job.model_dump() for job in jobs],
        ).model_dump()

    except Exception as exc:
        logger.exception("Error in get_job_matches: %s", exc)
        return ToolResponse(
            success=False,
            message=f"Failed to fetch job matches: {exc}",
            error_code="FETCH_ERROR",
        ).model_dump()


@mcp.tool()
async def filter_jobs_by_preferences(
    tech_stack: Optional[list[str]] = None,
    work_mode: Optional[str] = None,
    location: Optional[str] = None,
    min_salary: Optional[int] = None,
    keywords: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
    cv_path: Optional[str] = None,
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
        ctx: FastMCP Context object.

    Returns:
        dict: ToolResponse with filtered and match-scored job listings.
    """
    cache = _get_cache(ctx)
    cached_jobs = cache.get_all()

    if not cached_jobs:
        return ToolResponse(
            success=False,
            message="No job listings found in cache. Please call 'get_job_matches' first to load jobs.",
            error_code="NO_CACHED_JOBS",
        ).model_dump()

    parsed_work_mode: Optional[WorkMode] = None
    if work_mode:
        mode_clean = work_mode.strip().lower()
        try:
            parsed_work_mode = WorkMode(mode_clean)
        except ValueError:
            logger.debug("Unknown work mode: '%s'. Proceeding with string match.", work_mode)

    prefs = JobPreferences(
        tech_stack=tech_stack or [],
        work_mode=parsed_work_mode,
        location=location,
        min_salary=min_salary,
        keywords=keywords or [],
        exclude_keywords=exclude_keywords or [],
        cv_path=cv_path,
    )

    try:
        filtered = filter_jobs(cached_jobs, prefs)
        return ToolResponse(
            success=True,
            message=f"Found {len(filtered)} matching jobs (out of {len(cached_jobs)} total).",
            data=[job.model_dump() for job in filtered],
        ).model_dump()
    except Exception as exc:
        logger.exception("Error in filter_jobs_by_preferences: %s", exc)
        return ToolResponse(
            success=False,
            message=f"Failed to filter jobs: {exc}",
            error_code="FILTER_ERROR",
        ).model_dump()


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
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return ToolResponse(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            ).model_dump()

        page = await session.get_page()
        await browser_bookmark_job(page, job_id)

        # Update cache if job present
        cache = _get_cache(ctx)
        cached_job = cache.get_by_id(job_id)
        if cached_job:
            cached_job.is_bookmarked = True

        return ToolResponse(
            success=True,
            message=f"Successfully bookmarked job '{job_id}'.",
            data={"job_id": job_id, "is_bookmarked": True},
        ).model_dump()

    except Exception as exc:
        logger.exception("Error in bookmark_job for '%s': %s", job_id, exc)
        return ToolResponse(
            success=False,
            message=f"Failed to bookmark job '{job_id}': {exc}",
            error_code="BOOKMARK_ERROR",
        ).model_dump()


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
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return ToolResponse(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            ).model_dump()

        page = await session.get_page()
        await browser_delete_job(page, job_id)

        # Update cache
        cache = _get_cache(ctx)
        updated_jobs = [j for j in cache.get_all() if j.job_id != job_id]
        cache.update(updated_jobs)

        return ToolResponse(
            success=True,
            message=f"Successfully dismissed/deleted job '{job_id}'.",
            data={"job_id": job_id},
        ).model_dump()

    except Exception as exc:
        logger.exception("Error in delete_job for '%s': %s", job_id, exc)
        return ToolResponse(
            success=False,
            message=f"Failed to delete job '{job_id}': {exc}",
            error_code="DELETE_ERROR",
        ).model_dump()


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
            return ToolResponse(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            ).model_dump()

        page = await session.get_page()
        preview = await browser_preview_application(page, job_id)

        # Store preview in pending applications store
        _pending_applications[job_id] = preview.model_dump()

        return ToolResponse(
            success=True,
            message=(
                f"Application preview generated for job '{job_id}' ({preview.job_title} at {preview.company}). "
                f"Please review the application fields and warnings carefully. "
                f"To submit the application, call 'confirm_auto_apply(job_id=\"{job_id}\")'."
            ),
            data=preview.model_dump(),
        ).model_dump()

    except Exception as exc:
        logger.exception("Error in auto_apply_job for '%s': %s", job_id, exc)
        return ToolResponse(
            success=False,
            message=f"Failed to generate application preview for job '{job_id}': {exc}",
            error_code="PREVIEW_ERROR",
        ).model_dump()


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
        return ToolResponse(
            success=False,
            message=(
                f"No pending application preview found for job '{job_id}'. "
                f"You MUST call 'auto_apply_job(job_id=\"{job_id}\")' first to preview and verify before confirming."
            ),
            error_code="NO_PENDING_PREVIEW",
        ).model_dump()

    try:
        session, is_healthy = await _ensure_session(ctx)
        if not is_healthy:
            return ToolResponse(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            ).model_dump()

        page = await session.get_page()
        await browser_execute_application(page, job_id)

        preview_details = _pending_applications.pop(job_id)

        return ToolResponse(
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
        ).model_dump()

    except Exception as exc:
        logger.exception("Error in confirm_auto_apply for '%s': %s", job_id, exc)
        return ToolResponse(
            success=False,
            message=f"Failed to submit application for job '{job_id}': {exc}",
            error_code="APPLY_EXECUTION_ERROR",
        ).model_dump()


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
            return ToolResponse(
                success=False,
                message="Browser session is unauthenticated. Please log in first.",
                error_code="UNAUTHENTICATED",
            ).model_dump()

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

        return ToolResponse(
            success=True,
            message=f"Calibrated {matched_count}/{total_count} selectors successfully.",
            data={
                "results": results,
                "matched_count": matched_count,
                "total_count": total_count,
            },
        ).model_dump()

    except Exception as exc:
        logger.exception("Error during selector calibration: %s", exc)
        return ToolResponse(
            success=False,
            message=f"Selector calibration failed: {exc}",
            error_code="CALIBRATION_ERROR",
        ).model_dump()

