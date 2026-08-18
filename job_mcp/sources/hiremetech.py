"""HireMeTech job source implementation."""

from __future__ import annotations

import asyncio
from typing import Optional

from job_mcp.core.api_client import fetch_jobs_via_api, filter_jobs
from job_mcp.core.auth import BASE_URL, DASHBOARD_PATH, SessionManager
from job_mcp.core.browser import (
    bookmark_job as browser_bookmark_job,
    extract_jobs as browser_extract_jobs,
)
from job_mcp.models.schemas import Job, JobPreferences
from job_mcp.sources.base import BaseJobSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)


class HireMeTechSource(BaseJobSource):
    """Job source implementation for HireMeTech platform."""

    source_id: str = "hiremetech"
    display_name: str = "HireMeTech"
    description: str = "HireMeTech AI job board and application platform"
    supports_bookmarks: bool = True
    supports_auto_apply: bool = True

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        """Initialize HireMeTechSource with optional session manager."""
        self.session_manager = session_manager or SessionManager()
        self._is_authenticated: Optional[bool] = None

    @property
    def is_authenticated(self) -> bool:
        """Check if source is currently authenticated."""
        if self._is_authenticated is not None:
            return self._is_authenticated
        if self.session_manager is not None:
            return self.session_manager.is_running
        return False

    @is_authenticated.setter
    def is_authenticated(self, value: bool) -> None:
        """Set authentication state."""
        self._is_authenticated = value

    async def check_health(self) -> bool:
        """Check health of the HireMeTech session.

        Returns:
            bool: True if session is valid and authenticated, False otherwise.
        """
        if not self.session_manager:
            self._is_authenticated = False
            return False
        try:
            is_healthy = await self.session_manager.check_session_health()
            self._is_authenticated = is_healthy
            return is_healthy
        except Exception as exc:
            logger.warning("HireMeTech health check failed: %s", exc)
            self._is_authenticated = False
            return False

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job matches from HireMeTech via Direct REST API with DOM fallback.

        Args:
            preferences: Optional JobPreferences for filtering.
            limit: Maximum number of jobs to fetch.

        Returns:
            list[Job]: List of Job listings tagged with 'hiremetech' source.
        """
        jobs: list[Job] = []

        # 1. Primary data source: Direct REST API via pure HTTP (zero browser launch, <40MB RAM)
        try:
            jobs = await asyncio.wait_for(fetch_jobs_via_api(None, size=limit), timeout=6.0)
        except Exception as exc:
            logger.debug("Direct pure HTTP API fetch failed or timed out: %s", exc)
            jobs = []

        # 2. Fallback: Browser DOM scraping (lazy-loads Playwright only if direct API fails or returns empty)
        if not jobs and self.session_manager:
            try:
                page = await self.session_manager.ensure_ready(max_retries=1)

                # Try API with browser session cookies / request context if available
                if hasattr(page, "request"):
                    try:
                        jobs = await asyncio.wait_for(
                            fetch_jobs_via_api(page.request, size=limit), timeout=2.5
                        )
                    except Exception as exc:
                        logger.debug("Browser-context API fetch failed: %s", exc)
                        jobs = []

                if not jobs:
                    target_url = f"{BASE_URL}{DASHBOARD_PATH}"
                    if hasattr(page, "url") and DASHBOARD_PATH not in (page.url or ""):
                        if hasattr(page, "goto") and callable(page.goto):
                            await page.goto(target_url, wait_until="commit", timeout=5000)
                        if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                            t = page.wait_for_timeout(1000)
                            if asyncio.iscoroutine(t):
                                await t

                    if hasattr(page, "request"):
                        jobs = await asyncio.wait_for(browser_extract_jobs(page), timeout=3.0)
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("DOM extraction fallback timed out.")
                jobs = []
            except Exception as exc:
                logger.warning("DOM extraction fallback failed: %s", exc)
                jobs = []

        # Ensure source tagging
        tagged_jobs: list[Job] = []
        for j in jobs:
            j.source = "hiremetech"
            if "hiremetech" not in j.sources:
                j.sources.insert(0, "hiremetech")
            tagged_jobs.append(j)

        # Apply preferences filtering if provided
        if preferences:
            tagged_jobs = filter_jobs(tagged_jobs, preferences)

        # Truncate to requested limit
        if limit and len(tagged_jobs) > limit:
            tagged_jobs = tagged_jobs[:limit]

        return tagged_jobs

    async def bookmark_job(self, job_id: str) -> bool:
        """Bookmark a job on HireMeTech.

        Args:
            job_id: ID of the job to bookmark.

        Returns:
            bool: True if bookmarking was successful.
        """
        page = await self.session_manager.get_page()
        return await browser_bookmark_job(page, job_id)
