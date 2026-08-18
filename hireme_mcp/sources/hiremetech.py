"""HireMeTech job source implementation."""

from __future__ import annotations

import asyncio
from typing import Optional

from hireme_mcp.core.api_client import fetch_jobs_via_api, filter_jobs
from hireme_mcp.core.auth import BASE_URL, DASHBOARD_PATH, SessionManager
from hireme_mcp.core.browser import (
    bookmark_job as browser_bookmark_job,
    extract_jobs as browser_extract_jobs,
)
from hireme_mcp.models.schemas import Job, JobPreferences
from hireme_mcp.sources.base import BaseJobSource
from hireme_mcp.utils.logger import get_logger

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
            return bool(self.session_manager._initialized and self.session_manager.context is not None)
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
        """Fetch job matches from HireMeTech via API with DOM fallback.

        Args:
            preferences: Optional JobPreferences for filtering.
            limit: Maximum number of jobs to fetch.

        Returns:
            list[Job]: List of Job listings tagged with 'hiremetech' source.
        """
        page = await self.session_manager.get_page()
        jobs: list[Job] = []

        # 1. Primary data source: Direct API fetch (~100-200ms)
        try:
            jobs = await fetch_jobs_via_api(page.request, size=limit)
        except Exception as exc:
            logger.debug("Direct API fetch failed or timed out, falling back to DOM scraping: %s", exc)

        # 2. Fallback: Browser DOM scraping
        if not jobs:
            try:
                target_url = f"{BASE_URL}{DASHBOARD_PATH}"
                if DASHBOARD_PATH not in (page.url or ""):
                    await page.goto(target_url, wait_until="commit", timeout=10000)
                    if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                        t = page.wait_for_timeout(2500)
                        if asyncio.iscoroutine(t):
                            await t

                jobs = await browser_extract_jobs(page)
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
