"""Unit and integration tests for HireMeTech MCP tools."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Context

from job_mcp.core.api_client import (
    JobCache,
    extract_candidate_profile,
    fetch_jobs_via_api,
    filter_jobs,
)
from job_mcp.core.application import (
    ApplicationLedger,
    HybridApplicationDispatcher,
)
from job_mcp.core.auth import SessionManager
from job_mcp.main import (
    _ensure_session,
    _get_cache,
    _pending_applications,
    auto_apply_job,
    bookmark_job,
    confirm_auto_apply,
    delete_job,
    filter_jobs_by_preferences,
    get_job_matches,
    run_job_scout,
)
from job_mcp.models.schemas import (
    ApplicationPreview,
    Job,
    WorkMode,
)
from job_mcp.sources import JobAggregator, SourceRegistry
from job_mcp.sources.hiremetech import HireMeTechSource


class TestMcpTools(unittest.IsolatedAsyncioTestCase):
    """Test suite for FastMCP tools implementation."""

    def setUp(self):
        """Reset pending applications and setup test data before each test."""
        _pending_applications.clear()
        self.mock_jobs = [
            Job(
                job_id="job-1",
                title="Senior Backend Python Developer",
                company="TechCorp",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
                description="We need an experienced Python engineer for scalable cloud services.",
                salary_range="$140,000 - $160,000",
                is_bookmarked=False,
            ),
            Job(
                job_id="job-2",
                title="Lead Frontend Engineer",
                company="WebDev Inc",
                location="New York, NY",
                work_mode=WorkMode.HYBRID,
                tech_stack=["React", "TypeScript", "Next.js", "TailwindCSS"],
                description="Building next-generation frontend applications.",
                salary_range="$150,000",
                is_bookmarked=False,
            ),
            Job(
                job_id="job-3",
                title="Legacy PHP Maintenance",
                company="OldTech Co",
                location="Austin, TX",
                work_mode=WorkMode.ONSITE,
                tech_stack=["PHP", "MySQL"],
                description="Legacy code maintenance.",
                salary_range="$80,000",
                is_bookmarked=False,
            ),
        ]

    def _create_mock_context(self, session_mgr: SessionManager, cache: JobCache) -> MagicMock:
        """Create a mock FastMCP Context with lifespan state."""
        registry = SourceRegistry()
        registry.register(HireMeTechSource(session_manager=session_mgr))
        aggregator = JobAggregator(registry=registry, cache=cache)
        ledger = ApplicationLedger(db_path=":memory:")
        dispatcher = HybridApplicationDispatcher(ledger=ledger, session_manager=session_mgr)
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {
            "session": session_mgr,
            "cache": cache,
            "registry": registry,
            "aggregator": aggregator,
            "ledger": ledger,
            "dispatcher": dispatcher,
        }
        return ctx

    @patch("job_mcp.sources.hiremetech.browser_extract_jobs")
    async def test_get_job_matches_cache_hit(self, mock_extract):
        """Test get_job_matches returns cached data without calling browser extraction."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        res = await get_job_matches(force_refresh=False, ctx=ctx)

        self.assertTrue(res["success"])
        self.assertIn("Retrieved 3 cached", res["message"])
        self.assertEqual(len(res["data"]), 3)
        mock_extract.assert_not_called()

    @patch("job_mcp.sources.hiremetech.browser_extract_jobs")
    async def test_get_job_matches_stale_cache_returns_cached_immediately(self, mock_extract):
        """Test get_job_matches returns cached data even if stale when force_refresh=False."""
        cache = JobCache(ttl_minutes=0)  # Immediately stale
        cache.update(self.mock_jobs)
        self.assertTrue(cache.is_stale)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        res = await get_job_matches(force_refresh=False, ctx=ctx)

        self.assertTrue(res["success"])
        self.assertIn("Retrieved 3 cached", res["message"])
        self.assertEqual(len(res["data"]), 3)
        mock_extract.assert_not_called()

    @patch("job_mcp.sources.hiremetech.fetch_jobs_via_api")
    @patch("job_mcp.sources.hiremetech.browser_extract_jobs")
    async def test_get_job_matches_live_fetch_and_force_refresh(self, mock_extract, mock_api):
        """Test live extraction when cache is empty or force_refresh is True."""
        mock_api.side_effect = RuntimeError("API unavailable")
        cache = JobCache(ttl_minutes=10)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        mock_session.ensure_ready = AsyncMock(return_value=AsyncMock())
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/login"
        mock_session.get_page.return_value = mock_page

        mock_extract.return_value = self.mock_jobs
        ctx = self._create_mock_context(mock_session, cache)

        # 1. First fetch (cache empty)
        res = await get_job_matches(force_refresh=False, ctx=ctx)
        self.assertTrue(res["success"])
        self.assertIn("Successfully fetched 3 live", res["message"])
        self.assertEqual(len(res["data"]), 3)
        self.assertEqual(len(cache.get_all()), 3)
        mock_extract.assert_called_once()

        # 2. Second fetch with force_refresh=True
        mock_extract.reset_mock()
        res2 = await get_job_matches(force_refresh=True, ctx=ctx)
        self.assertTrue(res2["success"])
        mock_extract.assert_called_once()

    @patch("job_mcp.sources.hiremetech.fetch_jobs_via_api")
    @patch("job_mcp.sources.hiremetech.browser_extract_jobs")
    async def test_get_job_matches_api_first(self, mock_extract, mock_api):
        """Test get_job_matches uses API data when available and avoids DOM extraction."""
        cache = JobCache(ttl_minutes=10)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        mock_session.ensure_ready = AsyncMock(return_value=AsyncMock())
        mock_page = AsyncMock()
        mock_session.get_page.return_value = mock_page

        mock_api.return_value = self.mock_jobs
        ctx = self._create_mock_context(mock_session, cache)

        res = await get_job_matches(force_refresh=True, ctx=ctx)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 3)
        mock_api.assert_called_once()
        mock_extract.assert_not_called()

    @patch("job_mcp.sources.hiremetech.browser_extract_jobs")
    async def test_get_job_matches_unauthenticated(self, mock_extract):
        """Test get_job_matches returns UNAUTHENTICATED error when session is invalid."""
        cache = JobCache(ttl_minutes=10)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = False
        mock_session.ensure_ready.side_effect = RuntimeError("Session unauthenticated")
        ctx = self._create_mock_context(mock_session, cache)

        res = await get_job_matches(force_refresh=True, ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "UNAUTHENTICATED")
        mock_extract.assert_not_called()

    async def test_filter_jobs_no_cached_jobs(self):
        """Test filter_jobs_by_preferences returns error if cache has no jobs."""
        cache = JobCache(ttl_minutes=10)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        res = await filter_jobs_by_preferences(tech_stack=["Python"], ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "NO_CACHED_JOBS")

    async def test_filter_jobs_by_stack_and_work_mode(self):
        """Test filtering jobs by tech stack, work mode, and exclusion."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        # Filter for Python & remote, excluding PHP
        res = await filter_jobs_by_preferences(
            tech_stack=["Python", "FastAPI"],
            work_mode="remote",
            exclude_keywords=["PHP", "Legacy"],
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        data = res["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["job_id"], "job-1")
        self.assertEqual(data[0]["company"], "TechCorp")
        self.assertGreater(data[0]["match_score"], 80.0)

    async def test_filter_jobs_by_cv_file(self):
        """Test filtering jobs with CV file keyword extraction."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Fullstack developer proficient in React, TypeScript, and Next.js")
            cv_file = f.name

        try:
            res = await filter_jobs_by_preferences(cv_path=cv_file, ctx=ctx)
            self.assertTrue(res["success"])
            data = res["data"]
            self.assertGreater(len(data), 0)
            self.assertEqual(data[0]["job_id"], "job-2")
        finally:
            os.unlink(cv_file)

    @patch("job_mcp.main.browser_bookmark_job")
    async def test_bookmark_job_flow(self, mock_browser_bookmark):
        """Test bookmark_job updates browser and job cache."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        mock_page = AsyncMock()
        mock_session.get_page.return_value = mock_page
        mock_browser_bookmark.return_value = True
        ctx = self._create_mock_context(mock_session, cache)

        res = await bookmark_job("job-1", ctx=ctx)
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "job-1")
        self.assertTrue(res["data"]["is_bookmarked"])
        self.assertTrue(cache.get_by_id("job-1").is_bookmarked)
        mock_browser_bookmark.assert_called_once_with(mock_page, "job-1")

    @patch("job_mcp.main.browser_delete_job")
    async def test_delete_job_flow(self, mock_browser_delete):
        """Test delete_job dismisses job on page and removes from cache."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        mock_page = AsyncMock()
        mock_session.get_page.return_value = mock_page
        mock_browser_delete.return_value = True
        ctx = self._create_mock_context(mock_session, cache)

        res = await delete_job("job-3", ctx=ctx)
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "job-3")
        self.assertIsNone(cache.get_by_id("job-3"))
        self.assertEqual(len(cache.get_all()), 2)
        mock_browser_delete.assert_called_once_with(mock_page, "job-3")

    async def test_two_step_auto_apply_flow(self):
        """Test complete 2-step apply workflow: auto_apply_job then confirm_auto_apply."""
        self.mock_jobs[0].match_score = 95.0
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        ctx = self._create_mock_context(mock_session, cache)

        with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
            # Step 1: Preview application
            res1 = await auto_apply_job("job-1", ctx=ctx)
            self.assertTrue(res1["success"])
            self.assertIn("Application preview generated", res1["message"])
            self.assertIn("job-1", _pending_applications)
            self.assertEqual(res1["data"]["job_title"], "Senior Backend Python Developer")

            # Step 2: Confirm application
            res2 = await confirm_auto_apply("job-1", ctx=ctx)
            self.assertTrue(res2["success"])
            self.assertIn("Successfully submitted application", res2["message"])
            self.assertTrue(res2["data"]["submitted"])
            self.assertNotIn("job-1", _pending_applications)

    async def test_run_job_scout_basic(self):
        """Test run_job_scout discovers, filters, bookmarks, and returns structured tracking metrics."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        ctx = self._create_mock_context(mock_session, cache)

        res = await run_job_scout(
            tech_stack=["Python", "FastAPI"],
            top_tier_threshold=85,
            strong_match_threshold=60,
            disqualify_threshold=40,
            auto_apply=False,
            auto_bookmark=True,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        data = res["data"]
        self.assertEqual(data["mcp_status"], "success")
        self.assertFalse(data["fallback_used"])
        self.assertIn("mcp_tracking_ids", data)
        self.assertIn("submitted", data)
        self.assertIn("bookmarked", data)
        self.assertIn("removed_from_cache", data)
        self.assertIn("blocked", data)
        self.assertIn("deferred", data)
        self.assertIn("failed", data)
        self.assertIn("summary_text", data)
        self.assertGreater(data["total_fetched"], 0)
        self.assertTrue(len(data["bookmarked"]) > 0)

    @patch("job_mcp.main.browser_preview_application")
    @patch("job_mcp.main.browser_execute_application")
    async def test_run_job_scout_with_auto_apply(self, mock_execute, mock_preview):
        """Test run_job_scout with auto_apply=True applies to top tier jobs."""
        cache = JobCache(ttl_minutes=10)
        cache.update(self.mock_jobs)
        mock_session = AsyncMock(spec=SessionManager)
        mock_session._initialized = True
        mock_session.check_session_health.return_value = True
        mock_page = AsyncMock()
        mock_session.get_page.return_value = mock_page
        ctx = self._create_mock_context(mock_session, cache)

        mock_preview.return_value = ApplicationPreview(
            job_id="job-1",
            job_title="Senior Backend Python Developer",
            company="TechCorp",
            application_method="direct_submission",
            fields_to_submit={"full_name": "Alex Rivera", "email": "candidate@example.com"},
            warnings=[],
        )
        mock_execute.return_value = True

        res = await run_job_scout(
            tech_stack=["Python", "FastAPI"],
            top_tier_threshold=80,
            auto_apply=True,
            auto_bookmark=True,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        data = res["data"]
        self.assertIn("job-1", data["submitted"])


if __name__ == "__main__":
    unittest.main()
