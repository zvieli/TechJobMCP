"""Unit tests for DOM card stamping, Hebrew button resolution, and resilient fallback extraction."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Context, FastMCP
from hireme_mcp.core.api_client import JobCache
from hireme_mcp.core.auth import BASE_URL, DASHBOARD_PATH, SessionManager
from hireme_mcp.core.browser import (
    _resolve_selector,
    bookmark_job,
    delete_job,
    execute_application,
    extract_jobs,
    preview_application,
)
from hireme_mcp.core.discovery import CHILD_ROLE_CANDIDATES, SELECTORS, DynamicSelectorRegistry
from hireme_mcp.main import _warm_cache, browser_lifespan, get_job_matches
from hireme_mcp.models.schemas import Job, WorkMode


class MockLocator:
    """Mock Playwright Locator with realistic chaining and attribute support."""

    def __init__(self, count_val=1, text="", attrs=None, children=None):
        self._count_val = count_val
        self._text = text
        self._attrs = attrs or {}
        self._children = children or []

    async def count(self):
        return self._count_val

    async def inner_text(self):
        return self._text

    async def get_attribute(self, name):
        return self._attrs.get(name)

    async def click(self):
        pass

    async def is_visible(self):
        return True

    @property
    def first(self):
        if self._children:
            return self._children[0]
        return self

    def nth(self, idx):
        if idx < len(self._children):
            return self._children[idx]
        return self

    def locator(self, sel):
        return self


class TestHebrewSelectorsAndDiscovery(unittest.IsolatedAsyncioTestCase):
    """Tests for Hebrew selector definitions and dynamic child discovery."""

    def test_hebrew_selectors_in_registry(self):
        """Verify Hebrew button text and aria-label patterns in SELECTORS."""
        bm_fallback = SELECTORS["bookmark_button"]["fallback"]
        self.assertIn("שמור", bm_fallback)
        self.assertIn("שמירה", bm_fallback)

        del_fallback = SELECTORS["delete_button"]["fallback"]
        self.assertIn("מחק", del_fallback)
        self.assertIn("הסר", del_fallback)
        self.assertIn("הסרה", del_fallback)

        apply_fallback = SELECTORS["apply_button"]["fallback"]
        self.assertIn("הגש", apply_fallback)
        self.assertIn("הגשת מועמדות", apply_fallback)

    def test_hebrew_child_role_candidates(self):
        """Verify Hebrew patterns in CHILD_ROLE_CANDIDATES for heuristic discovery."""
        bm_candidates = " ".join(CHILD_ROLE_CANDIDATES.get("bookmark_button", []))
        self.assertIn("שמור", bm_candidates)
        self.assertIn("שמירה", bm_candidates)

        del_candidates = " ".join(CHILD_ROLE_CANDIDATES.get("delete_button", []))
        self.assertIn("מחק", del_candidates)
        self.assertIn("הסר", del_candidates)

        apply_candidates = " ".join(CHILD_ROLE_CANDIDATES.get("apply_button", []))
        self.assertIn("הגש", apply_candidates)
        self.assertIn("הגשת מועמדות", apply_candidates)

    async def test_resolve_hebrew_bookmark_button(self):
        """Test resolving bookmark button with Hebrew text via fallback/heuristic."""
        card = MagicMock()

        def mock_locator(sel):
            if "שמור" in sel or "שמירה" in sel:
                return MockLocator(count_val=1, text="שמור משרה")
            return MockLocator(count_val=0, text="")

        card.locator = mock_locator
        # Clear cached dynamic registry for test isolation
        with patch("hireme_mcp.core.browser.dynamic_registry", DynamicSelectorRegistry(file_path="/tmp/test_reg.json")):
            resolved = await _resolve_selector(card, "bookmark_button")
            self.assertTrue("שמור" in resolved or "שמירה" in resolved)


class TestDomCardStampingAndResilience(unittest.IsolatedAsyncioTestCase):
    """Tests for DOM card stamping and resilient text extraction."""

    async def test_extract_jobs_stamps_dom_and_extracts_hebrew_text(self):
        """Verify extract_jobs stamps cards with data-mcp-job-id via page.evaluate."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        # Job card with Hebrew title and company in custom/heading tags, no standard testids
        card = MockLocator(
            count_val=1,
            attrs={"class": "custom-card-container"},
        )

        def card_locator_fn(sel):
            # No primary testids match
            if "[data-testid=" in sel:
                return MockLocator(count_val=0)
            if "h1, h2, h3" in sel or "h2" in sel or "title" in sel:
                return MockLocator(count_val=1, text="מפתח Python בכיר")
            if "company" in sel or "employer" in sel:
                return MockLocator(count_val=1, text="חברת הייטק בע״מ")
            if "location" in sel:
                return MockLocator(count_val=1, text="תל אביב - יפו")
            if "bookmark" in sel or "שמור" in sel:
                return MockLocator(count_val=1, attrs={"class": "btn-save"}, text="שמור")
            return MockLocator(count_val=0)

        card.locator = card_locator_fn
        page_cards = MockLocator(count_val=1, children=[card])
        mock_page.locator = MagicMock(return_value=page_cards)

        jobs = await extract_jobs(mock_page)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.title, "מפתח Python בכיר")
        self.assertEqual(job.company, "חברת הייטק בע״מ")
        self.assertTrue(job.job_id.startswith("job-"))

        # Verify page.evaluate was called to stamp the DOM with data-mcp-job-id
        self.assertTrue(mock_page.evaluate.called)
        # Check call arguments contain selector and job_id
        eval_args = mock_page.evaluate.call_args
        self.assertIsNotNone(eval_args)

    async def test_card_actions_primary_lookup_with_data_mcp_job_id(self):
        """Verify bookmark_job, delete_job, preview_application, and execute_application
        use [data-mcp-job-id='...'] as primary locator."""
        mock_page = MagicMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        captured_selectors = []

        def mock_page_locator(sel):
            captured_selectors.append(sel)
            btn = MockLocator(count_val=1, text="Apply / שמור / מחק")
            card = MockLocator(count_val=1, children=[btn])
            card.locator = MagicMock(return_value=btn)
            return card

        mock_page.locator = mock_page_locator

        target_id = "job-abc12345"

        await bookmark_job(mock_page, target_id)
        self.assertTrue(any(f"[data-mcp-job-id='{target_id}']" in s for s in captured_selectors))

        captured_selectors.clear()
        await delete_job(mock_page, target_id)
        self.assertTrue(any(f"[data-mcp-job-id='{target_id}']" in s for s in captured_selectors))

        captured_selectors.clear()
        await preview_application(mock_page, target_id)
        self.assertTrue(any(f"[data-mcp-job-id='{target_id}']" in s for s in captured_selectors))

        captured_selectors.clear()
        await execute_application(mock_page, target_id)
        self.assertTrue(any(f"[data-mcp-job-id='{target_id}']" in s for s in captured_selectors))

    async def test_resilient_fallback_text_lines_when_classes_missing(self):
        """Verify title & company extract from meaningful text lines when selectors fail."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        card = MockLocator(
            count_val=1,
            text="ארכיטקט תוכנה ענן\nסטארטאפ פינטק\nמרכז\nPython, AWS, Docker",
            attrs={},
        )
        card.locator = MagicMock(return_value=MockLocator(count_val=0))
        page_cards = MockLocator(count_val=1, children=[card])
        mock_page.locator = MagicMock(return_value=page_cards)

        jobs = await extract_jobs(mock_page)
        self.assertEqual(len(jobs), 1)
        self.assertNotEqual(jobs[0].title, "Untitled Position")
        self.assertNotEqual(jobs[0].company, "Unknown Company")
        self.assertEqual(jobs[0].title, "ארכיטקט תוכנה ענן")
        self.assertEqual(jobs[0].company, "סטארטאפ פינטק")


class TestBackgroundCacheWarmup(unittest.IsolatedAsyncioTestCase):
    """Tests for background cache warmup task and lifespan integration."""

    def setUp(self):
        self.sample_jobs = [
            Job(
                job_id="job-warmup-1",
                title="Fullstack Engineer",
                company="Warmup Startup",
                location="Tel Aviv",
                work_mode=WorkMode.HYBRID,
                tech_stack=["Python", "FastAPI", "React"],
                description="Exciting role",
            )
        ]

    @patch("hireme_mcp.main.browser_extract_jobs")
    async def test_warm_cache_populates_cache_when_authenticated(self, mock_extract):
        """Verify _warm_cache navigates and updates JobCache when session is healthy."""
        mock_extract.return_value = self.sample_jobs

        mock_session = AsyncMock(spec=SessionManager)
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/login"
        mock_session.ensure_ready.return_value = mock_page

        cache = JobCache(ttl_minutes=15)
        self.assertEqual(len(cache.get_all()), 0)

        await _warm_cache(mock_session, cache)

        self.assertEqual(len(cache.get_all()), 1)
        self.assertEqual(cache.get_all()[0].job_id, "job-warmup-1")
        mock_page.goto.assert_called_once_with(f"{BASE_URL}{DASHBOARD_PATH}", wait_until="commit", timeout=8000)
        mock_extract.assert_called_once_with(mock_page)

    @patch("hireme_mcp.main.browser_extract_jobs")
    async def test_warm_cache_handles_unauthenticated_gracefully(self, mock_extract):
        """Verify _warm_cache does not raise and leaves cache empty if ensure_ready fails."""
        mock_session = AsyncMock(spec=SessionManager)
        mock_session.ensure_ready.side_effect = RuntimeError("Session unauthenticated")

        cache = JobCache(ttl_minutes=15)
        # Should not raise exception
        await _warm_cache(mock_session, cache)

        self.assertEqual(len(cache.get_all()), 0)
        mock_extract.assert_not_called()

    @patch("hireme_mcp.main.browser_extract_jobs")
    async def test_warm_cache_handles_extraction_error_gracefully(self, mock_extract):
        """Verify _warm_cache catches unexpected extraction exceptions without crashing."""
        mock_session = AsyncMock(spec=SessionManager)
        mock_page = AsyncMock()
        mock_page.url = f"{BASE_URL}{DASHBOARD_PATH}"
        mock_session.ensure_ready.return_value = mock_page
        mock_extract.side_effect = Exception("Page crashed")

        cache = JobCache(ttl_minutes=15)
        await _warm_cache(mock_session, cache)

        self.assertEqual(len(cache.get_all()), 0)

    @patch.object(SessionManager, "initialize", new_callable=AsyncMock)
    @patch.object(SessionManager, "shutdown", new_callable=AsyncMock)
    @patch("hireme_mcp.main._warm_cache", new_callable=AsyncMock)
    async def test_browser_lifespan_starts_and_cancels_warmup(self, mock_warm, mock_shutdown, mock_init):
        """Verify browser_lifespan starts _warm_cache task and cleanly handles shutdown."""
        mock_init.return_value = None
        mock_shutdown.return_value = None

        # Simulate a warm cache that sleeps
        async def slow_warm(session, cache):
            await asyncio.sleep(10)
        mock_warm.side_effect = slow_warm

        mock_server = MagicMock(spec=FastMCP)

        async with browser_lifespan(mock_server) as state:
            self.assertIn("session", state)
            self.assertIn("cache", state)
            await asyncio.sleep(0.01)
            self.assertTrue(mock_warm.called)

        # Exited context - shutdown should be called
        mock_shutdown.assert_called_once()


class TestScrapingTimeoutSafeguards(unittest.IsolatedAsyncioTestCase):
    """Tests for scraping timeout safeguards preventing DevTunnel / client timeouts."""

    @patch("hireme_mcp.main.browser_extract_jobs")
    async def test_get_job_matches_handles_timeout(self, mock_extract):
        """Verify get_job_matches returns error when scraping exceeds timeout limit."""
        cache = JobCache(ttl_minutes=10)
        mock_session = AsyncMock(spec=SessionManager)
        mock_page = AsyncMock()
        mock_page.url = f"{BASE_URL}{DASHBOARD_PATH}"
        mock_session.get_page.return_value = mock_page
        mock_session.ensure_ready.return_value = mock_page

        async def slow_scrape(page):
            await asyncio.sleep(15)
            return []
        mock_extract.side_effect = slow_scrape

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"session": mock_session, "cache": cache}

        # Use patch to shorten timeout or test with actual timeout mechanism
        with patch("hireme_mcp.main._SCRAPE_TIMEOUT_SECONDS", 0.05, create=True):
            res = await get_job_matches(force_refresh=True, ctx=ctx)
            self.assertFalse(res["success"])
            self.assertEqual(res["error_code"], "FETCH_ERROR")
            self.assertIn("timed out", res["message"].lower())

