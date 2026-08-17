"""Unit tests for hireme_mcp core modules."""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import zipfile

from hireme_mcp.core.api_client import (
    JobCache,
    extract_cv_keywords,
    filter_jobs,
)
from hireme_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    LOGIN_PATH,
    SessionManager,
)
from hireme_mcp.core.browser import (
    SELECTORS,
    _extract_tech_from_text,
    _parse_work_mode,
    _resolve_selector,
    bookmark_job,
    delete_job,
    extract_jobs,
    preview_application,
    execute_application,
)
from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode


class TestAuth(unittest.IsolatedAsyncioTestCase):
    """Tests for auth and session management."""

    def test_auth_constants(self):
        self.assertTrue(BASE_URL.startswith("https://hiremetech.com"))
        self.assertIn(DASHBOARD_PATH, ("/he-il/jobs-app", "/dashboard"))
        self.assertEqual(LOGIN_PATH, "/login")

    def test_session_manager_init_defaults(self):
        manager = SessionManager()
        self.assertIn(".hireme_mcp", str(manager.user_data_dir))
        self.assertTrue(manager.headless)

    def test_session_manager_custom_init(self):
        manager = SessionManager(user_data_dir="/tmp/test_profile", headless=False)
        self.assertEqual(str(manager.user_data_dir), "/tmp/test_profile")
        self.assertFalse(manager.headless)

    @patch("hireme_mcp.core.auth.async_playwright")
    async def test_session_manager_lifecycle(self, mock_async_playwright):
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)
        mock_context.pages = [mock_page]
        mock_pw.chromium.launch_persistent_context.return_value = mock_context

        # async_playwright() returns a PlaywrightContextManager whose .start() is an async coroutine returning mock_pw
        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw)
        mock_async_playwright.return_value = mock_cm

        manager = SessionManager(user_data_dir="/tmp/test_profile")
        await manager.initialize()
        self.assertTrue(manager._initialized)

        page = await manager.get_page()
        self.assertEqual(page, mock_page)

        # Test check_session_health - success
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.url = "https://hiremetech.com/dashboard"
        self.assertTrue(await manager.check_session_health())

        # Test check_session_health - redirected to login
        mock_page.url = "https://hiremetech.com/login"
        self.assertFalse(await manager.check_session_health())

        # Test check_session_health - 401 unauthorized
        mock_page.goto.return_value = MagicMock(status=401)
        mock_page.url = "https://hiremetech.com/dashboard"
        self.assertFalse(await manager.check_session_health())

        # Test inject_session_storage
        await manager.inject_session_storage({"token": "123", "state": {"a": 1}})
        mock_page.evaluate.assert_called_once()

        # Test shutdown
        await manager.shutdown()
        self.assertFalse(manager._initialized)
        self.assertIsNone(manager.page)


class MockLocator:
    """Helper class to mock Playwright Locator behavior accurately."""

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


class TestBrowser(unittest.IsolatedAsyncioTestCase):
    """Tests for browser selector registry and operations."""

    def test_selectors_registry(self):
        required_keys = [
            "job_card",
            "job_title",
            "job_company",
            "bookmark_button",
            "delete_button",
            "apply_button",
        ]
        for key in required_keys:
            self.assertIn(key, SELECTORS)
            self.assertIn("primary", SELECTORS[key])
            self.assertIn("fallback", SELECTORS[key])

    def test_helper_extractors(self):
        self.assertEqual(_parse_work_mode("Remote position in NY"), WorkMode.REMOTE)
        self.assertEqual(_parse_work_mode("Hybrid role 2 days in office"), WorkMode.HYBRID)
        self.assertEqual(_parse_work_mode("On-site in London"), WorkMode.ONSITE)
        self.assertIsNone(_parse_work_mode("Full time position"))

        techs = _extract_tech_from_text("Looking for Python, FastAPI, and Docker expertise.")
        self.assertIn("Python", techs)
        self.assertIn("FastAPI", techs)
        self.assertIn("Docker", techs)

    async def test_resolve_selector_primary(self):
        mock_page = MagicMock()
        mock_loc = MockLocator(count_val=1)
        mock_page.locator.return_value = mock_loc

        res = await _resolve_selector(mock_page, "job_card")
        self.assertEqual(res, SELECTORS["job_card"]["primary"])

    async def test_extract_jobs_mock(self):
        mock_page = MagicMock()

        card = MockLocator(
            count_val=1,
            attrs={"data-job-id": "job-101", "class": "job-card"},
        )
        # Custom locator routing for card
        def card_locator_fn(sel):
            if "job-title" in sel or "title" in sel:
                return MockLocator(count_val=1, text="Staff Backend Engineer")
            elif "company" in sel:
                return MockLocator(count_val=1, text="Google DeepMind")
            elif "location" in sel:
                return MockLocator(count_val=1, text="Remote")
            elif "tech-badge" in sel or "badge" in sel:
                b1 = MockLocator(count_val=1, text="Python")
                b2 = MockLocator(count_val=1, text="FastMCP")
                return MockLocator(count_val=2, children=[b1, b2])
            elif "bookmark" in sel:
                return MockLocator(count_val=1, attrs={"class": "bookmark-btn active saved"})
            elif "description" in sel:
                return MockLocator(count_val=1, text="Exciting role building AI infrastructure")
            return MockLocator(count_val=0, text="")

        card.locator = card_locator_fn
        page_cards = MockLocator(count_val=1, children=[card])
        mock_page.locator = MagicMock(return_value=page_cards)

        jobs = await extract_jobs(mock_page)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "job-101")
        self.assertEqual(jobs[0].title, "Staff Backend Engineer")
        self.assertEqual(jobs[0].company, "Google DeepMind")
        self.assertEqual(jobs[0].work_mode, WorkMode.REMOTE)
        self.assertTrue(jobs[0].is_bookmarked)

    async def test_bookmark_and_delete_job(self):
        mock_page = MagicMock()
        btn = MockLocator(count_val=1)
        card = MockLocator(count_val=1, children=[btn])
        card.locator = MagicMock(return_value=btn)
        mock_page.locator = MagicMock(return_value=card)

        self.assertTrue(await bookmark_job(mock_page, "job-101"))
        self.assertTrue(await delete_job(mock_page, "job-101"))

    async def test_preview_and_execute_application(self):
        mock_page = MagicMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.keyboard.press = AsyncMock()

        btn = MockLocator(count_val=1, text="Apply")
        card = MockLocator(count_val=1, children=[btn])
        card.locator = MagicMock(return_value=btn)
        mock_page.locator = MagicMock(return_value=card)

        preview = await preview_application(mock_page, "job-101")
        self.assertEqual(preview.job_id, "job-101")
        self.assertEqual(preview.application_method, "direct_submission")

        executed = await execute_application(mock_page, "job-101")
        self.assertTrue(executed)


class TestApiClient(unittest.TestCase):
    """Tests for cache, keyword extraction, and job filtering."""

    def test_job_cache(self):
        cache = JobCache(ttl_minutes=1)
        self.assertTrue(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 0)

        sample_jobs = [
            Job(job_id="j1", title="Backend Engineer", company="Acme", tech_stack=["Python", "FastAPI"]),
            Job(job_id="j2", title="Frontend Engineer", company="Beta", tech_stack=["React", "TypeScript"]),
        ]
        cache.update(sample_jobs)
        self.assertFalse(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 2)
        self.assertEqual(cache.get_by_id("j1").title, "Backend Engineer")
        self.assertIsNone(cache.get_by_id("j999"))

        cache.clear()
        self.assertTrue(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 0)

    def test_extract_cv_keywords_txt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Experienced Senior Software Engineer skilled in Python, Kubernetes, PostgreSQL, and AWS.")
            temp_path = f.name

        try:
            keywords = extract_cv_keywords(temp_path)
            self.assertIn("Python", keywords)
            self.assertIn("Kubernetes", keywords)
            self.assertIn("PostgreSQL", keywords)
            self.assertIn("AWS", keywords)
        finally:
            os.unlink(temp_path)

    def test_extract_cv_keywords_docx(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            docx_path = f.name

        try:
            # Build valid minimal docx zip
            xml_content = b'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Proficient in TypeScript, React, Docker, and Redis.</w:t></w:r></w:p></w:body></w:document>'
            with zipfile.ZipFile(docx_path, "w") as z:
                z.writestr("word/document.xml", xml_content)

            keywords = extract_cv_keywords(docx_path)
            self.assertIn("TypeScript", keywords)
            self.assertIn("React", keywords)
            self.assertIn("Docker", keywords)
            self.assertIn("Redis", keywords)
        finally:
            os.unlink(docx_path)

    def test_filter_jobs(self):
        jobs = [
            Job(
                job_id="1",
                title="Senior Python Developer",
                company="Alpha",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
                description="Build scalable microservices with Python and FastAPI.",
            ),
            Job(
                job_id="2",
                title="Frontend Developer",
                company="Beta",
                location="New York, NY",
                work_mode=WorkMode.HYBRID,
                tech_stack=["React", "TypeScript", "Next.js"],
                description="Frontend UI development in Next.js.",
            ),
            Job(
                job_id="3",
                title="Legacy PHP Developer",
                company="Gamma",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["PHP", "MySQL"],
                description="Maintain legacy PHP codebase.",
            ),
        ]

        # 1. Test exclude keywords
        prefs = JobPreferences(
            tech_stack=["Python"],
            exclude_keywords=["PHP", "Legacy"],
        )
        filtered = filter_jobs(jobs, prefs)
        ids = [j.job_id for j in filtered]
        self.assertIn("1", ids)
        self.assertNotIn("3", ids)

        # 2. Test ranking and match score
        prefs_python = JobPreferences(
            tech_stack=["Python", "FastAPI", "Docker"],
            work_mode=WorkMode.REMOTE,
        )
        scored = filter_jobs(jobs, prefs_python)
        self.assertEqual(scored[0].job_id, "1")
        self.assertGreater(scored[0].match_score, 80.0)


if __name__ == "__main__":
    unittest.main()
