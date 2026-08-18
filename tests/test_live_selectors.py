"""Tests for live production selectors and sibling-based job extraction."""

import unittest
from playwright.async_api import async_playwright

from job_mcp.core.browser import (
    JS_EXTRACT_ALL_JOBS,
    SELECTORS,
    _extract_jobs_via_locators,
    bookmark_job,
    dynamic_registry,
    extract_jobs,
    _resolve_selector,
)
from job_mcp.core.discovery import (
    CHILD_ROLE_CANDIDATES,
    calibrate_all_selectors,
    discover_child_selector,
)
from job_mcp.models.schemas import WorkMode


LIVE_CARD_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>HireMeTech Live Mock</title></head>
<body>
<main>
  <div class="rounded-2xl ring-1 ring-[var(--theme-border-light)] jobs-app-glass-surface shadow-ht-card p-4" data-id="job-live-1">
    <h4 class="font-bold text-gray-900 mb-1 text-sm">Fullstack AI Engineer</h4>
    <div class="text-xs text-gray-600">Acme Labs • Tel Aviv</div>
    <button class="group inline-flex items-center justify-center gap-2 px-4 py-2 min-h-[36px] text-white bg-ht-primary-500 rounded-xl text-xs font-semibold shadow-sm transition-colors">שמור</button>
  </div>
</main>
</body>
</html>
"""

RICH_CARD_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>HireMeTech Rich Card Mock</title></head>
<body>
<main>
  <div class="rounded-2xl jobs-app-glass-surface shadow-ht-card p-4" data-job-id="job-rich-99">
    <h4 class="font-bold text-gray-900 mb-1 text-sm">Senior Python AI Architect</h4>
    <div class="text-xs text-gray-600">Anthropic Labs • Tel Aviv - Hybrid</div>
    <p class="description text-xs text-gray-500 my-2">Design state of the art LLM agent architectures using FastMCP.</p>
    <div class="salary-range font-medium">35,000 - 45,000 ILS</div>
    <div class="flex gap-1 my-2">
      <span class="tech-badge bg-gray-100 rounded px-2 py-1 text-xs">Python</span>
      <span class="tech-badge bg-gray-100 rounded px-2 py-1 text-xs">FastMCP</span>
      <span class="tech-badge bg-gray-100 rounded px-2 py-1 text-xs">Playwright</span>
    </div>
    <a href="/jobs/job-rich-99" class="text-blue-600 underline">View Job Details</a>
    <button class="bg-ht-primary-500 saved active rounded-xl" aria-pressed="true">שמור</button>
  </div>
</main>
</body>
</html>
"""

LIVE_MULTI_CARDS_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>HireMeTech Multi Live Mock</title></head>
<body>
<main>
  <div class="rounded-2xl ring-1 jobs-app-glass-surface shadow-ht-card p-4" id="job-card-1">
    <h4 class="font-bold text-gray-900 mb-1 text-sm">Fullstack AI Engineer</h4>
    <div class="text-xs text-gray-600">Acme Labs • Tel Aviv</div>
    <button class="bg-ht-primary-500 rounded-xl">שמור</button>
  </div>
  <div class="rounded-2xl ring-1 jobs-app-glass-surface shadow-ht-card p-4" id="job-card-2">
    <h4 class="font-bold text-gray-900 mb-1 text-sm">Backend Team Lead</h4>
    <p class="text-xs text-gray-600">CyberShield | Herzliya</p>
    <button class="bg-ht-primary-500 rounded-xl">שמור</button>
  </div>
  <div class="rounded-2xl ring-1 jobs-app-glass-surface shadow-ht-card p-4" id="job-card-3">
    <h4 class="font-bold text-gray-900 mb-1 text-sm">DevOps Specialist</h4>
    <span class="text-xs text-gray-600">CloudFlow - Remote</span>
    <button class="bg-ht-primary-500 rounded-xl">שמור</button>
  </div>
</main>
</body>
</html>
"""


class TestLiveSelectorDefinitions(unittest.TestCase):
    """Verify live production selectors and child role candidates are registered."""

    def setUp(self):
        dynamic_registry.clear()

    def tearDown(self):
        dynamic_registry.clear()

    def test_job_card_selectors(self):
        primary = SELECTORS["job_card"]["primary"]
        fallback = SELECTORS["job_card"]["fallback"]
        self.assertIn("div.jobs-app-glass-surface", primary)
        self.assertIn("div.shadow-ht-card", primary)
        self.assertIn("jobs-app-glass-surface", fallback)
        self.assertIn("shadow-ht-card", fallback)

    def test_job_title_selectors(self):
        primary = SELECTORS["job_title"]["primary"]
        fallback = SELECTORS["job_title"]["fallback"]
        self.assertIn("h4.font-bold", primary)
        self.assertIn("text-gray-900", primary)
        self.assertIn("h4", fallback)

    def test_bookmark_button_selectors(self):
        primary = SELECTORS["bookmark_button"]["primary"]
        fallback = SELECTORS["bookmark_button"]["fallback"]
        self.assertIn("button.bg-ht-primary-500", primary)
        self.assertIn("bg-ht-primary", primary)
        self.assertIn("שמור", fallback)

    def test_child_role_candidates(self):
        self.assertIn("h4.font-bold", CHILD_ROLE_CANDIDATES.get("job_title", []))
        self.assertIn("button.bg-ht-primary-500", CHILD_ROLE_CANDIDATES.get("bookmark_button", []))


class TestLiveDOMInteraction(unittest.IsolatedAsyncioTestCase):
    """Test actual DOM interaction with Playwright on live HireMeTech HTML structure."""

    async def asyncSetUp(self):
        dynamic_registry.clear()
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def asyncTearDown(self):
        dynamic_registry.clear()
        await self.context.close()
        await self.browser.close()
        await self.pw.stop()

    async def test_extract_jobs_from_live_html(self):
        await self.page.set_content(LIVE_CARD_HTML)

        jobs = await extract_jobs(self.page)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.title, "Fullstack AI Engineer")
        self.assertEqual(job.company, "Acme Labs")
        self.assertEqual(job.location, "Tel Aviv")
        self.assertFalse(job.is_bookmarked)

    async def test_bookmark_job_live_html(self):
        await self.page.set_content(LIVE_CARD_HTML)

        jobs = await extract_jobs(self.page)
        self.assertEqual(len(jobs), 1)
        job_id = jobs[0].job_id

        # Track button click
        clicked = False
        async def on_click(route):
            nonlocal clicked
            clicked = True

        await self.page.expose_binding("recordClick", lambda source: True)
        await self.page.evaluate(
            """() => {
                document.querySelector('button').addEventListener('click', () => {
                    window.__button_clicked = true;
                });
            }"""
        )

        res = await bookmark_job(self.page, job_id)
        self.assertTrue(res)

        button_clicked = await self.page.evaluate("() => window.__button_clicked === true")
        self.assertTrue(button_clicked)

    async def test_extract_multi_jobs_with_various_separators(self):
        await self.page.set_content(LIVE_MULTI_CARDS_HTML)

        jobs = await extract_jobs(self.page)
        self.assertEqual(len(jobs), 3)

        # Card 1: bullet separator
        self.assertEqual(jobs[0].title, "Fullstack AI Engineer")
        self.assertEqual(jobs[0].company, "Acme Labs")
        self.assertEqual(jobs[0].location, "Tel Aviv")

        # Card 2: pipe separator
        self.assertEqual(jobs[1].title, "Backend Team Lead")
        self.assertEqual(jobs[1].company, "CyberShield")
        self.assertEqual(jobs[1].location, "Herzliya")

        # Card 3: dash separator & remote work mode
        self.assertEqual(jobs[2].title, "DevOps Specialist")
        self.assertEqual(jobs[2].company, "CloudFlow")
        self.assertEqual(jobs[2].location, "Remote")

    async def test_live_selector_calibration(self):
        await self.page.set_content(LIVE_CARD_HTML)

        report = await calibrate_all_selectors(self.page)
        self.assertIn("job_card", report)
        self.assertEqual(report["job_card"]["status"], "primary_matched")
        self.assertIn("job_title", report)
        self.assertEqual(report["job_title"]["status"], "primary_matched")
        self.assertIn("bookmark_button", report)
        self.assertEqual(report["bookmark_button"]["status"], "primary_matched")

    async def test_extract_jobs_single_pass_rich_fields(self):
        await self.page.set_content(RICH_CARD_HTML)

        jobs = await extract_jobs(self.page)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.job_id, "job-rich-99")
        self.assertEqual(job.title, "Senior Python AI Architect")
        self.assertEqual(job.company, "Anthropic Labs")
        self.assertIn("Tel Aviv", job.location)
        self.assertEqual(job.work_mode, WorkMode.HYBRID)
        self.assertEqual(job.salary_range, "35,000 - 45,000 ILS")
        self.assertIn("Python", job.tech_stack)
        self.assertIn("FastMCP", job.tech_stack)
        self.assertIn("Playwright", job.tech_stack)
        self.assertEqual(job.url, "https://hiremetech.com/jobs/job-rich-99")
        self.assertTrue(job.is_bookmarked)

        # Verify element was stamped with data-mcp-job-id in the DOM
        stamped_id = await self.page.evaluate(
            "() => document.querySelector('[data-job-id=\"job-rich-99\"]').getAttribute('data-mcp-job-id')"
        )
        self.assertEqual(stamped_id, "job-rich-99")

    async def test_extract_jobs_locator_fallback(self):
        await self.page.set_content(RICH_CARD_HTML)

        # Directly invoke locator fallback to verify parity
        jobs = await _extract_jobs_via_locators(
            self.page, "div.jobs-app-glass-surface, div.shadow-ht-card"
        )
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.job_id, "job-rich-99")
        self.assertEqual(job.title, "Senior Python AI Architect")
        self.assertEqual(job.company, "Anthropic Labs")
        self.assertEqual(job.salary_range, "35,000 - 45,000 ILS")
        self.assertTrue(job.is_bookmarked)

