"""Unit and integration tests for Adaptive DOM Discovery and Dynamic Selector Registry."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Context

from hireme_mcp.core.auth import SessionManager
from hireme_mcp.core.browser import (
    SELECTORS,
    _resolve_selector,
    dynamic_registry,
)
from hireme_mcp.core.discovery import (
    CHILD_ROLE_CANDIDATES,
    DEFAULT_DYNAMIC_SELECTORS_PATH,
    DynamicSelectorRegistry,
    calibrate_all_selectors,
    discover_card_selector,
    discover_child_selector,
)
from hireme_mcp.main import calibrate_selectors


class MockLocator:
    """Helper mock for Playwright Locator."""

    def __init__(
        self,
        count_val: int = 1,
        text: str = "",
        attrs: dict = None,
        children: list = None,
        router=None,
    ):
        self._count_val = count_val
        self._text = text
        self._attrs = attrs or {}
        self._children = children or []
        self._router = router

    async def count(self) -> int:
        return self._count_val

    async def inner_text(self) -> str:
        return self._text

    async def get_attribute(self, name: str):
        return self._attrs.get(name)

    @property
    def first(self):
        if self._children:
            return self._children[0]
        return self

    def nth(self, idx: int):
        if idx < len(self._children):
            return self._children[idx]
        return self

    def locator(self, sel: str):
        if self._router:
            return self._router(sel)
        return self


class TestDynamicSelectorRegistry(unittest.TestCase):
    """Tests for DynamicSelectorRegistry loading, saving, and querying."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test_selectors.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_and_env_path_resolution(self):
        # Default path
        reg_default = DynamicSelectorRegistry()
        self.assertIn(".hireme_mcp", str(reg_default.file_path))

        # Env path
        with patch.dict(os.environ, {"DYNAMIC_SELECTORS_PATH": str(self.test_file)}):
            reg_env = DynamicSelectorRegistry()
            self.assertEqual(reg_env.file_path, self.test_file.resolve())

        # Explicit path
        reg_explicit = DynamicSelectorRegistry(file_path=self.test_file)
        self.assertEqual(reg_explicit.file_path, self.test_file.resolve())

    def test_get_set_get_all_clear(self):
        reg = DynamicSelectorRegistry(file_path=self.test_file)
        self.assertIsNone(reg.get("job_card"))
        self.assertEqual(reg.get_all(), {})

        reg.set("job_card", "div.custom-job-card")
        self.assertEqual(reg.get("job_card"), "div.custom-job-card")
        self.assertEqual(reg.get_all(), {"job_card": "div.custom-job-card"})

        # Reload in new instance
        reg2 = DynamicSelectorRegistry(file_path=self.test_file)
        self.assertEqual(reg2.get("job_card"), "div.custom-job-card")

        reg2.clear()
        self.assertIsNone(reg2.get("job_card"))
        self.assertEqual(reg2.get_all(), {})

    def test_corrupted_or_invalid_json_handling(self):
        # Corrupted JSON
        self.test_file.write_text("{corrupt json content", encoding="utf-8")
        reg = DynamicSelectorRegistry(file_path=self.test_file)
        self.assertEqual(reg.get_all(), {})

        # Non-dict JSON (e.g., list)
        self.test_file.write_text("[\"item1\", \"item2\"]", encoding="utf-8")
        reg2 = DynamicSelectorRegistry(file_path=self.test_file)
        self.assertEqual(reg2.get_all(), {})

    def test_save_error_handling(self):
        reg = DynamicSelectorRegistry(file_path=self.test_file)
        with patch("builtins.open", side_effect=PermissionError("Mock Permission Denied")):
            # Should not raise exception
            reg.set("key", "val")


class TestHeuristicDiscovery(unittest.IsolatedAsyncioTestCase):
    """Tests for DOM heuristic discovery functions."""

    async def test_discover_card_selector_success(self):
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value="div.detected-job-card")
        mock_loc = MockLocator(count_val=4)
        mock_page.locator.return_value = mock_loc

        result = await discover_card_selector(mock_page)
        self.assertEqual(result, "div.detected-job-card")
        mock_page.evaluate.assert_called_once()

    async def test_discover_card_selector_none_or_error(self):
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value=None)
        self.assertIsNone(await discover_card_selector(mock_page))

        mock_page.evaluate = AsyncMock(side_effect=Exception("JS Evaluation Failed"))
        self.assertIsNone(await discover_card_selector(mock_page))

    async def test_discover_child_selector_matches(self):
        mock_card = MagicMock()

        # Mock behavior where candidate matches
        def mock_locator_fn(sel):
            if "job-title" in sel or "h2" in sel:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        mock_card.locator = mock_locator_fn

        discovered = await discover_child_selector(mock_card, "job_title")
        self.assertIsNotNone(discovered)
        self.assertTrue("title" in discovered or "h2" in discovered)

    async def test_discover_child_selector_no_match(self):
        mock_card = MagicMock()
        mock_card.locator = MagicMock(return_value=MockLocator(count_val=0))

        discovered = await discover_child_selector(mock_card, "unknown_role")
        self.assertIsNone(discovered)


class TestFourTierSelectorResolution(unittest.IsolatedAsyncioTestCase):
    """Tests for 4-tier selector resolution in _resolve_selector."""

    def setUp(self):
        dynamic_registry.clear()

    def tearDown(self):
        dynamic_registry.clear()

    async def test_tier_1_dynamic_registry_hit(self):
        mock_page = MagicMock()
        mock_page.locator.return_value = MockLocator(count_val=1)

        dynamic_registry.set("job_card", "section.cached-card")
        resolved = await _resolve_selector(mock_page, "job_card")
        self.assertEqual(resolved, "section.cached-card")

    async def test_tier_2_primary_hit(self):
        mock_page = MagicMock()
        mock_page.locator.return_value = MockLocator(count_val=1)

        resolved = await _resolve_selector(mock_page, "job_title")
        self.assertEqual(resolved, SELECTORS["job_title"]["primary"])

    async def test_tier_3_fallback_hit(self):
        mock_page = MagicMock()

        def locator_fn(sel):
            # Primary fails (count=0), fallback matches
            if sel == SELECTORS["job_title"]["primary"]:
                return MockLocator(count_val=0)
            if "h2.job-title" in sel or "title" in sel:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        mock_page.locator = locator_fn
        resolved = await _resolve_selector(mock_page, "job_title")
        self.assertIn("title", resolved)
        # Should be saved into dynamic registry
        self.assertEqual(dynamic_registry.get("job_title"), resolved)

    async def test_tier_4_heuristic_discovery_hit(self):
        mock_page = MagicMock()

        # Everything in primary and fallback fails
        def locator_fn(sel):
            if "detected" in sel:
                return MockLocator(count_val=3)
            return MockLocator(count_val=0)

        mock_page.locator = locator_fn
        mock_page.evaluate = AsyncMock(return_value="div.detected-card")

        resolved = await _resolve_selector(mock_page, "job_card")
        self.assertEqual(resolved, "div.detected-card")
        self.assertEqual(dynamic_registry.get("job_card"), "div.detected-card")

    async def test_all_tiers_failed_raises_value_error(self):
        mock_page = MagicMock()
        mock_page.locator.return_value = MockLocator(count_val=0)
        mock_page.evaluate = AsyncMock(return_value=None)

        with self.assertRaises(ValueError) as ctx:
            await _resolve_selector(mock_page, "job_card")
        self.assertIn("Failed to resolve selector for 'job_card'", str(ctx.exception))

    async def test_tier_3_combined_fallback_hit(self):
        mock_page = MagicMock()

        def locator_fn(sel):
            if sel == SELECTORS["job_company"]["primary"]:
                return MockLocator(count_val=0)
            if sel == SELECTORS["job_company"]["fallback"]:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        mock_page.locator = locator_fn
        resolved = await _resolve_selector(mock_page, "job_company")
        self.assertEqual(resolved, SELECTORS["job_company"]["fallback"])
        self.assertEqual(dynamic_registry.get("job_company"), SELECTORS["job_company"]["fallback"])

    async def test_tier_4_heuristic_child_discovery_hit(self):
        mock_card = MagicMock()

        # Primary and fallback fail, heuristic matches [data-testid*='bookmark']
        def locator_fn(sel):
            if "aria-label*='bookmark'" in sel:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        mock_card.locator = locator_fn
        resolved = await _resolve_selector(mock_card, "bookmark_button")
        self.assertIn("bookmark", resolved)
        self.assertEqual(dynamic_registry.get("bookmark_button"), resolved)

    async def test_unregistered_key_heuristic_discovery(self):
        mock_card = MagicMock()

        def locator_fn(sel):
            if "apply" in sel:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        mock_card.locator = locator_fn
        resolved = await _resolve_selector(mock_card, "apply_button")
        self.assertIn("apply", resolved)

    async def test_unregistered_key_raw_fallback(self):
        mock_page = MagicMock()
        mock_page.locator.return_value = MockLocator(count_val=1)

        resolved = await _resolve_selector(mock_page, "button.custom-btn")
        self.assertEqual(resolved, "button.custom-btn")


class TestCalibrateAllSelectors(unittest.IsolatedAsyncioTestCase):
    """Tests for calibrate_all_selectors functionality."""

    async def test_calibrate_all_selectors_mixed_results(self):
        mock_page = MagicMock()
        test_registry = DynamicSelectorRegistry(file_path=tempfile.mktemp())

        def card_locator_fn(sel):
            if sel == SELECTORS["job_title"]["primary"]:
                return MockLocator(count_val=1)
            elif sel == ".favorite-btn":
                return MockLocator(count_val=1)
            elif "Apply" in sel:
                return MockLocator(count_val=1)
            return MockLocator(count_val=0)

        def page_locator_fn(sel):
            if sel == SELECTORS["job_card"]["primary"]:
                return MockLocator(count_val=5, router=card_locator_fn)
            return card_locator_fn(sel)

        mock_page.locator = page_locator_fn
        mock_page.evaluate = AsyncMock(return_value=None)

        report = await calibrate_all_selectors(mock_page, test_registry)
        self.assertIn("job_card", report)
        self.assertEqual(report["job_card"]["status"], "primary_matched")
        self.assertEqual(report["job_card"]["count"], 5)
        self.assertEqual(test_registry.get("job_card"), SELECTORS["job_card"]["primary"])

        self.assertIn("job_title", report)
        self.assertEqual(report["job_title"]["status"], "primary_matched")

        self.assertIn("bookmark_button", report)
        self.assertEqual(report["bookmark_button"]["status"], "fallback_matched")
        self.assertEqual(report["bookmark_button"]["selector"], ".favorite-btn")

    async def test_calibrate_all_selectors_discovered_and_failed(self):
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value="section.custom-job-card")
        test_registry = DynamicSelectorRegistry(file_path=tempfile.mktemp())

        def page_locator_fn(sel):
            if sel == "section.custom-job-card":
                return MockLocator(count_val=4)
            return MockLocator(count_val=0)

        mock_page.locator = page_locator_fn

        report = await calibrate_all_selectors(mock_page, test_registry)
        self.assertEqual(report["job_card"]["status"], "discovered_heuristic")
        self.assertEqual(report["job_card"]["selector"], "section.custom-job-card")
        self.assertEqual(report["job_card"]["count"], 4)
        self.assertEqual(test_registry.get("job_card"), "section.custom-job-card")


class TestCalibrateSelectorsTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the calibrate_selectors MCP tool handler."""

    def setUp(self):
        dynamic_registry.clear()

    def tearDown(self):
        dynamic_registry.clear()

    @patch("hireme_mcp.main._ensure_session")
    async def test_calibrate_selectors_unauthenticated(self, mock_ensure_session):
        mock_ensure_session.return_value = (MagicMock(), False)
        resp = await calibrate_selectors()
        self.assertFalse(resp["success"])
        self.assertEqual(resp["error_code"], "UNAUTHENTICATED")

    @patch("hireme_mcp.main._ensure_session")
    @patch("hireme_mcp.main.calibrate_all_selectors")
    async def test_calibrate_selectors_success(self, mock_calibrate, mock_ensure_session):
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://hiremetech.com/dashboard"
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_session.get_page = AsyncMock(return_value=mock_page)

        mock_ensure_session.return_value = (mock_session, True)
        mock_calibrate.return_value = {
            "job_card": {"status": "primary_matched", "selector": "[data-testid='job-card']", "count": 10},
            "job_title": {"status": "primary_matched", "selector": "[data-testid='job-title']", "count": 1},
        }

        resp = await calibrate_selectors(force_recalibrate=True)
        self.assertTrue(resp["success"])
        self.assertIn("Calibrated 2/2 selectors successfully", resp["message"])
        self.assertEqual(resp["data"]["matched_count"], 2)

    @patch("hireme_mcp.main._ensure_session")
    async def test_calibrate_selectors_exception_handling(self, mock_ensure_session):
        mock_session = MagicMock()
        mock_session.get_page = AsyncMock(side_effect=RuntimeError("Browser page crashed"))
        mock_ensure_session.return_value = (mock_session, True)

        resp = await calibrate_selectors()
        self.assertFalse(resp["success"])
        self.assertEqual(resp["error_code"], "CALIBRATION_ERROR")
