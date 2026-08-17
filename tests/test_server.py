"""Unit and integration tests for FastMCP server setup, lifespan, and CLI entry points."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import FastMCP

from hireme_mcp import mcp
from hireme_mcp.__main__ import main as server_main
from hireme_mcp.core.api_client import JobCache
from hireme_mcp.core.auth import SessionManager
from hireme_mcp.main import browser_lifespan
from hireme_mcp.setup import main as setup_main, run_setup


class TestServerRegistration(unittest.IsolatedAsyncioTestCase):
    """Tests for FastMCP server registration, tools, and metadata."""

    def test_server_metadata(self):
        """Test server name and instructions."""
        self.assertEqual(mcp.name, "HireMeTech")
        self.assertIn("HireMeTech MCP Server", mcp.instructions)

    async def test_all_tools_registered(self):
        """Verify all 6 required tools are properly registered on the FastMCP instance."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "get_job_matches",
            "filter_jobs_by_preferences",
            "bookmark_job",
            "delete_job",
            "auto_apply_job",
            "confirm_auto_apply",
            "calibrate_selectors",
        ]

        for expected in expected_tools:
            self.assertIn(expected, tool_names)

    @patch.object(SessionManager, "initialize", new_callable=AsyncMock)
    @patch.object(SessionManager, "check_session_health", new_callable=AsyncMock)
    @patch.object(SessionManager, "shutdown", new_callable=AsyncMock)
    async def test_browser_lifespan(self, mock_shutdown, mock_health, mock_init):
        """Test browser_lifespan startup, state yield, and cleanup shutdown."""
        mock_init.return_value = None
        mock_health.return_value = True
        mock_shutdown.return_value = None

        mock_server = MagicMock(spec=FastMCP)

        async with browser_lifespan(mock_server) as state:
            self.assertIn("session", state)
            self.assertIn("cache", state)
            self.assertIsInstance(state["session"], SessionManager)
            self.assertIsInstance(state["cache"], JobCache)
            mock_init.assert_called_once()
            mock_health.assert_called_once()

        mock_shutdown.assert_called_once()


class TestCliAndSetup(unittest.IsolatedAsyncioTestCase):
    """Tests for setup CLI and __main__ execution."""

    @patch("hireme_mcp.setup.async_playwright")
    async def test_run_setup_success(self, mock_async_playwright):
        """Test run_setup happy path with successful authentication."""
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/dashboard"
        mock_page.goto.return_value = MagicMock(status=200)
        mock_context.pages = [mock_page]
        mock_pw.chromium.launch_persistent_context.return_value = mock_context

        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw)
        mock_async_playwright.return_value = mock_cm

        with patch("builtins.input", return_value=""):
            success = await run_setup(profile_dir="/tmp/test_setup_profile")
            self.assertTrue(success)
            mock_context.close.assert_called_once()
            mock_pw.stop.assert_called_once()

    @patch("hireme_mcp.setup.async_playwright")
    async def test_run_setup_failure(self, mock_async_playwright):
        """Test run_setup when authentication verification fails."""
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/login"
        mock_page.goto.return_value = MagicMock(status=200)
        mock_context.pages = [mock_page]
        mock_pw.chromium.launch_persistent_context.return_value = mock_context

        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw)
        mock_async_playwright.return_value = mock_cm

        with patch("builtins.input", return_value=""):
            success = await run_setup(profile_dir="/tmp/test_setup_profile_fail")
            self.assertFalse(success)
            mock_context.close.assert_called_once()
            mock_pw.stop.assert_called_once()

    @patch("hireme_mcp.__main__.mcp.run")
    def test_main_stdio(self, mock_run):
        """Test __main__.py stdio transport default."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            server_main()
            mock_run.assert_called_once_with(transport="stdio")

    @patch("hireme_mcp.__main__.mcp.run")
    def test_main_http(self, mock_run):
        """Test __main__.py http transport with host and port."""
        with patch.dict(os.environ, {
            "MCP_TRANSPORT": "http",
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": "8080",
        }):
            server_main()
            mock_run.assert_called_once_with(transport="http", host="127.0.0.1", port=8080)

    @patch("hireme_mcp.__main__.mcp.run")
    def test_main_sse(self, mock_run):
        """Test __main__.py sse transport with host and port."""
        with patch.dict(os.environ, {
            "MCP_TRANSPORT": "sse",
            "MCP_HOST": "0.0.0.0",
            "MCP_PORT": "9000",
        }):
            server_main()
            mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9000)


if __name__ == "__main__":
    unittest.main()
