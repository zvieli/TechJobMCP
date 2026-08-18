"""Unit and integration tests for FastMCP server setup, lifespan, and CLI entry points."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import FastMCP

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from job_mcp import mcp
from job_mcp.__main__ import main as server_main
from job_mcp.core.api_client import JobCache
from job_mcp.core.auth import SessionManager
from job_mcp.main import GeminiProbeMiddleware, browser_lifespan
from job_mcp.setup import main as setup_main, run_setup


class TestGeminiProbeMiddleware(unittest.TestCase):
    """Integration tests for GeminiProbeMiddleware ASGI handling of probe requests."""

    @classmethod
    def setUpClass(cls):
        app = mcp.http_app(transport="http")
        app.add_middleware(GeminiProbeMiddleware)
        cls.client = TestClient(app)

    def test_options_preflight(self):
        """Verify OPTIONS requests return 200 with CORS headers."""
        for path in ("/mcp", "/sse", "/health", "/any-endpoint"):
            response = self.client.options(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
            self.assertEqual(response.headers.get("access-control-allow-methods"), "*")
            self.assertEqual(response.headers.get("access-control-allow-headers"), "*")

    def test_head_probe(self):
        """Verify HEAD requests return 200 with CORS headers."""
        for path in ("/mcp", "/sse", "/health", "/"):
            response = self.client.head(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_get_non_sse_probe_on_mcp_and_sse(self):
        """Verify GET without text/event-stream on /mcp or /sse returns 200 'MCP Server Active'."""
        for path in ("/mcp", "/sse"):
            response = self.client.get(path, headers={"accept": "application/json"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/plain", response.headers.get("content-type", ""))
            self.assertEqual(response.text, "MCP Server Active")
            self.assertEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_delete_probe_on_mcp_and_sse(self):
        """Verify DELETE on /mcp and /sse returns 200 with CORS headers."""
        for path in ("/mcp", "/sse"):
            response = self.client.delete(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_oauth_metadata_discovery_probes(self):
        """Verify .well-known/oauth-* endpoints return standard OAuth metadata with CORS headers."""
        for path in (
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/v1",
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-authorization-server/default",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("resource", data)
            self.assertEqual(data.get("authorization_servers"), [])
            self.assertEqual(data.get("scopes_supported"), ["mcp"])
            self.assertEqual(data.get("response_types_supported"), ["code"])
            self.assertEqual(data.get("grant_types_supported"), ["authorization_code"])
            self.assertEqual(
                data.get("token_endpoint_auth_methods_supported"),
                ["none", "client_secret_post"],
            )
            self.assertEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_intercept_409_conflict_on_mcp_endpoints(self):
        """Verify downstream 409 Conflict on /mcp or /sse is intercepted and converted to 200 OK."""
        async def conflict_route(request):
            return Response(b"Downstream conflict", status_code=409)

        async def other_route(request):
            return Response(b"Other conflict", status_code=409)

        app = Starlette(
            routes=[
                Route("/mcp", conflict_route, methods=["GET", "POST"]),
                Route("/sse", conflict_route, methods=["GET", "POST"]),
                Route("/other", other_route, methods=["GET", "POST"]),
            ]
        )
        app.add_middleware(GeminiProbeMiddleware)
        client = TestClient(app)

        # /mcp 409 should be intercepted and mapped to 200 OK
        resp_mcp = client.post("/mcp")
        self.assertEqual(resp_mcp.status_code, 200)
        self.assertEqual(resp_mcp.text, "MCP Session Reset")
        self.assertEqual(resp_mcp.headers.get("access-control-allow-origin"), "*")

        # /sse 409 should be intercepted and mapped to 200 OK
        resp_sse = client.get("/sse")
        self.assertEqual(resp_sse.status_code, 200)
        self.assertEqual(resp_sse.headers.get("access-control-allow-origin"), "*")

        # Non-MCP path /other 409 should NOT be intercepted
        resp_other = client.get("/other")
        self.assertEqual(resp_other.status_code, 409)
        self.assertEqual(resp_other.text, "Other conflict")

    def test_passthrough_custom_routes(self):
        """Verify regular routes like /health and / pass through to the FastMCP route handlers."""
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json().get("status"), "ok")

        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)
        self.assertEqual(root_resp.json().get("status"), "ok")


class TestServerRegistration(unittest.IsolatedAsyncioTestCase):
    """Tests for FastMCP server registration, tools, and metadata."""

    def test_server_metadata(self):
        """Test server name and instructions."""
        self.assertEqual(mcp.name, "HireMeTech")
        self.assertIn("HireMeTech MCP Server", mcp.instructions)

    async def test_all_tools_registered(self):
        """Verify all required tools are properly registered on the FastMCP instance."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "list_job_sources",
            "get_job_matches",
            "filter_jobs_by_preferences",
            "bookmark_job",
            "delete_job",
            "auto_apply_job",
            "confirm_auto_apply",
            "calibrate_selectors",
            "set_operation_mode",
        ]

        for expected in expected_tools:
            self.assertIn(expected, tool_names)

    @patch.object(SessionManager, "initialize", new_callable=AsyncMock)
    @patch.object(SessionManager, "check_session_health", new_callable=AsyncMock)
    @patch.object(SessionManager, "shutdown", new_callable=AsyncMock)
    async def test_browser_lifespan_lazy(self, mock_shutdown, mock_health, mock_init):
        """Test browser_lifespan boots without eager browser initialization."""
        mock_init.return_value = None
        mock_health.return_value = True
        mock_shutdown.return_value = None

        mock_server = MagicMock(spec=FastMCP)

        async with browser_lifespan(mock_server) as state:
            self.assertIn("session", state)
            self.assertIn("cache", state)
            self.assertIn("registry", state)
            self.assertIn("aggregator", state)
            self.assertIsInstance(state["session"], SessionManager)
            self.assertIsInstance(state["cache"], JobCache)
            mock_init.assert_not_called()
            self.assertFalse(state["session"].is_running)

        mock_shutdown.assert_not_called()

    @patch.object(SessionManager, "shutdown", new_callable=AsyncMock)
    async def test_browser_lifespan_shutdown_when_running(self, mock_shutdown):
        """Test browser_lifespan cleanly shuts down session if it was lazily started."""
        mock_server = MagicMock(spec=FastMCP)

        async with browser_lifespan(mock_server) as state:
            session = state["session"]
            session._initialized = True
            session.context = MagicMock()
            self.assertTrue(session.is_running)

        mock_shutdown.assert_called_once()

    async def test_tool_response_contains_trace_id(self):
        """Verify that tool responses include an auto-generated trace_id."""
        from job_mcp.main import set_operation_mode
        res = await set_operation_mode(mode="autonomous")
        self.assertTrue(res["success"])
        self.assertIn("trace_id", res)
        self.assertIsNotNone(res["trace_id"])
        self.assertEqual(len(res["trace_id"]), 8)

        # Invalid mode error response
        err_res = await set_operation_mode(mode="invalid_mode_xyz")
        self.assertFalse(err_res["success"])
        self.assertIn("trace_id", err_res)
        self.assertIsNotNone(err_res["trace_id"])
        self.assertEqual(len(err_res["trace_id"]), 8)


class TestCliAndSetup(unittest.IsolatedAsyncioTestCase):
    """Tests for setup CLI and __main__ execution."""

    @patch("job_mcp.setup.async_playwright")
    async def test_run_setup_success(self, mock_async_playwright):
        """Test run_setup happy path with successful authentication."""
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/dashboard"
        mock_page.goto.return_value = MagicMock(status=200)
        mock_locator = MagicMock()
        mock_locator.count = MagicMock(return_value=0)
        mock_page.locator = MagicMock(return_value=mock_locator)
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

    @patch("job_mcp.setup.async_playwright")
    async def test_run_setup_failure(self, mock_async_playwright):
        """Test run_setup when authentication verification fails."""
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://hiremetech.com/login"
        mock_page.goto.return_value = MagicMock(status=200)
        mock_locator = MagicMock()
        mock_locator.count = MagicMock(return_value=0)
        mock_page.locator = MagicMock(return_value=mock_locator)
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

    @patch("job_mcp.__main__.mcp.run")
    def test_main_stdio(self, mock_run):
        """Test __main__.py stdio transport default."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            server_main()
            mock_run.assert_called_once_with(transport="stdio")

    @patch("job_mcp.__main__.uvicorn.run")
    @patch("job_mcp.__main__.mcp.http_app")
    def test_main_http(self, mock_http_app, mock_uvicorn_run):
        """Test __main__.py http transport with host and port."""
        mock_app = MagicMock()
        mock_http_app.return_value = mock_app

        with patch.dict(os.environ, {
            "MCP_TRANSPORT": "http",
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": "8080",
        }):
            server_main()
            mock_http_app.assert_called_once_with(transport="http")
            mock_app.add_middleware.assert_called_once_with(GeminiProbeMiddleware)
            mock_uvicorn_run.assert_called_once_with(mock_app, host="127.0.0.1", port=8080)

    @patch("job_mcp.__main__.uvicorn.run")
    @patch("job_mcp.__main__.mcp.http_app")
    def test_main_sse(self, mock_http_app, mock_uvicorn_run):
        """Test __main__.py sse transport with host and port."""
        mock_app = MagicMock()
        mock_http_app.return_value = mock_app

        with patch.dict(os.environ, {
            "MCP_TRANSPORT": "sse",
            "MCP_HOST": "0.0.0.0",
            "MCP_PORT": "9000",
        }):
            server_main()
            mock_http_app.assert_called_once_with(transport="sse")
            mock_app.add_middleware.assert_called_once_with(GeminiProbeMiddleware)
            mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=9000)


if __name__ == "__main__":
    unittest.main()
