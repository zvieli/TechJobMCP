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
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences, WorkMode
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
        self.assertEqual(mcp.name, "TechJobMCP")
        self.assertIn("Tech Job MCP Server", mcp.instructions)

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
            "get_application_history",
            "calibrate_selectors",
            "set_operation_mode",
            "search_linkedin_jobs",
            "get_linkedin_job_details",
            "notify_new_jobs",
            "test_notifier",
            "run_job_scout",
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
            self.assertIn("tracker", state)
            self.assertIn("notifier", state)
            self.assertIn("ledger", state)
            self.assertIn("dispatcher", state)
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


class TestLinkedInTools(unittest.IsolatedAsyncioTestCase):
    """Tests for LinkedIn MCP tools: search_linkedin_jobs and get_linkedin_job_details."""

    @patch("job_mcp.main.search_linkedin_jobs_api", new_callable=AsyncMock)
    async def test_search_linkedin_jobs_success_and_cache(self, mock_api):
        """Verify search_linkedin_jobs parses args, updates cache, and returns ToolResponse."""
        from fastmcp import Context
        from job_mcp.main import search_linkedin_jobs
        from job_mcp.models.schemas import Job, WorkMode

        sample_jobs = [
            Job(
                job_id="linkedin_12345",
                title="Full Stack Engineer",
                company="TechCo",
                location="Tel Aviv",
                work_mode=WorkMode.HYBRID,
                source="linkedin",
                sources=["linkedin"],
            ),
            Job(
                job_id="linkedin_67890",
                title="Backend Developer",
                company="DataInc",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                source="linkedin",
                sources=["linkedin"],
            ),
        ]
        mock_api.return_value = sample_jobs

        cache = JobCache()
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await search_linkedin_jobs(
            keywords="Engineer",
            location="Israel",
            work_mode="hybrid",
            limit=10,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertIn("Successfully fetched 2 LinkedIn", res["message"])
        self.assertEqual(len(res["data"]), 2)
        self.assertIn("trace_id", res)
        self.assertEqual(len(cache.get_all()), 2)
        mock_api.assert_called_once_with(
            keywords="Engineer",
            location="Israel",
            start=0,
            work_mode=WorkMode.HYBRID,
            f_WT=None,
            f_TPR=None,
            f_AL=None,
            f_E=None,
            sort_by=None,
        )

    @patch("job_mcp.main.search_linkedin_jobs_api", new_callable=AsyncMock)
    async def test_search_linkedin_jobs_limit_truncation(self, mock_api):
        """Verify search_linkedin_jobs respects limit parameter."""
        from job_mcp.main import search_linkedin_jobs
        from job_mcp.models.schemas import Job

        mock_api.return_value = [
            Job(job_id=f"linkedin_{i}", title=f"Job {i}", company="Co", source="linkedin")
            for i in range(10)
        ]

        res = await search_linkedin_jobs(keywords="Python", limit=3)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 3)

    @patch("job_mcp.main.search_linkedin_jobs_api", new_callable=AsyncMock)
    async def test_search_linkedin_jobs_error_handling(self, mock_api):
        """Verify search_linkedin_jobs gracefully catches exceptions and returns error ToolResponse."""
        from job_mcp.main import search_linkedin_jobs
        mock_api.side_effect = RuntimeError("LinkedIn rate limit exceeded")

        res = await search_linkedin_jobs(keywords="Python")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "LINKEDIN_SEARCH_ERROR")
        self.assertIn("Failed to search LinkedIn jobs", res["message"])
        self.assertIn("trace_id", res)

    @patch("job_mcp.main._get_linkedin_source")
    async def test_get_linkedin_job_details_success(self, mock_get_source):
        """Verify get_linkedin_job_details retrieves and formats job details."""
        from job_mcp.main import get_linkedin_job_details
        from job_mcp.models.schemas import WorkMode

        mock_source = AsyncMock()
        mock_source.fetch_job_details.return_value = {
            "title": "Senior Staff Architect",
            "company": "Enterprise Tech",
            "location": "Tel Aviv",
            "description": "Leading architecture initiatives",
            "work_mode": WorkMode.HYBRID,
            "apply_url": "https://linkedin.com/jobs/view/123",
            "tech_stack": ["Python", "Kubernetes"],
        }
        mock_get_source.return_value = mock_source

        res = await get_linkedin_job_details(job_id="linkedin_12345")
        self.assertTrue(res["success"])
        self.assertIn("Successfully retrieved details", res["message"])
        self.assertEqual(res["data"]["title"], "Senior Staff Architect")
        self.assertEqual(res["data"]["work_mode"], "hybrid")
        self.assertIn("trace_id", res)

    @patch("job_mcp.main._get_linkedin_source")
    async def test_get_linkedin_job_details_not_found(self, mock_get_source):
        """Verify get_linkedin_job_details returns error when job details not found."""
        from job_mcp.main import get_linkedin_job_details

        mock_source = AsyncMock()
        mock_source.fetch_job_details.return_value = None
        mock_get_source.return_value = mock_source

        res = await get_linkedin_job_details(job_id="linkedin_99999")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "JOB_NOT_FOUND")
        self.assertIn("Could not retrieve details", res["message"])

    @patch("job_mcp.main._get_linkedin_source")
    async def test_get_linkedin_job_details_error_handling(self, mock_get_source):
        """Verify get_linkedin_job_details handles unexpected exceptions."""
        from job_mcp.main import get_linkedin_job_details

        mock_source = AsyncMock()
        mock_source.fetch_job_details.side_effect = RuntimeError("Network timeout")
        mock_get_source.return_value = mock_source

        res = await get_linkedin_job_details(job_id="linkedin_err")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "LINKEDIN_DETAILS_ERROR")
        self.assertIn("Failed to get LinkedIn job details", res["message"])


class TestNotificationTools(unittest.IsolatedAsyncioTestCase):
    """Tests for notify_new_jobs and test_notifier MCP tools."""

    async def test_notify_new_jobs_unsupported_channel(self):
        """Verify notify_new_jobs rejects unsupported notification channels."""
        from job_mcp.main import notify_new_jobs

        res = await notify_new_jobs(channel="discord_webhook")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "UNSUPPORTED_CHANNEL")
        self.assertIn("Unsupported notification channel", res["message"])

    async def test_notify_new_jobs_unconfigured_notifier(self):
        """Verify notify_new_jobs returns error when Telegram bot is not configured."""
        from job_mcp.main import notify_new_jobs
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_notifier = TelegramNotifier(bot_token="", chat_id="")
        ctx = MagicMock()
        ctx.lifespan_context = {"notifier": mock_notifier}

        res = await notify_new_jobs(channel="telegram", ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "NOTIFIER_NOT_CONFIGURED")
        self.assertIn("Telegram notifier is not configured", res["message"])

    async def test_notify_new_jobs_no_unseen_jobs(self):
        """Verify notify_new_jobs returns cleanly when all jobs have already been seen."""
        from job_mcp.main import notify_new_jobs
        from job_mcp.models.schemas import Job
        from job_mcp.notifiers.telegram import TelegramNotifier
        from job_mcp.notifiers.tracker import JobTracker
        from job_mcp.sources import JobAggregator, SourceRegistry

        job = Job(job_id="job_seen_1", title="DevOps", company="Co", source="comeet")
        tracker = JobTracker()
        tracker.mark_seen(job)

        mock_agg = AsyncMock(spec=JobAggregator)
        mock_agg.fetch_all_jobs.return_value = [job]

        mock_notifier = MagicMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True

        ctx = MagicMock()
        ctx.lifespan_context = {
            "notifier": mock_notifier,
            "tracker": tracker,
            "aggregator": mock_agg,
        }

        res = await notify_new_jobs(channel="telegram", ctx=ctx)
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["notified_count"], 0)
        self.assertEqual(res["data"]["jobs"], [])
        mock_notifier.send_alert.assert_not_called()

    async def test_notify_new_jobs_success_and_mark_seen(self):
        """Verify notify_new_jobs filters unseen, sends alert, and marks them seen."""
        from job_mcp.main import notify_new_jobs
        from job_mcp.models.schemas import Job, WorkMode
        from job_mcp.notifiers.telegram import TelegramNotifier
        from job_mcp.notifiers.tracker import JobTracker
        from job_mcp.sources import JobAggregator

        jobs = [
            Job(job_id="job_new_1", title="Backend Lead", company="Alpha", source="comeet"),
            Job(job_id="job_new_2", title="ML Engineer", company="Beta", source="workday"),
        ]

        tracker = JobTracker()
        mock_agg = AsyncMock(spec=JobAggregator)
        mock_agg.fetch_all_jobs.return_value = jobs

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.send_alert.return_value = True

        ctx = MagicMock()
        ctx.lifespan_context = {
            "notifier": mock_notifier,
            "tracker": tracker,
            "aggregator": mock_agg,
        }

        res = await notify_new_jobs(
            channel="telegram",
            work_mode="remote",
            auto_mark_seen=True,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["notified_count"], 2)
        self.assertEqual(len(res["data"]["jobs"]), 2)
        self.assertIn("trace_id", res)
        mock_notifier.send_alert.assert_called_once()
        self.assertTrue(tracker.is_seen("job_new_1"))
        self.assertTrue(tracker.is_seen("job_new_2"))

    async def test_notify_new_jobs_with_min_score_filter(self):
        """Verify notify_new_jobs filters by minimum match score."""
        from job_mcp.main import notify_new_jobs
        from job_mcp.models.schemas import Job
        from job_mcp.notifiers.telegram import TelegramNotifier
        from job_mcp.notifiers.tracker import JobTracker
        from job_mcp.sources import JobAggregator

        jobs = [
            Job(job_id="job_high_score", title="Python Lead", company="Alpha", match_score=90.0, source="comeet"),
            Job(job_id="job_low_score", title="Frontend", company="Beta", match_score=40.0, source="workday"),
        ]

        tracker = JobTracker()
        mock_agg = AsyncMock(spec=JobAggregator)
        mock_agg.fetch_all_jobs.return_value = jobs

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.send_alert.return_value = True

        ctx = MagicMock()
        ctx.lifespan_context = {
            "notifier": mock_notifier,
            "tracker": tracker,
            "aggregator": mock_agg,
        }

        res = await notify_new_jobs(channel="telegram", min_score=80.0, ctx=ctx)
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["notified_count"], 1)
        self.assertEqual(res["data"]["jobs"][0]["job_id"], "job_high_score")

    async def test_notify_new_jobs_dispatch_failure(self):
        """Verify notify_new_jobs reports error when send_alert returns False."""
        from job_mcp.main import notify_new_jobs
        from job_mcp.models.schemas import Job
        from job_mcp.notifiers.telegram import TelegramNotifier
        from job_mcp.notifiers.tracker import JobTracker
        from job_mcp.sources import JobAggregator

        jobs = [Job(job_id="job_1", title="Backend", company="Alpha", source="comeet")]
        tracker = JobTracker()
        mock_agg = AsyncMock(spec=JobAggregator)
        mock_agg.fetch_all_jobs.return_value = jobs

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.send_alert.return_value = False

        ctx = MagicMock()
        ctx.lifespan_context = {
            "notifier": mock_notifier,
            "tracker": tracker,
            "aggregator": mock_agg,
        }

        res = await notify_new_jobs(channel="telegram", ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "NOTIFICATION_FAILED")
        self.assertFalse(tracker.is_seen("job_1"))

    async def test_test_notifier_unsupported_channel(self):
        """Verify test_notifier rejects invalid channel."""
        from job_mcp.main import test_notifier

        res = await test_notifier(channel="unknown_channel")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "UNSUPPORTED_CHANNEL")

    async def test_test_notifier_unconfigured(self):
        """Verify test_notifier handles unconfigured notifier."""
        from job_mcp.main import test_notifier
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_notifier = TelegramNotifier(bot_token="", chat_id="")
        ctx = MagicMock()
        ctx.lifespan_context = {"notifier": mock_notifier}

        res = await test_notifier(channel="telegram", ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "NOTIFIER_NOT_CONFIGURED")

    async def test_test_notifier_health_failure(self):
        """Verify test_notifier handles health check failure."""
        from job_mcp.main import test_notifier
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.check_health.return_value = False

        ctx = MagicMock()
        ctx.lifespan_context = {"notifier": mock_notifier}

        res = await test_notifier(channel="telegram", ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "HEALTH_CHECK_FAILED")

    async def test_test_notifier_success(self):
        """Verify test_notifier passes health check and dispatches test message."""
        from job_mcp.main import test_notifier
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.check_health.return_value = True
        mock_notifier.send_alert.return_value = True

        ctx = MagicMock()
        ctx.lifespan_context = {"notifier": mock_notifier}

        res = await test_notifier(channel="telegram", ctx=ctx)
        self.assertTrue(res["success"])
        self.assertTrue(res["data"]["healthy"])
        self.assertTrue(res["data"]["delivered"])
        self.assertIn("trace_id", res)
        mock_notifier.check_health.assert_called_once()
        mock_notifier.send_alert.assert_called_once()

    async def test_test_notifier_delivery_failure(self):
        """Verify test_notifier handles delivery failure after healthy check."""
        from job_mcp.main import test_notifier
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_notifier = AsyncMock(spec=TelegramNotifier)
        mock_notifier.is_configured = True
        mock_notifier.check_health.return_value = True
        mock_notifier.send_alert.return_value = False

        ctx = MagicMock()
        ctx.lifespan_context = {"notifier": mock_notifier}

        res = await test_notifier(channel="telegram", ctx=ctx)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "DELIVERY_FAILED")


class TestExternalSourceBookmarkAndDelete(unittest.IsolatedAsyncioTestCase):
    """Tests for bookmark_job and delete_job on external multi-source jobs."""

    async def test_bookmark_external_sources_in_cache(self):
        """Verify bookmark_job on external sources marks cache without launching browser."""
        from job_mcp.main import bookmark_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        jobs = [
            Job(job_id="workday_wix_123", title="Engineer", company="Wix", source="workday"),
            Job(job_id="eightfold_nvidia_456", title="Architect", company="Nvidia", source="eightfold"),
            Job(job_id="direct_google_789", title="Lead", company="Google", source="direct_tech"),
            Job(job_id="linkedin_112233", title="Specialist", company="LinkedIn Corp", source="linkedin"),
        ]
        cache.update(jobs)

        ctx = MagicMock()
        ctx.lifespan_context = {"cache": cache}

        for j in jobs:
            res = await bookmark_job(job_id=j.job_id, ctx=ctx)
            self.assertTrue(res["success"])
            self.assertIn("Successfully bookmarked", res["message"])
            self.assertTrue(res["data"]["is_bookmarked"])
            self.assertTrue(cache.get_by_id(j.job_id).is_bookmarked)

    async def test_delete_external_sources_in_cache(self):
        """Verify delete_job on external sources dismisses from cache without launching browser."""
        from job_mcp.main import delete_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        jobs = [
            Job(job_id="workday_wix_123", title="Engineer", company="Wix", source="workday"),
            Job(job_id="eightfold_nvidia_456", title="Architect", company="Nvidia", source="eightfold"),
            Job(job_id="direct_google_789", title="Lead", company="Google", source="direct_tech"),
            Job(job_id="linkedin_112233", title="Specialist", company="LinkedIn Corp", source="linkedin"),
        ]
        cache.update(jobs)

        ctx = MagicMock()
        ctx.lifespan_context = {"cache": cache}

        for j in jobs:
            res = await delete_job(job_id=j.job_id, ctx=ctx)
            self.assertTrue(res["success"])
            self.assertIn("Successfully dismissed", res["message"])
            self.assertIsNone(cache.get_by_id(j.job_id))


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
    @patch("job_mcp.__main__.uvicorn.run")
    @patch("job_mcp.__main__.mcp.http_app")
    def test_main_https_alias(self, mock_http_app, mock_uvicorn_run):
        """Test __main__.py maps https transport alias to http without error."""
        mock_app = MagicMock()
        mock_http_app.return_value = mock_app

        with patch.dict(os.environ, {
            "MCP_TRANSPORT": "https",
            "MCP_HOST": "0.0.0.0",
            "MCP_PORT": "8000",
        }):
            server_main()
            mock_http_app.assert_called_once_with(transport="http")
            mock_app.add_middleware.assert_called_once_with(GeminiProbeMiddleware)
            mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=8000)


class TestServerDynamicQueryPropagation(unittest.IsolatedAsyncioTestCase):
    """Tests for dynamic CV query resolution and propagation across FastMCP tools."""

    def setUp(self) -> None:
        """Set up test fixtures with sample junior CV and test jobs."""
        self.junior_cv_text = (
            "Alex Smith\n"
            "Computer Science Student at HIT (Holon Institute of Technology)\n"
            "Email: alex@example.com | Tel Aviv, Israel\n\n"
            "Technical Skills:\n"
            "Languages & Frameworks: Python, FastAPI, Docker, PostgreSQL, Redis, Git\n\n"
            "Projects:\n"
            "FastAPI backend microservice with PostgreSQL database."
        )

        self.junior_job = Job(
            job_id="job_junior_10",
            title="Junior Python Developer",
            company="CloudTech",
            location="Tel Aviv",
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "FastAPI", "PostgreSQL"],
            description="Seeking a junior Python developer for our backend team.",
            source="linkedin",
            sources=["linkedin"],
        )
        self.senior_job = Job(
            job_id="job_senior_20",
            title="Senior Lead Software Architect",
            company="EnterpriseCorp",
            location="Tel Aviv",
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "Kubernetes", "AWS"],
            description="Senior Architect with 10+ years experience.",
            source="workday",
            sources=["workday"],
        )

    def _create_mock_context(self, cache: JobCache, aggregator: Any = None) -> MagicMock:
        """Create mock FastMCP Context with cache and optional aggregator."""
        from fastmcp import Context
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {
            "cache": cache,
            "aggregator": aggregator,
            "session": MagicMock(spec=SessionManager),
        }
        return ctx

    async def test_get_job_matches_with_cv_path(self) -> None:
        """Verify get_job_matches extracts CV profile, filters out senior jobs, and scores junior matches."""
        from job_mcp.main import get_job_matches

        cache = JobCache()
        cache.update([self.junior_job, self.senior_job])
        ctx = self._create_mock_context(cache)

        res = await get_job_matches(
            cv_path=self.junior_cv_text,
            force_refresh=False,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertIn("trace_id", res)
        # Senior job should be excluded by junior CV suggested exclusions
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["job_id"], "job_junior_10")
        self.assertGreater(res["data"][0]["match_score"], 60.0)
        self.assertIn("Python", res["data"][0]["matched_skills"])

    async def test_get_job_matches_with_all_preference_params(self) -> None:
        """Verify get_job_matches handles all search and preference parameters combined."""
        from job_mcp.main import get_job_matches

        cache = JobCache()
        cache.update([self.junior_job, self.senior_job])
        ctx = self._create_mock_context(cache)

        res = await get_job_matches(
            tech_stack=["Python", "FastAPI"],
            work_mode="hybrid",
            location="Tel Aviv",
            min_salary=50000,
            keywords=["Python"],
            exclude_keywords=["PHP"],
            cv_path=self.junior_cv_text,
            limit=5,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertIn("trace_id", res)
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["job_id"], "job_junior_10")

    async def test_filter_jobs_by_preferences_with_cv_path(self) -> None:
        """Verify filter_jobs_by_preferences parses CV and filters cached listings."""
        from job_mcp.main import filter_jobs_by_preferences

        cache = JobCache()
        cache.update([self.junior_job, self.senior_job])
        ctx = self._create_mock_context(cache)

        res = await filter_jobs_by_preferences(
            cv_path=self.junior_cv_text,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertIn("Found 1 matching jobs", res["message"])
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["job_id"], "job_junior_10")
        self.assertIn("Python", res["data"][0]["matched_skills"])
        self.assertGreater(res["data"][0]["match_score"], 60.0)

    @patch("job_mcp.main.search_linkedin_jobs_api", new_callable=AsyncMock)
    async def test_search_linkedin_jobs_dynamic_cv_keyword(self, mock_api) -> None:
        """Verify search_linkedin_jobs derives dynamic query from CV when keywords is empty."""
        from job_mcp.main import search_linkedin_jobs

        mock_api.return_value = [self.junior_job]
        cache = JobCache()
        ctx = self._create_mock_context(cache)

        res = await search_linkedin_jobs(
            keywords="",
            cv_path=self.junior_cv_text,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        mock_api.assert_called_once()
        call_kwargs = mock_api.call_args.kwargs
        called_keywords = call_kwargs.get("keywords") or (mock_api.call_args.args[0] if mock_api.call_args.args else "")
        self.assertTrue(len(called_keywords) > 0)
        self.assertTrue("python" in called_keywords.lower())

    async def test_get_job_matches_live_fetch_derives_and_passes_primary_stack(self) -> None:
        """Verify get_job_matches dynamically derives primary_stack from CV and passes it to aggregator."""
        from job_mcp.main import get_job_matches

        mock_aggregator = MagicMock()
        mock_aggregator.fetch_all_jobs = AsyncMock(return_value=[self.junior_job])
        cache = JobCache()
        ctx = self._create_mock_context(cache, aggregator=mock_aggregator)

        res = await get_job_matches(
            cv_path=self.junior_cv_text,
            tech_stack=None,
            force_refresh=True,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 1)
        mock_aggregator.fetch_all_jobs.assert_called_once()
        call_kwargs = mock_aggregator.fetch_all_jobs.call_args.kwargs
        passed_prefs = call_kwargs.get("preferences")
        self.assertIsNotNone(passed_prefs)
        self.assertTrue(len(passed_prefs.tech_stack) > 0)
        self.assertIn("Python", passed_prefs.tech_stack)
        self.assertIsNotNone(call_kwargs.get("profile"))
        self.assertIn("Python", call_kwargs.get("profile").primary_stack)

    async def test_filter_jobs_by_preferences_dynamic_primary_stack(self) -> None:
        """Verify filter_jobs_by_preferences derives primary_stack when tech_stack is omitted."""
        from job_mcp.main import filter_jobs_by_preferences

        cache = JobCache()
        cache.update([self.junior_job, self.senior_job])
        ctx = self._create_mock_context(cache)

        res = await filter_jobs_by_preferences(
            cv_path=self.junior_cv_text,
            tech_stack=None,
            ctx=ctx,
        )

        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["job_id"], "job_junior_10")
        self.assertIn("Python", res["data"][0]["matched_skills"])


class TestSessionThrottlingAndUnauthenticatedFastActions(unittest.IsolatedAsyncioTestCase):
    """Tests for non-blocking dismissal, bookmarking, and session recovery cooldown."""

    def setUp(self) -> None:
        import job_mcp.main as main_module
        main_module._last_session_failure_time = 0.0

    async def test_delete_hiremetech_job_unauthenticated_skips_portal_recovery(self) -> None:
        """Verify delete_job on hiremetech source skips portal launch when unauthenticated."""
        from fastmcp import Context
        from job_mcp.main import delete_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        job = Job(job_id="hmt_job_101", title="Backend Engineer", company="HireMeTech Co", source="hiremetech")
        cache.update([job])

        mock_session = AsyncMock(spec=SessionManager)
        mock_session.is_running = False
        mock_session.is_authenticated = False
        mock_session.check_session_health.return_value = False

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache, "session": mock_session}

        res = await delete_job(job_id="hmt_job_101", ctx=ctx)

        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "hmt_job_101")
        self.assertFalse(res["data"]["portal_deleted"])
        self.assertIn("portal dismissal skipped: unauthenticated session", res["message"])
        self.assertIsNone(cache.get_by_id("hmt_job_101"))
        mock_session.ensure_ready.assert_not_called()

    async def test_bookmark_hiremetech_job_unauthenticated_skips_portal_recovery(self) -> None:
        """Verify bookmark_job on hiremetech source updates cache and skips portal launch when unauthenticated."""
        from fastmcp import Context
        from job_mcp.main import bookmark_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        job = Job(job_id="hmt_job_202", title="Frontend Engineer", company="HireMeTech Co", source="hiremetech")
        cache.update([job])

        mock_session = AsyncMock(spec=SessionManager)
        mock_session.is_running = False
        mock_session.is_authenticated = False
        mock_session.check_session_health.return_value = False

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache, "session": mock_session}

        res = await bookmark_job(job_id="hmt_job_202", ctx=ctx)

        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "hmt_job_202")
        self.assertTrue(res["data"]["is_bookmarked"])
        self.assertFalse(res["data"]["portal_bookmarked"])
        self.assertIn("portal bookmark skipped: unauthenticated session", res["message"])
        self.assertTrue(cache.get_by_id("hmt_job_202").is_bookmarked)
        mock_session.ensure_ready.assert_not_called()

    async def test_delete_external_job_by_source_attribute(self) -> None:
        """Verify delete_job on cached job with non-hiremetech source returns immediately."""
        from fastmcp import Context
        from job_mcp.main import delete_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        job = Job(job_id="custom_ext_999", title="DevOps", company="External ATS", source="comeet")
        cache.update([job])

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await delete_job(job_id="custom_ext_999", ctx=ctx)

        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "custom_ext_999")
        self.assertIn("Successfully dismissed external job", res["message"])
        self.assertIsNone(cache.get_by_id("custom_ext_999"))

    async def test_bookmark_external_job_by_source_attribute(self) -> None:
        """Verify bookmark_job on cached job with non-hiremetech source returns immediately."""
        from fastmcp import Context
        from job_mcp.main import bookmark_job
        from job_mcp.models.schemas import Job

        cache = JobCache()
        job = Job(job_id="custom_ext_888", title="ML Lead", company="External ATS", source="workday")
        cache.update([job])

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"cache": cache}

        res = await bookmark_job(job_id="custom_ext_888", ctx=ctx)

        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["job_id"], "custom_ext_888")
        self.assertTrue(res["data"]["is_bookmarked"])
        self.assertIn("Successfully bookmarked external job", res["message"])
        self.assertTrue(cache.get_by_id("custom_ext_888").is_bookmarked)

    async def test_ensure_session_cooldown_throttling(self) -> None:
        """Verify _ensure_session applies cooldown and does not trigger recovery storms."""
        import time
        from fastmcp import Context
        from job_mcp.main import _ensure_session
        import job_mcp.main as main_module

        mock_session = AsyncMock(spec=SessionManager)
        mock_session.ensure_ready.side_effect = RuntimeError("Browser connection failed")

        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {"session": mock_session}

        # First call triggers ensure_ready and fails
        session, is_healthy = await _ensure_session(ctx)
        self.assertFalse(is_healthy)
        self.assertEqual(mock_session.ensure_ready.call_count, 1)

        # Immediate second call is throttled by cooldown (does not call ensure_ready)
        session2, is_healthy2 = await _ensure_session(ctx)
        self.assertFalse(is_healthy2)
        self.assertEqual(mock_session.ensure_ready.call_count, 1)

        # After advancing failure timestamp beyond cooldown window, ensure_ready is called again
        mock_session._last_failure_time = time.monotonic() - 30.0
        main_module._last_session_failure_time = time.monotonic() - 30.0
        session3, is_healthy3 = await _ensure_session(ctx)
        self.assertFalse(is_healthy3)
        self.assertEqual(mock_session.ensure_ready.call_count, 2)


if __name__ == "__main__":
    unittest.main()



