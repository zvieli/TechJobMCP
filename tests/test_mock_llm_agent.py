"""Unit and integration tests for MockLLMAgent, StepTrace, and PipelineResult."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Context

from job_mcp.core.api_client import JobCache
from job_mcp.core.auth import SessionManager
from job_mcp.main import _pending_applications
from job_mcp.models.schemas import (
    ApplicationPreview,
    CandidateProfile,
    Job,
    WorkMode,
)
from job_mcp.sources import JobAggregator, SourceRegistry
from job_mcp.sources.hiremetech import HireMeTechSource
from job_mcp.testing import MockLLMAgent, PipelineResult, StepTrace


class TestStepTraceAndPipelineResult(unittest.TestCase):
    """Test schema validation and defaults for StepTrace and PipelineResult."""

    def test_step_trace_defaults_and_serialization(self):
        """Test StepTrace instantiation, default values, and serialization."""
        trace = StepTrace(
            step_number=1,
            thought="Test reasoning",
            tool_name="list_job_sources",
        )
        self.assertEqual(trace.step_number, 1)
        self.assertEqual(trace.thought, "Test reasoning")
        self.assertEqual(trace.tool_name, "list_job_sources")
        self.assertEqual(trace.arguments, {})
        self.assertEqual(trace.response, {})
        self.assertEqual(trace.duration_ms, 0.0)

        dump = trace.model_dump()
        self.assertEqual(dump["step_number"], 1)
        self.assertEqual(dump["tool_name"], "list_job_sources")

    def test_pipeline_result_defaults_and_serialization(self):
        """Test PipelineResult instantiation, default values, and serialization."""
        res = PipelineResult(success=True)
        self.assertTrue(res.success)
        self.assertIsNone(res.profile)
        self.assertEqual(res.steps, [])
        self.assertEqual(res.sources_found, [])
        self.assertEqual(res.total_jobs_fetched, 0)
        self.assertEqual(res.top_tier_jobs, [])
        self.assertEqual(res.strong_match_jobs, [])
        self.assertEqual(res.bookmarked_job_ids, [])
        self.assertEqual(res.staged_apply_ids, [])
        self.assertEqual(res.confirmed_apply_ids, [])
        self.assertEqual(res.deleted_job_ids, [])
        self.assertEqual(res.execution_time_ms, 0.0)

        dump = res.model_dump()
        self.assertTrue(dump["success"])
        self.assertIsNone(dump["profile"])
        self.assertIsInstance(dump["steps"], list)

        # Test with CandidateProfile attached
        prof = CandidateProfile(
            skills=["Python", "FastAPI"],
            top_skills=["Python", "FastAPI"],
            seniority_level="Junior",
            suggested_exclusions=["Senior", "Lead"],
        )
        res_with_prof = PipelineResult(success=True, profile=prof)
        self.assertEqual(res_with_prof.profile.seniority_level, "Junior")
        self.assertEqual(res_with_prof.model_dump()["profile"]["seniority_level"], "Junior")


class TestMockLLMAgentCallTool(unittest.IsolatedAsyncioTestCase):
    """Test suite for MockLLMAgent.call_tool execution and tracing."""

    def setUp(self):
        _pending_applications.clear()
        self.cache = JobCache(ttl_minutes=10)
        self.session_mgr = AsyncMock(spec=SessionManager)
        self.session_mgr._initialized = True
        self.session_mgr.context = MagicMock()
        self.session_mgr.is_healthy = AsyncMock(return_value=True)
        self.mock_page = AsyncMock()
        self.mock_page.url = "https://app.hireme.tech/dashboard"
        self.session_mgr.get_page = AsyncMock(return_value=self.mock_page)
        self.session_mgr.ensure_ready = AsyncMock()

        self.registry = SourceRegistry()
        self.registry.register(HireMeTechSource(session_manager=self.session_mgr))
        self.aggregator = JobAggregator(registry=self.registry, cache=self.cache)

        self.ctx = MagicMock(spec=Context)
        self.ctx.lifespan_context = {
            "session": self.session_mgr,
            "cache": self.cache,
            "registry": self.registry,
            "aggregator": self.aggregator,
        }

        self.sample_jobs = [
            Job(
                job_id="job-1",
                title="Senior Python Backend Engineer",
                company="PyTech",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "FastAPI", "PostgreSQL"],
                description="Expert in Python microservices and cloud scalability.",
                salary_range="$140,000 - $160,000",
            ),
            Job(
                job_id="job-2",
                title="Frontend React Developer",
                company="WebCo",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["React", "TypeScript"],
                description="Frontend developer building reactive web apps.",
                salary_range="$120,000",
            ),
        ]
        self.cache.update(self.sample_jobs)
        self.agent = MockLLMAgent(context=self.ctx)

    async def test_call_tool_list_job_sources(self):
        """Test calling list_job_sources tool logs StepTrace."""
        resp = await self.agent.call_tool("list_job_sources", thought="Listing platforms")
        self.assertTrue(resp.get("success"))
        self.assertEqual(len(self.agent.history), 1)

        trace = self.agent.history[0]
        self.assertEqual(trace.step_number, 1)
        self.assertEqual(trace.thought, "Listing platforms")
        self.assertEqual(trace.tool_name, "list_job_sources")
        self.assertGreaterEqual(trace.duration_ms, 0.0)
        self.assertTrue(trace.response.get("success"))

    async def test_call_tool_get_job_matches(self):
        """Test calling get_job_matches returns cached jobs and logs trace."""
        resp = await self.agent.call_tool("get_job_matches", arguments={"force_refresh": False})
        self.assertTrue(resp.get("success"))
        self.assertEqual(len(resp.get("data", [])), 2)
        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(self.agent.history[0].tool_name, "get_job_matches")

    async def test_call_tool_filter_jobs_by_preferences(self):
        """Test calling filter_jobs_by_preferences scores jobs correctly."""
        resp = await self.agent.call_tool(
            "filter_jobs_by_preferences",
            arguments={"tech_stack": ["Python", "FastAPI"]},
            thought="Filtering for Python FastAPI",
        )
        self.assertTrue(resp.get("success"))
        data = resp.get("data", [])
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["job_id"], "job-1")
        self.assertEqual(data[0]["match_score"], 100.0)

        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(self.agent.history[0].thought, "Filtering for Python FastAPI")

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    async def test_call_tool_bookmark_job(self, mock_browser_bookmark):
        """Test calling bookmark_job updates cache and logs trace."""
        mock_browser_bookmark.return_value = None
        resp = await self.agent.call_tool("bookmark_job", arguments={"job_id": "job-1"})
        self.assertTrue(resp.get("success"))
        self.assertTrue(self.cache.get_by_id("job-1").is_bookmarked)
        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(self.agent.history[0].tool_name, "bookmark_job")

    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_call_tool_delete_job(self, mock_browser_delete):
        """Test calling delete_job removes job from cache and logs trace."""
        mock_browser_delete.return_value = None
        resp = await self.agent.call_tool("delete_job", arguments={"job_id": "job-2"})
        self.assertTrue(resp.get("success"))
        self.assertIsNone(self.cache.get_by_id("job-2"))
        self.assertEqual(len(self.agent.history), 1)

    @patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock)
    async def test_call_tool_auto_apply_flow(self, mock_exec, mock_preview):
        """Test two-step auto apply calling preview and confirm."""
        mock_preview.return_value = ApplicationPreview(
            job_id="job-1",
            job_title="Senior Python Backend Engineer",
            company="PyTech",
            application_method="1-Click Apply",
            fields_to_submit={"name": "Candidate"},
            warnings=[],
        )
        mock_exec.return_value = None

        preview_resp = await self.agent.call_tool("auto_apply_job", arguments={"job_id": "job-1"})
        self.assertTrue(preview_resp.get("success"))

        confirm_resp = await self.agent.call_tool("confirm_auto_apply", arguments={"job_id": "job-1"})
        self.assertTrue(confirm_resp.get("success"))
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.history[0].tool_name, "auto_apply_job")
        self.assertEqual(self.agent.history[1].tool_name, "confirm_auto_apply")

    async def test_call_tool_set_operation_mode(self):
        """Test calling set_operation_mode tool."""
        resp = await self.agent.call_tool("set_operation_mode", arguments={"mode": "autonomous"})
        self.assertTrue(resp.get("success"))
        self.assertEqual(resp.get("data", {}).get("mode"), "autonomous")

    @patch("job_mcp.main.calibrate_all_selectors", new_callable=AsyncMock)
    async def test_call_tool_calibrate_selectors(self, mock_calibrate):
        """Test calling calibrate_selectors tool."""
        mock_calibrate.return_value = {"job_card": {"status": "verified"}}
        resp = await self.agent.call_tool("calibrate_selectors", arguments={"force_recalibrate": True})
        self.assertTrue(resp.get("success"))
        self.assertEqual(resp.get("data", {}).get("matched_count"), 1)

    async def test_call_tool_unknown_tool(self):
        """Test calling unknown tool returns standardized error and logs trace."""
        resp = await self.agent.call_tool("nonexistent_tool_xyz", arguments={"foo": "bar"})
        self.assertFalse(resp.get("success"))
        self.assertEqual(resp.get("error_code"), "UNKNOWN_TOOL")
        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(self.agent.history[0].tool_name, "nonexistent_tool_xyz")

    @patch("job_mcp.testing.mock_llm.TOOL_DISPATCH")
    async def test_call_tool_exception_handling(self, mock_dispatch):
        """Test that exceptions raised by tools are caught and recorded."""
        mock_tool = AsyncMock(side_effect=RuntimeError("Simulated tool crash"))
        mock_dispatch.get.return_value = mock_tool

        resp = await self.agent.call_tool("list_job_sources")
        self.assertFalse(resp.get("success"))
        self.assertEqual(resp.get("error_code"), "TOOL_EXECUTION_ERROR")
        self.assertIn("Simulated tool crash", resp.get("message", ""))
        self.assertEqual(len(self.agent.history), 1)

    def test_reset_history(self):
        """Test clearing agent history."""
        self.agent.history.append(StepTrace(step_number=1, tool_name="list_job_sources"))
        self.assertEqual(len(self.agent.history), 1)
        self.agent.reset_history()
        self.assertEqual(len(self.agent.history), 0)


class TestMockLLMAgentRunPipeline(unittest.IsolatedAsyncioTestCase):
    """Test suite for MockLLMAgent.run_pipeline end-to-end execution."""

    def setUp(self):
        _pending_applications.clear()
        self.cache = JobCache(ttl_minutes=10)
        self.session_mgr = AsyncMock(spec=SessionManager)
        self.session_mgr._initialized = True
        self.session_mgr.context = MagicMock()
        self.session_mgr.is_healthy = AsyncMock(return_value=True)
        self.mock_page = AsyncMock()
        self.mock_page.url = "https://app.hireme.tech/dashboard"
        self.session_mgr.get_page = AsyncMock(return_value=self.mock_page)
        self.session_mgr.ensure_ready = AsyncMock()

        self.registry = SourceRegistry()
        self.registry.register(HireMeTechSource(session_manager=self.session_mgr))
        self.aggregator = JobAggregator(registry=self.registry, cache=self.cache)

        self.ctx = MagicMock(spec=Context)
        self.ctx.lifespan_context = {
            "session": self.session_mgr,
            "cache": self.cache,
            "registry": self.registry,
            "aggregator": self.aggregator,
        }

        # Setup 4 diverse jobs:
        # 1. Top-tier match (Score = 100.0) -> Python, FastAPI, Docker
        # 2. Strong match (Score = 75.0) -> Python, React (partially matches Python)
        # 3. Disqualified match (Score = 0.0) -> Java, Spring (0% skill overlap)
        # 4. Excluded job (Contains 'PHP') -> Excluded in filter
        self.mock_jobs = [
            Job(
                job_id="job-top",
                title="Principal Python Architect",
                company="CloudScale",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
                description="Architect scalable Python systems with FastAPI and Docker.",
                salary_range="$180,000",
            ),
            Job(
                job_id="job-strong",
                title="Fullstack Developer",
                company="AppWorks",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "React", "AWS"],
                description="Fullstack developer with Python and frontend experience.",
                salary_range="$130,000",
            ),
            Job(
                job_id="job-disqualified",
                title="Java Enterprise Developer",
                company="BigBank",
                location="New York",
                work_mode=WorkMode.ONSITE,
                tech_stack=["Java", "Spring", "Oracle"],
                description="Java Enterprise backend developer.",
                salary_range="$110,000",
            ),
            Job(
                job_id="job-excluded",
                title="Legacy PHP Developer",
                company="OldStack",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["PHP", "WordPress"],
                description="Maintain PHP legacy applications.",
                salary_range="$75,000",
            ),
        ]
        self.cache.update(self.mock_jobs)
        self.agent = MockLLMAgent(context=self.ctx)

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_run_pipeline_full_workflow(
        self,
        mock_delete,
        mock_execute,
        mock_preview,
        mock_bookmark,
    ):
        """Test end-to-end autonomous pipeline with top tier auto-apply, strong match bookmark, and disqualified cleanup."""
        mock_preview.return_value = ApplicationPreview(
            job_id="job-top",
            job_title="Principal Python Architect",
            company="CloudScale",
            application_method="1-Click Apply",
            fields_to_submit={"name": "Candidate"},
            warnings=[],
        )

        result = await self.agent.run_pipeline(
            tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
            exclude_keywords=["PHP"],
            top_tier_threshold=85,
            strong_match_threshold=70,
            disqualify_threshold=50,
            auto_apply=True,
        )

        self.assertTrue(result.success)
        self.assertIn("hiremetech", result.sources_found)
        self.assertEqual(result.total_jobs_fetched, 3)

        # Top-tier job verification
        self.assertEqual(len(result.top_tier_jobs), 1)
        self.assertEqual(result.top_tier_jobs[0]["job_id"], "job-top")
        self.assertIn("job-top", result.bookmarked_job_ids)
        self.assertIn("job-top", result.staged_apply_ids)
        self.assertIn("job-top", result.confirmed_apply_ids)

        # Disqualified jobs verification:
        self.assertIn("job-disqualified", result.deleted_job_ids)

        # Verification of step trace recording
        self.assertGreater(len(result.steps), 4)
        for i, step in enumerate(result.steps, 1):
            self.assertEqual(step.step_number, i)
            self.assertTrue(step.thought)
            self.assertTrue(step.tool_name)
            self.assertGreaterEqual(step.duration_ms, 0.0)

        self.assertGreater(result.execution_time_ms, 0.0)

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_run_pipeline_strong_matches_partition(
        self,
        mock_delete,
        mock_execute,
        mock_preview,
        mock_bookmark,
    ):
        """Test pipeline partitioning with both top-tier and strong match jobs."""
        mock_preview.return_value = ApplicationPreview(
            job_id="job-top",
            job_title="Principal Python Architect",
            company="CloudScale",
            application_method="1-Click Apply",
            fields_to_submit={},
            warnings=[],
        )

        result = await self.agent.run_pipeline(
            tech_stack=["Python", "FastAPI", "React", "Docker"],
            exclude_keywords=["PHP"],
            top_tier_threshold=85,
            strong_match_threshold=70,
            auto_apply=True,
        )

        self.assertTrue(result.success)
        # job-top has Python, FastAPI, Docker (3 out of 4 = 75.0 score) -> Strong match (70-79)
        self.assertIn("job-top", result.bookmarked_job_ids)
        self.assertNotIn("job-top", result.staged_apply_ids)
        self.assertNotIn("job-top", result.confirmed_apply_ids)

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_run_pipeline_auto_apply_disabled(
        self,
        mock_delete,
        mock_execute,
        mock_preview,
        mock_bookmark,
    ):
        """Test pipeline with auto_apply=False bookmarks top-tier jobs without applying."""
        result = await self.agent.run_pipeline(
            tech_stack=["Python", "FastAPI"],
            auto_apply=False,
        )

        self.assertTrue(result.success)
        self.assertIn("job-top", result.bookmarked_job_ids)
        self.assertEqual(result.staged_apply_ids, [])
        self.assertEqual(result.confirmed_apply_ids, [])
        mock_preview.assert_not_called()
        mock_execute.assert_not_called()

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_run_pipeline_with_cv_file(self, mock_delete, mock_bookmark):
        """Test pipeline with candidate CV file scoring."""
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("Senior Python Architect Resume:\nExperience in Python, FastAPI, PostgreSQL, Docker cloud systems.")
            cv_path = f.name

        agent = MockLLMAgent(cv_path=cv_path, context=self.ctx)
        result = await agent.run_pipeline(
            tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
            exclude_keywords=["PHP"],
            auto_apply=False,
        )

        self.assertTrue(result.success)
        self.assertIn("job-top", result.bookmarked_job_ids)
        self.assertIsNotNone(result.profile)
        self.assertIsInstance(result.profile, CandidateProfile)

    @patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock)
    @patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock)
    async def test_run_pipeline_purely_dynamic_cv(
        self,
        mock_delete,
        mock_execute,
        mock_preview,
        mock_bookmark,
    ):
        """Test pipeline executes seamlessly with purely dynamic CV extraction without explicit stack or exclude keywords."""
        mock_preview.return_value = ApplicationPreview(
            job_id="job-top",
            job_title="Principal Python Architect",
            company="CloudScale",
            application_method="1-Click Apply",
            fields_to_submit={},
            warnings=[],
        )

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("Senior Python Software Architect Resume:\nSkills: Python, FastAPI, Docker, PostgreSQL.\nTarget: AI Engineer, Backend Architect.")
            cv_path = f.name

        agent = MockLLMAgent(cv_path=cv_path, context=self.ctx)
        # Call run_pipeline with NO tech_stack, NO exclude_keywords, NO keywords, NO target_roles
        result = await agent.run_pipeline(
            cv_path=cv_path,
            auto_apply=True,
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.profile)
        self.assertEqual(result.profile.seniority_level, "Senior")
        self.assertTrue(any("python" in s.lower() for s in result.profile.top_skills or result.profile.skills))
        self.assertIn("job-top", result.bookmarked_job_ids)
        self.assertIn("job-top", result.confirmed_apply_ids)

        # Verify get_job_matches step argument propagation in step trace
        get_matches_step = next(s for s in result.steps if s.tool_name == "get_job_matches")
        self.assertEqual(get_matches_step.arguments.get("cv_path"), cv_path)
        self.assertTrue(get_matches_step.arguments.get("tech_stack"))
        self.assertTrue(get_matches_step.arguments.get("exclude_keywords"))

    async def test_run_pipeline_step_failure(self):
        """Test pipeline handling when a step returns success=False."""
        with patch.object(self.agent, "call_tool") as mock_call:
            mock_call.side_effect = [
                {"success": False, "message": "Failed to list sources", "error_code": "ERROR"},
                {"success": True, "data": []},
                {"success": True, "data": []},
            ]
            result = await self.agent.run_pipeline()
            self.assertFalse(result.success)
