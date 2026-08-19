"""Comprehensive unit and integration tests for the visual CLI runner script.

Validates argument parsing, observation summarization, Rich UI rendering,
step callbacks, JSON mode, remote client integration, and CLI entry points.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from job_mcp.models.schemas import CandidateProfile
from job_mcp.testing.mock_llm import MockLLMAgent, PipelineResult, StepTrace
from scripts.run_mock_llm_pipeline import (
    build_parser,
    execute_cli_pipeline,
    main,
    parse_comma_separated,
    render_header,
    render_job_insight_card,
    render_step_trace,
    render_summary_dashboard,
    resolve_default_cv,
    summarize_observation,
)


class TestArgumentParsing:
    """Test suite for CLI argument parser configuration and defaults."""

    def test_default_arguments(self):
        parser = build_parser()
        args = parser.parse_args([])

        assert args.stack is None
        assert args.exclude is None
        assert args.top_tier == 85
        assert args.strong_match == 70
        assert args.disqualify_threshold == 50
        assert args.auto_apply is True
        assert args.mode == "autonomous"
        assert args.work_mode is None
        assert args.location is None
        assert args.min_salary is None
        assert args.target_roles is None
        assert args.keywords is None
        assert args.force_refresh is False
        assert args.remote_url is None
        assert args.json is False
        assert args.verbose is False
        assert args.inspect is True

    def test_resolve_default_cv(self, tmp_path, monkeypatch):
        # 1. Test when DEFAULT_CV_PATH is set
        custom_cv = tmp_path / "custom_cv.pdf"
        custom_cv.write_text("Dummy content")
        monkeypatch.setenv("DEFAULT_CV_PATH", str(custom_cv))
        assert resolve_default_cv() == str(custom_cv)

        # 2. Test when CV_PATH is set
        monkeypatch.delenv("DEFAULT_CV_PATH", raising=False)
        monkeypatch.setenv("CV_PATH", str(custom_cv))
        assert resolve_default_cv() == str(custom_cv)

    def test_custom_arguments(self):
        parser = build_parser()
        args = parser.parse_args([
            "--cv", "test_cv.pdf",
            "--stack", "Python,FastAPI,PyTorch",
            "--exclude", "Junior,Intern",
            "--top-tier", "90",
            "--strong-match", "75",
            "--disqualify-threshold", "40",
            "--no-auto-apply",
            "--no-inspect",
            "--mode", "supervised",
            "--work-mode", "remote",
            "--location", "Tel Aviv",
            "--min-salary", "120000",
            "--target-roles", "AI Engineer,MLOps",
            "--keywords", "LLM,RAG",
            "--force-refresh",
            "--remote-url", "http://localhost:8000/mcp",
            "--json",
            "-v",
        ])

        assert args.cv == "test_cv.pdf"
        assert args.stack == "Python,FastAPI,PyTorch"
        assert args.exclude == "Junior,Intern"
        assert args.top_tier == 90
        assert args.strong_match == 75
        assert args.disqualify_threshold == 40
        assert args.auto_apply is False
        assert args.inspect is False
        assert args.mode == "supervised"
        assert args.work_mode == "remote"
        assert args.location == "Tel Aviv"
        assert args.min_salary == 120000
        assert args.target_roles == "AI Engineer,MLOps"
        assert args.keywords == "LLM,RAG"
        assert args.force_refresh is True
        assert args.remote_url == "http://localhost:8000/mcp"
        assert args.json is True
        assert args.verbose is True

    def test_inspect_flags(self):
        parser = build_parser()
        assert parser.parse_args([]).inspect is True
        assert parser.parse_args(["--inspect"]).inspect is True
        assert parser.parse_args(["--show-jobs"]).inspect is True
        assert parser.parse_args(["--no-inspect"]).inspect is False


class TestHelperFunctions:
    """Test suite for utility functions in the CLI runner."""

    def test_parse_comma_separated(self):
        assert parse_comma_separated(None) == []
        assert parse_comma_separated("") == []
        assert parse_comma_separated("   ") == []
        assert parse_comma_separated("Python") == ["Python"]
        assert parse_comma_separated("Python, FastAPI , Docker") == ["Python", "FastAPI", "Docker"]
        assert parse_comma_separated("a, , b,,c, ") == ["a", "b", "c"]

    def test_summarize_observation_error(self):
        resp = {"success": False, "message": "Connection refused", "error_code": "NETWORK_ERR"}
        summary = summarize_observation("get_job_matches", {}, resp)
        assert "FAILED:" in summary
        assert "Connection refused" in summary

    def test_summarize_observation_set_operation_mode(self):
        resp = {"success": True, "data": {"mode": "autonomous"}}
        summary = summarize_observation("set_operation_mode", {"mode": "autonomous"}, resp)
        assert "autonomous" in summary

    def test_summarize_observation_list_job_sources(self):
        resp = {
            "success": True,
            "data": {"sources": [{"name": "HireMeTech"}, {"name": "Comeet"}]},
        }
        summary = summarize_observation("list_job_sources", {}, resp)
        assert "Discovered" in summary
        assert "HireMeTech" in summary
        assert "Comeet" in summary

    def test_summarize_observation_get_job_matches(self):
        resp = {
            "success": True,
            "data": [
                {"job_id": "1", "source": "hiremetech"},
                {"job_id": "2", "source": "comeet"},
            ],
        }
        summary = summarize_observation("get_job_matches", {}, resp)
        assert "Aggregated" in summary
        assert "2" in summary

    def test_summarize_observation_filter_jobs(self):
        resp = {
            "success": True,
            "data": [
                {"job_id": "1", "match_score": 92.0},
                {"job_id": "2", "match_score": 75.0},
            ],
        }
        summary = summarize_observation("filter_jobs_by_preferences", {}, resp)
        assert "Scored" in summary
        assert "Top-Tier" in summary

    def test_summarize_observation_bookmark_job(self):
        resp = {"success": True, "data": {"job_id": "hmt_123"}}
        summary = summarize_observation("bookmark_job", {"job_id": "hmt_123"}, resp)
        assert "hmt_123" in summary
        assert "bookmarks" in summary

    def test_summarize_observation_auto_apply_job(self):
        resp = {"success": True, "data": {"company": "Acme AI"}}
        summary = summarize_observation("auto_apply_job", {"job_id": "hmt_123"}, resp)
        assert "hmt_123" in summary
        assert "Acme AI" in summary

    def test_summarize_observation_confirm_auto_apply(self):
        resp = {"success": True, "data": {"confirmation_id": "CONF-999"}}
        summary = summarize_observation("confirm_auto_apply", {"job_id": "hmt_123"}, resp)
        assert "CONF-999" in summary
        assert "submitted" in summary.lower()

    def test_summarize_observation_delete_job(self):
        resp = {"success": True, "data": {"job_id": "disq_123"}}
        summary = summarize_observation("delete_job", {"job_id": "disq_123"}, resp)
        assert "disq_123" in summary
        assert "Dismissed" in summary


class TestRichRendering:
    """Test suite verifying Rich console UI rendering."""

    @pytest.fixture
    def string_console(self):
        file = io.StringIO()
        console = Console(file=file, force_terminal=False, color_system=None, width=120)
        return console, file

    def test_render_header(self, string_console):
        console, file = string_console
        parser = build_parser()
        args = parser.parse_args(["--cv", "cv.pdf", "--work-mode", "remote", "--location", "Tel Aviv"])
        render_header(
            console,
            args,
            tech_stack=["Python", "FastAPI"],
            exclude_keywords=["Senior"],
        )
        output = file.getvalue()
        assert "HireMeTech" in output
        assert "Candidate CV" in output
        assert "Python, FastAPI" in output
        assert "Senior" in output
        assert "Top-Tier" in output

    def test_render_header_with_candidate_profile(self, string_console):
        console, file = string_console
        parser = build_parser()
        args = parser.parse_args(["--cv", "cv.pdf"])
        profile = CandidateProfile(
            skills=["Python", "FastAPI", "Docker", "LangGraph"],
            top_skills=["Python", "FastAPI", "Docker", "LangGraph"],
            seniority_level="Junior",
            target_roles=["AI Engineer", "Backend Developer"],
            suggested_exclusions=["Senior", "Lead", "10+ years"],
        )
        render_header(
            console,
            args,
            tech_stack=[],
            exclude_keywords=[],
            profile=profile,
        )
        output = file.getvalue()
        assert "Candidate CV" in output
        assert "Seniority Level" in output
        assert "Junior" in output
        assert "Top Skills" in output
        assert "Python" in output
        assert "Target Roles" in output
        assert "AI Engineer" in output
        assert "Senior, Lead" in output
        assert "Tech Stack Target" in output
        assert "(Dynamic from CV)" in output

    def test_render_header_dynamic_primary_stack(self, string_console):
        console, file = string_console
        parser = build_parser()
        args = parser.parse_args(["--cv", "cv.pdf"])
        profile = CandidateProfile(
            skills=["Python", "FastAPI", "Docker", "PostgreSQL", "React"],
            top_skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
            primary_stack=["Python", "FastAPI", "Docker"],
            seniority_level="Mid-Level",
        )
        render_header(
            console,
            args,
            tech_stack=[],
            exclude_keywords=[],
            profile=profile,
        )
        output = file.getvalue()
        assert "Primary Tech Stack" in output
        assert "Python, FastAPI, Docker" in output
        assert "Tech Stack Target" in output
        assert "Python, FastAPI, Docker (Dynamic from CV)" in output

    def test_render_header_explicit_stack_overrides_dynamic(self, string_console):
        console, file = string_console
        parser = build_parser()
        args = parser.parse_args(["--cv", "cv.pdf", "--stack", "Go,Rust"])
        profile = CandidateProfile(
            skills=["Python", "FastAPI"],
            primary_stack=["Python", "FastAPI"],
        )
        render_header(
            console,
            args,
            tech_stack=["Go", "Rust"],
            exclude_keywords=[],
            profile=profile,
        )
        output = file.getvalue()
        assert "Primary Tech Stack" in output
        assert "Go, Rust" in output
        assert "(Dynamic from CV)" not in output

    def test_render_step_trace(self, string_console):
        console, file = string_console
        step = StepTrace(
            step_number=1,
            thought="Discovering platforms for matching...",
            tool_name="list_job_sources",
            arguments={},
            response={"success": True, "data": {"sources": [{"name": "HireMeTech"}]}},
            duration_ms=42.5,
        )
        render_step_trace(console, step, verbose=True)
        output = file.getvalue()
        assert "Step 1" in output
        assert "list_job_sources" in output
        assert "Discovering platforms" in output
        assert "42.5 ms" in output

    def test_render_job_insight_card_top_tier(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=True,
            top_tier_jobs=[],
            confirmed_apply_ids=["cmt-1"],
            bookmarked_job_ids=["cmt-1"],
            staged_apply_ids=["cmt-1"],
        )
        job = {
            "job_id": "cmt-1",
            "title": "Junior Python & AI Agent Developer",
            "company": "CommIT",
            "source": "comeet",
            "sources": ["comeet", "hiremetech"],
            "match_score": 92.0,
            "seniority_level": "Junior",
            "work_mode": "hybrid",
            "location": "Tel Aviv",
            "salary_range": "$140,000 - $160,000",
            "match_reasons": [
                "CV matched skills: Python, FastAPI, LangGraph",
                "Work mode matches hybrid preference",
            ],
            "matched_skills": ["Python", "FastAPI", "Docker", "LangGraph"],
            "missing_skills": ["Kubernetes"],
            "description_summary": "Building autonomous AI agents and scalable backend services with FastAPI and Python.",
            "apply_url": "https://app.comeet.com/jobs/commit/cmt-1/apply",
        }

        render_job_insight_card(console, job, result, tier_label="🏆 Top-Tier Match", border_style="green")
        output = file.getvalue()

        assert "Junior Python & AI Agent Developer" in output
        assert "CommIT" in output
        assert "Comeet (Direct ATS)" in output
        assert "HireMeTech" in output
        assert "Score: 92/100" in output
        assert "Level: Junior" in output
        assert "Work Mode: Hybrid" in output
        assert "Location: Tel Aviv" in output
        assert "CV matched skills: Python, FastAPI, LangGraph" in output
        assert "Python" in output
        assert "FastAPI" in output
        assert "LangGraph" in output
        assert "Kubernetes" in output
        assert "Building autonomous AI agents" in output
        assert "https://app.comeet.com/jobs/commit/cmt-1/apply" in output
        assert "Applied & Confirmed" in output

    def test_render_job_insight_card_strong_match(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=True,
            strong_match_jobs=[],
            bookmarked_job_ids=["aj-2"],
        )
        job = {
            "job_id": "aj-2",
            "title": "Python Data Engineer",
            "company": "DataTech",
            "source": "alljobs",
            "sources": ["alljobs"],
            "match_score": 76.0,
            "seniority_level": "Mid-Level",
            "work_mode": "remote",
            "location": "Remote",
            "match_reasons": ["Target stack matched: Python, Pandas"],
            "matched_skills": ["Python", "Pandas"],
            "missing_skills": [],
            "description_summary": "Data pipelines with Python.",
            "url": "https://www.alljobs.co.il/job/aj-2",
        }

        render_job_insight_card(console, job, result, tier_label="⭐ Strong Match", border_style="yellow")
        output = file.getvalue()

        assert "Python Data Engineer" in output
        assert "DataTech" in output
        assert "AllJobs" in output
        assert "Score: 76/100" in output
        assert "Level: Mid-Level" in output
        assert "Work Mode: Remote" in output
        assert "Bookmarked for Review" in output
        assert "https://www.alljobs.co.il/job/aj-2" in output

    def test_render_summary_dashboard_populated(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=True,
            steps=[StepTrace(step_number=1, tool_name="list_job_sources", duration_ms=10.0)],
            sources_found=["hiremetech", "comeet"],
            total_jobs_fetched=25,
            top_tier_jobs=[
                {
                    "job_id": "job_1",
                    "title": "Staff AI Engineer",
                    "company": "Acme Tech",
                    "location": "Tel Aviv",
                    "work_mode": "hybrid",
                    "sources": ["hiremetech"],
                    "match_score": 95.0,
                    "matched_skills": ["Python", "FastAPI"],
                    "match_reasons": ["Target stack matched: Python, FastAPI"],
                    "description_summary": "Lead AI engineer designing distributed agentic services.",
                    "apply_url": "https://hireme.tech/apply/job_1",
                }
            ],
            strong_match_jobs=[
                {
                    "job_id": "job_2",
                    "title": "Python Backend Engineer",
                    "company": "Beta Cloud",
                    "location": "Remote",
                    "work_mode": "remote",
                    "sources": ["comeet"],
                    "match_score": 78.0,
                    "matched_skills": ["Python"],
                    "description_summary": "Backend engineering with Python and Docker.",
                    "url": "https://app.comeet.com/jobs/betacloud/job_2",
                }
            ],
            bookmarked_job_ids=["job_1", "job_2"],
            staged_apply_ids=["job_1"],
            confirmed_apply_ids=["job_1"],
            deleted_job_ids=["job_3"],
            execution_time_ms=1250.0,
        )

        render_summary_dashboard(
            console,
            result,
            top_tier_threshold=85,
            strong_match_threshold=70,
        )
        output = file.getvalue()
        assert "Pipeline Execution Summary Dashboard" in output
        assert "Staff AI Engineer" in output
        assert "Beta Cloud" in output
        assert "Acme Tech" in output
        assert "PIPELINE SUCCESS" in output
        assert "1.25 s" in output
        # Default inspect=True verifies Job Insight Cards are rendered
        assert "Detailed Job Insight Cards" in output
        assert "Lead AI engineer designing distributed agentic services" in output
        assert "Backend engineering with Python and Docker" in output
        assert "https://hireme.tech/apply/job_1" in output

    def test_render_summary_dashboard_no_inspect(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=True,
            top_tier_jobs=[
                {
                    "job_id": "job_1",
                    "title": "Staff AI Engineer",
                    "company": "Acme Tech",
                    "location": "Tel Aviv",
                    "sources": ["hiremetech"],
                    "match_score": 95.0,
                    "description_summary": "Lead AI engineer summary.",
                }
            ],
            strong_match_jobs=[],
            execution_time_ms=100.0,
        )

        render_summary_dashboard(
            console,
            result,
            top_tier_threshold=85,
            strong_match_threshold=70,
            inspect_jobs=False,
        )
        output = file.getvalue()
        assert "Pipeline Execution Summary Dashboard" in output
        assert "Staff AI Engineer" in output
        # Verify cards section is not rendered when inspect_jobs=False
        assert "Detailed Job Insight Cards" not in output
        assert "Lead AI engineer summary" not in output

    def test_render_summary_dashboard_flipped_arguments(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=True,
            top_tier_jobs=[],
            strong_match_jobs=[],
            execution_time_ms=10.0,
        )
        # Calling as render_summary_dashboard(result, console)
        render_summary_dashboard(result, console)
        output = file.getvalue()
        assert "Pipeline Execution Summary Dashboard" in output

    def test_render_summary_dashboard_empty(self, string_console):
        console, file = string_console
        result = PipelineResult(
            success=False,
            steps=[],
            sources_found=[],
            total_jobs_fetched=0,
            top_tier_jobs=[],
            strong_match_jobs=[],
            execution_time_ms=50.0,
        )

        render_summary_dashboard(
            console,
            result,
            top_tier_threshold=85,
            strong_match_threshold=70,
        )
        output = file.getvalue()
        assert "No jobs reached the Top-Tier" in output
        assert "No jobs in the Strong Match" in output
        assert "PIPELINE FAILED" in output


class TestExecuteCliPipeline:
    """Test suite for the async execution pipeline controller."""

    @pytest.mark.asyncio
    async def test_execute_cli_pipeline_in_memory(self):
        mock_result = PipelineResult(
            success=True,
            steps=[StepTrace(step_number=1, tool_name="list_job_sources", duration_ms=5.0)],
            sources_found=["hiremetech"],
            total_jobs_fetched=10,
            top_tier_jobs=[],
            strong_match_jobs=[],
            execution_time_ms=100.0,
        )

        parser = build_parser()
        args = parser.parse_args(["--stack", "Python,AI", "--no-auto-apply"])
        console = Console(file=io.StringIO(), force_terminal=False)

        with patch.object(MockLLMAgent, "run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            res = await execute_cli_pipeline(args, console=console)

            assert res == mock_result
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["tech_stack"] == ["Python", "AI"]
            assert call_kwargs["auto_apply"] is False
            assert callable(call_kwargs["step_callback"])

    @pytest.mark.asyncio
    async def test_execute_cli_pipeline_dynamic_cv(self, tmp_path):
        cv_file = tmp_path / "resume.txt"
        cv_file.write_text("Junior Developer with skills in Python, FastAPI, Docker, and React.")

        mock_result = PipelineResult(
            success=True,
            steps=[StepTrace(step_number=1, tool_name="list_job_sources", duration_ms=5.0)],
            sources_found=["hiremetech"],
            total_jobs_fetched=5,
            execution_time_ms=50.0,
        )

        parser = build_parser()
        args = parser.parse_args(["--cv", str(cv_file)])
        console = Console(file=io.StringIO(), force_terminal=False)

        with patch.object(MockLLMAgent, "run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            res = await execute_cli_pipeline(args, console=console)

            assert res == mock_result
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["cv_path"] == str(cv_file)
            # Verify tech_stack is passed as None, allowing dynamic resolution in MockLLMAgent/FastMCP
            assert call_kwargs.get("tech_stack") is None

    @pytest.mark.asyncio
    async def test_execute_cli_pipeline_json_mode(self, capsys):
        mock_result = PipelineResult(
            success=True,
            steps=[StepTrace(step_number=1, tool_name="list_job_sources", duration_ms=5.0)],
            sources_found=["hiremetech"],
            total_jobs_fetched=5,
            top_tier_jobs=[],
            strong_match_jobs=[],
            execution_time_ms=50.0,
        )

        parser = build_parser()
        args = parser.parse_args(["--json"])
        console = Console(file=io.StringIO(), force_terminal=False)

        with patch.object(MockLLMAgent, "run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            res = await execute_cli_pipeline(args, console=console)

            assert res == mock_result

        captured = capsys.readouterr()
        json_data = json.loads(captured.out)
        assert json_data["success"] is True
        assert json_data["total_jobs_fetched"] == 5

    @pytest.mark.asyncio
    async def test_execute_cli_pipeline_remote_url(self):
        mock_result = PipelineResult(
            success=True,
            steps=[],
            sources_found=["hiremetech"],
            total_jobs_fetched=10,
            execution_time_ms=80.0,
        )

        parser = build_parser()
        args = parser.parse_args(["--remote-url", "http://localhost:8000/mcp"])
        console = Console(file=io.StringIO(), force_terminal=False)

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("fastmcp.client.Client", return_value=mock_client):
            with patch.object(MockLLMAgent, "run_pipeline", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = mock_result
                res = await execute_cli_pipeline(args, console=console)

                assert res == mock_result
                mock_run.assert_called_once()


class TestMainCliEntryPoint:
    """Test suite for main() CLI entry point and exit codes."""

    def test_main_success(self):
        mock_result = PipelineResult(
            success=True,
            steps=[],
            execution_time_ms=10.0,
        )
        with patch("scripts.run_mock_llm_pipeline.execute_cli_pipeline", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            code = main(["--stack", "Python"])
            assert code == 0

    def test_main_failure(self):
        mock_result = PipelineResult(
            success=False,
            steps=[],
            execution_time_ms=10.0,
        )
        with patch("scripts.run_mock_llm_pipeline.execute_cli_pipeline", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            code = main(["--stack", "Python"])
            assert code == 1

    def test_main_keyboard_interrupt(self):
        with patch("scripts.run_mock_llm_pipeline.execute_cli_pipeline", side_effect=KeyboardInterrupt):
            code = main(["--stack", "Python"])
            assert code == 130

    def test_main_generic_exception(self):
        with patch("scripts.run_mock_llm_pipeline.execute_cli_pipeline", side_effect=RuntimeError("Test crash")):
            code = main(["--stack", "Python", "-v"])
            assert code == 1


class TestSubprocessExecution:
    """Integration test executing the CLI script via subprocess."""

    def test_cli_help_flag(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_mock_llm_pipeline.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "HireMeTech Autonomous Job Search LLM Agent Pipeline" in result.stdout
        assert "--stack" in result.stdout
        assert "--exclude" in result.stdout
        assert "--top-tier" in result.stdout
        assert "--auto-apply" in result.stdout
        assert "--inspect" in result.stdout
        assert "--no-inspect" in result.stdout
        assert "--remote-url" in result.stdout

    def test_cli_json_mode_mock_run(self):
        mock_result = PipelineResult(
            success=True,
            steps=[StepTrace(step_number=1, tool_name="list_job_sources", duration_ms=12.0)],
            sources_found=["hiremetech", "comeet", "alljobs"],
            total_jobs_fetched=30,
            top_tier_jobs=[],
            strong_match_jobs=[],
            execution_time_ms=450.0,
        )
        with patch("scripts.run_mock_llm_pipeline.MockLLMAgent.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            code = main(["--json", "--stack", "Python,FastAPI"])
            assert code == 0
