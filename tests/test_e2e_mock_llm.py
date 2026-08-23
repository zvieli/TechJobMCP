"""Automated End-to-End (E2E) Test Suite for Mock LLM Agent and Multi-Source Pipeline.

This module validates:
1. Full discovery-to-apply autonomous pipeline with realistic multi-source jobs (HireMeTech, Comeet, AllJobs).
2. CV scoring, tech stack ranking, and seniority exclusion filtering across LaTeX, PDF, and DOCX formats.
3. Two-step safety barrier preventing unconfirmed job applications.
4. Multi-source entity deduplication, source merging, tech stack consolidation, and ATS URL preservation.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastmcp import Context

from job_mcp.core.api_client import (
    JobCache,
    extract_cv_keywords,
    filter_jobs,
)
from job_mcp.core.application import (
    ApplicationLedger,
    HybridApplicationDispatcher,
)
from job_mcp.core.auth import SessionManager
from job_mcp.main import (
    _pending_applications,
    auto_apply_job,
    confirm_auto_apply,
)
from job_mcp.models.schemas import (
    ApplicationPreview,
    Job,
    JobPreferences,
    WorkMode,
)
from job_mcp.sources import (
    JobAggregator,
    create_default_registry,
)
from job_mcp.sources.dedup import deduplicate_jobs, merge_job_entities
from job_mcp.testing import MockLLMAgent, PipelineResult, StepTrace


def _create_mock_docx(file_path: str, text: str) -> None:
    """Create a valid minimal DOCX file with given text for CV testing."""
    xml_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(file_path, "w") as z:
        z.writestr("word/document.xml", xml_content)


def _create_mock_tex(file_path: str, text: str) -> None:
    """Create a LaTeX .tex file with given text for CV testing."""
    tex_content = f"""\\documentclass{{article}}
\\begin{{document}}
\\section{{Technical Skills}}
{text}
\\end{{document}}
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tex_content)


@pytest.fixture
def test_lifespan_context():
    """Fixture creating mocked SessionManager, JobCache, SourceRegistry, and JobAggregator."""
    _pending_applications.clear()
    cache = JobCache(ttl_minutes=15)

    session_mgr = AsyncMock(spec=SessionManager)
    session_mgr._initialized = True
    session_mgr.context = MagicMock()
    session_mgr.is_healthy = AsyncMock(return_value=True)
    mock_page = AsyncMock()
    mock_page.url = "https://app.hireme.tech/dashboard"
    session_mgr.get_page = AsyncMock(return_value=mock_page)
    session_mgr.ensure_ready = AsyncMock()

    registry = create_default_registry(session_manager=session_mgr)
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
    return ctx, cache, session_mgr, registry, aggregator


@pytest.mark.asyncio
async def test_e2e_full_discovery_to_apply_pipeline(test_lifespan_context):
    """Test 1: Comprehensive E2E discovery-to-apply autonomous pipeline.

    Requirements:
    - Initialize MockLLMAgent with candidate CV (or temp CV).
    - Set up test environment with realistic multi-source jobs (HireMeTech, Comeet, AllJobs).
    - Run agent.run_pipeline with tech stack and seniority exclusion preferences.
    - Assert:
      * All 6 pipeline steps executed successfully in sequence.
      * sources_found contains ['hiremetech', 'comeet'].
      * total_jobs_fetched > 0.
      * Top-tier jobs (score >= 85) are bookmarked and have staged & confirmed applications.
      * Strong matches (70-84) are bookmarked only without applying.
      * Disqualified jobs (< 50 or excluded seniority) are deleted.
      * Every StepTrace contains valid response dict with success: True and non-empty trace_id.
    """
    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context

    # Resolve CV path (use workspace CV if present, else temporary text CV)
    repo_cv = None
    for cand in [Path("cv.pdf"), Path("resume.pdf")]:
        if cand.exists():
            repo_cv = cand
            break
    if repo_cv:
        cv_path = str(repo_cv)
    else:
        temp_cv = tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False)
        temp_cv.write(
            "Candidate: Alex Rivera\n"
            "Skills: Python, FastAPI, LangGraph, LLM, NLP, LangChain, Docker, Git, Linux, React, TypeScript, Azure, Pandas, Scikit-Learn\n"
        )
        temp_cv.close()
        cv_path = temp_cv.name

    # Construct realistic jobs spanning HireMeTech, Comeet, and AllJobs
    multi_source_jobs = [
        # 1. Top-Tier Match (Score >= 85): Matches desired tech stack + extensive candidate CV skills
        Job(
            job_id="hmt-top-1",
            title="Python AI Engineer (FastAPI & LangGraph)",
            company="TechNova",
            source="hiremetech",
            sources=["hiremetech"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "LangGraph", "Docker", "Git", "Linux", "LLM", "NLP", "LangChain", "Pandas", "Scikit-Learn", "React", "TypeScript", "Azure", "TailwindCSS"],
            description="We are seeking an AI Engineer specializing in Python, FastAPI, and LangGraph. Experience with Docker, Git, Linux, LLMs, NLP, LangChain, Pandas, Scikit-Learn, React, TypeScript, and Azure required.",
            salary_range="$145,000 - $165,000",
            url="https://hireme.tech/jobs/hmt-top-1",
        ),
        # 2. Strong Match (70 <= Score < 85): Good match on tech stack + partial CV keywords
        Job(
            job_id="cmt-strong-1",
            title="Python Developer",
            company="CloudWorks",
            source="comeet",
            sources=["comeet"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "AI", "MongoDB"],
            description="Python developer with FastAPI, AI, and MongoDB services.",
            salary_range="$130,000",
            apply_url="https://app.comeet.com/jobs/cloudworks/cmt-strong-1",
        ),
        # 3. Disqualified - Excluded Seniority ("Senior" / "5+ years"): Should be dismissed
        Job(
            job_id="hmt-excluded-senior",
            title="Senior Python Architect",
            company="EnterpriseCloud",
            source="hiremetech",
            sources=["hiremetech"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "LangGraph", "Docker", "Git", "Linux", "LLM", "NLP"],
            description="Senior leader with 5+ years experience required.",
            salary_range="$190,000",
        ),
        # 4. Disqualified - Excluded Seniority ("Lead"): Should be dismissed
        Job(
            job_id="aj-excluded-lead",
            title="Lead AI Engineer",
            company="GlobalTech",
            source="alljobs",
            sources=["alljobs"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "AI"],
            description="Lead a team of 10 engineers building ML systems.",
            salary_range="$180,000",
        ),
        # 5. Disqualified - Low Match Score (< 50): Mainframe COBOL / Fortran
        Job(
            job_id="aj-disqualified-low",
            title="Legacy Mainframe Programmer",
            company="OldBank",
            source="alljobs",
            sources=["alljobs"],
            work_mode=WorkMode.ONSITE,
            tech_stack=["COBOL", "Fortran"],
            description="Maintain legacy mainframe banking systems.",
            salary_range="$75,000",
        ),
    ]

    cache.update(multi_source_jobs)

    with patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock) as mock_bookmark, \
         patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock) as mock_preview, \
         patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock) as mock_execute, \
         patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock) as mock_delete, \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_http_post:

        mock_http_post.return_value = httpx.Response(200, json={"status": "received", "application_id": "app-123"})

        mock_preview.return_value = ApplicationPreview(
            job_id="hmt-top-1",
            job_title="Python AI Engineer (FastAPI & LangGraph)",
            company="TechNova",
            application_method="1-Click Apply",
            fields_to_submit={"name": "Alex Rivera", "email": "candidate@example.com"},
            warnings=[],
        )

        agent = MockLLMAgent(cv_path=cv_path, context=ctx)
        result: PipelineResult = await agent.run_pipeline(
            tech_stack=["Python", "FastAPI", "AI", "LangGraph"],
            exclude_keywords=["Senior", "Lead", "5+ years"],
            top_tier_threshold=85,
            strong_match_threshold=70,
            disqualify_threshold=50,
            auto_apply=True,
        )

        # 1. Pipeline execution status
        assert result.success is True, f"Pipeline failed: {result}"
        assert result.execution_time_ms > 0.0
        assert result.profile is not None

        # 2. Source discovery (defaults to hiremetech, comeet)
        assert "hiremetech" in result.sources_found
        assert "comeet" in result.sources_found

        # 3. Job fetching (3 jobs fetched across sources matching search queries and seniority exclusions)
        assert result.total_jobs_fetched == 3

        # 4. Top-tier job verification (Score >= 85)
        top_ids = [j["job_id"] for j in result.top_tier_jobs]
        assert "hmt-top-1" in top_ids
        assert "hmt-top-1" in result.bookmarked_job_ids
        assert "hmt-top-1" in result.staged_apply_ids
        assert "hmt-top-1" in result.confirmed_apply_ids

        # 5. Strong match verification (70 <= Score < 85)
        strong_ids = [j["job_id"] for j in result.strong_match_jobs]
        assert "cmt-strong-1" in strong_ids
        assert "cmt-strong-1" in result.bookmarked_job_ids
        assert "cmt-strong-1" not in result.staged_apply_ids
        assert "cmt-strong-1" not in result.confirmed_apply_ids

        # 6. Disqualified cleanup verification (< 50)
        assert "aj-disqualified-low" in result.deleted_job_ids

        # 7. Step sequencing and StepTrace observability validation
        # Sequence: list_job_sources -> get_job_matches -> filter_jobs_by_preferences -> top bookmark/apply -> strong bookmark -> delete disqualified
        step_tools = [s.tool_name for s in result.steps]
        assert step_tools[0] == "list_job_sources"
        assert step_tools[1] == "get_job_matches"
        assert step_tools[2] == "filter_jobs_by_preferences"
        assert "bookmark_job" in step_tools
        assert "auto_apply_job" in step_tools
        assert "confirm_auto_apply" in step_tools
        assert "delete_job" in step_tools

        for i, step in enumerate(result.steps, start=1):
            assert step.step_number == i
            assert isinstance(step.thought, str) and len(step.thought) > 0
            assert isinstance(step.tool_name, str) and len(step.tool_name) > 0
            assert isinstance(step.response, dict)
            assert step.response.get("success") is True, f"Step {i} failed: {step.response}"
            trace_id = step.response.get("trace_id")
            assert isinstance(trace_id, str) and len(trace_id) > 0, f"Step {i} missing trace_id"
            assert step.duration_ms >= 0.0

        # Verify underlying browser automation was invoked as expected for HireMeTech portal jobs
        mock_bookmark.assert_called_once_with(session_mgr.get_page.return_value, "hmt-top-1")


@pytest.mark.asyncio
async def test_e2e_scoring_and_filtering_accuracy(test_lifespan_context):
    """Test 2: CV keyword extraction, scoring hierarchy, and seniority exclusion.

    Requirements:
    - Verify CV keyword extraction against LaTeX (.tex), Word (.docx), and PDF formats.
    - Verify that jobs matching tech stack and CV receive higher scores than irrelevant jobs.
    - Verify seniority exclusion filters out Lead/Senior titles even with matching tech keywords.
    """
    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context

    with tempfile.TemporaryDirectory() as tmpdir:
        # A. LaTeX CV test profile
        tex_path = os.path.join(tmpdir, "cv_engineer.tex")
        _create_mock_tex(
            tex_path,
            "Proficient in Python, FastAPI, Docker, Kubernetes, AWS, PostgreSQL, React, and Git.",
        )
        tex_keywords = extract_cv_keywords(tex_path)
        expected_tex_keywords = {"Python", "FastAPI", "Docker", "Kubernetes", "AWS", "PostgreSQL", "React", "Git"}
        assert expected_tex_keywords.issubset(set(tex_keywords)), f"Missing keywords in TeX: {tex_keywords}"

        # B. DOCX CV test profile
        docx_path = os.path.join(tmpdir, "cv_engineer.docx")
        _create_mock_docx(
            docx_path,
            "Software Engineer experienced in Python, LangChain, PyTorch, Scikit-Learn, Pandas, and Linux.",
        )
        docx_keywords = extract_cv_keywords(docx_path)
        expected_docx_keywords = {"Python", "LangChain", "PyTorch", "Scikit-Learn", "Pandas", "Linux"}
        assert expected_docx_keywords.issubset(set(docx_keywords)), f"Missing keywords in DOCX: {docx_keywords}"

        # C. Real PDF or Fallback PDF CV extraction
        repo_cv = None
        for cand in [Path("cv.pdf"), Path("resume.pdf")]:
            if cand.exists():
                repo_cv = cand
                break
        if repo_cv and repo_cv.exists():
            pdf_keywords = extract_cv_keywords(str(repo_cv))
            assert len(pdf_keywords) >= 5

        # D. Scoring hierarchy verification: full match > partial match > irrelevant match
        sample_jobs = [
            Job(
                job_id="job-full-match",
                title="Python FastAPI Backend Engineer",
                company="AlphaCo",
                tech_stack=["Python", "FastAPI", "Docker", "Kubernetes", "AWS", "PostgreSQL"],
                description="Developing scalable cloud services with Python, FastAPI, Docker, Kubernetes, AWS, PostgreSQL.",
            ),
            Job(
                job_id="job-partial-match",
                title="Python Developer",
                company="BetaCo",
                tech_stack=["Python", "MySQL"],
                description="Python developer working with relational databases.",
            ),
            Job(
                job_id="job-irrelevant",
                title="PHP WordPress Webmaster",
                company="OldCo",
                tech_stack=["PHP", "WordPress"],
                description="Maintaining legacy PHP and WordPress plugins.",
            ),
        ]

        prefs = JobPreferences(
            tech_stack=["Python", "FastAPI", "Docker"],
            keywords=["Kubernetes"],
            cv_path=tex_path,
        )

        ranked = filter_jobs(sample_jobs, prefs)
        assert len(ranked) == 3
        scores = {j.job_id: j.match_score for j in ranked}
        assert scores["job-full-match"] > scores["job-partial-match"]
        assert scores["job-partial-match"] > scores["job-irrelevant"]
        assert scores["job-full-match"] >= 85.0
        assert scores["job-irrelevant"] <= 10.0

        # E. Seniority exclusion verification
        senior_jobs = [
            Job(
                job_id="job-senior",
                title="Senior Python Backend Engineer",
                company="Enterprise Inc",
                tech_stack=["Python", "FastAPI", "Docker"],
                description="Senior developer driving architecture.",
            ),
            Job(
                job_id="job-lead",
                title="Tech Lead - Python Team",
                company="CloudScale",
                tech_stack=["Python", "FastAPI"],
                description="Lead engineering roadmap.",
            ),
            Job(
                job_id="job-years",
                title="Python Developer",
                company="FinTech",
                tech_stack=["Python", "FastAPI"],
                description="Requires 5+ years experience building financial platforms.",
            ),
            Job(
                job_id="job-mid",
                title="Python Backend Developer",
                company="StartupCo",
                tech_stack=["Python", "FastAPI", "Docker"],
                description="Collaborative Python developer for core product APIs.",
            ),
        ]

        exclude_prefs = JobPreferences(
            tech_stack=["Python", "FastAPI"],
            exclude_keywords=["Senior", "Lead", "5+ years"],
        )
        filtered_senior = filter_jobs(senior_jobs, exclude_prefs)
        filtered_ids = [j.job_id for j in filtered_senior]

        assert "job-mid" in filtered_ids
        assert "job-senior" not in filtered_ids
        assert "job-lead" not in filtered_ids
        assert "job-years" not in filtered_ids
        assert len(filtered_senior) == 1


@pytest.mark.asyncio
async def test_e2e_two_step_safety_barrier(test_lifespan_context):
    """Test 3: Safety barrier preventing unconfirmed job applications.

    Requirements:
    - Direct test: calling confirm_auto_apply without auto_apply_job produces NO_PENDING_PREVIEW error code.
    - Calling auto_apply_job stages preview in pending store.
    - Subsequent confirm_auto_apply consumes preview and executes successfully.
    - Calling confirm_auto_apply for an un-staged job ID fails safely.
    """
    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context
    _pending_applications.clear()

    mock_execute = AsyncMock()
    mock_preview = AsyncMock()

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        # 1. Calling confirm_auto_apply directly without preview fails safely
        resp_fail = await confirm_auto_apply(job_id="unpreviewed-job-999", ctx=ctx)
        assert resp_fail["success"] is False
        assert resp_fail["error_code"] == "NO_PENDING_PREVIEW"
        assert "No pending application preview found" in resp_fail["message"]

        # 2. Call via MockLLMAgent tool execution
        agent = MockLLMAgent(context=ctx)
        agent_fail_resp = await agent.call_tool(
            "confirm_auto_apply",
            arguments={"job_id": "unpreviewed-job-999"},
            thought="Attempting confirmation without preview",
        )
        assert agent_fail_resp["success"] is False
        assert agent_fail_resp["error_code"] == "NO_PENDING_PREVIEW"

        # 3. Proper 2-step workflow: preview first, then confirm
        # Step 1: Preview
        preview_resp = await agent.call_tool(
            "auto_apply_job",
            arguments={"job_id": "safe-job-123"},
            thought="Staging application preview",
        )
        assert preview_resp["success"] is True
        assert preview_resp["data"]["job_id"] == "safe-job-123"
        assert "safe-job-123" in _pending_applications

        # Step 2: Confirm
        confirm_resp = await agent.call_tool(
            "confirm_auto_apply",
            arguments={"job_id": "safe-job-123"},
            thought="Confirming application submission",
        )
        assert confirm_resp["success"] is True
        assert confirm_resp["data"]["job_id"] == "safe-job-123"
        assert "safe-job-123" not in _pending_applications


@pytest.mark.asyncio
async def test_e2e_multi_source_dedup_integrity(test_lifespan_context):
    """Test 4: Multi-source deduplication, source merging, and ATS link priority.

    Requirements:
    - Feed duplicate jobs across Comeet and HireMeTech (and AllJobs).
    - Verify pipeline deduplicates them, preserves sources: ['comeet', 'hiremetech'],
      combines tech stack, and preserves the highest match score, ATS apply_url, and longest description.
    """
    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context

    job_hmt = Job(
        job_id="hmt-job-101",
        title="Senior Python Backend Developer (m/f/d)",
        company="Acme Corporation Ltd.",
        source="hiremetech",
        sources=["hiremetech"],
        work_mode=WorkMode.REMOTE,
        tech_stack=["Python", "FastAPI"],
        description="Short description of backend role.",
        match_score=85.0,
        url="https://hireme.tech/jobs/101",
        apply_url="https://hireme.tech/apply/101",
    )

    job_comeet = Job(
        job_id="cmt-job-202",
        title="Senior Python Backend Developer (Hybrid)",
        company="Acme Corporation Inc.",
        source="comeet",
        sources=["comeet"],
        work_mode=WorkMode.HYBRID,
        tech_stack=["FastAPI", "Docker", "PostgreSQL", "AWS"],
        description="Comprehensive in-depth description with team scope, requirements, and microservices stack.",
        match_score=94.5,
        url="https://app.comeet.com/jobs/acme/202",
        apply_url="https://app.comeet.com/jobs/acme/202/apply",
        department="Core Infrastructure",
    )

    job_alljobs = Job(
        job_id="aj-job-303",
        title="Senior Python Backend Developer",
        company="Acme Corp",
        source="alljobs",
        sources=["alljobs"],
        work_mode=None,
        tech_stack=["Python", "Redis"],
        description="Medium description from AllJobs.",
        match_score=78.0,
        url="https://alljobs.co.il/jobs/303",
        apply_url="https://alljobs.co.il/apply/303",
    )

    distinct_job = Job(
        job_id="cmt-job-404",
        title="Frontend React Specialist",
        company="Beta Soft",
        source="comeet",
        sources=["comeet"],
        tech_stack=["React", "TypeScript", "TailwindCSS"],
        description="Frontend specialist building modern web applications.",
        match_score=82.0,
    )

    raw_jobs = [job_hmt, job_comeet, job_alljobs, distinct_job]
    deduped = deduplicate_jobs(raw_jobs)

    # 1. Deduplication count check (3 Acme jobs merged into 1 + 1 Beta Soft job)
    assert len(deduped) == 2

    # 2. Find the merged Acme entity
    merged_acme = next(j for j in deduped if "acme" in j.company.lower())

    # 3. Verify sources union
    assert set(merged_acme.sources) == {"hiremetech", "comeet", "alljobs"}

    # 4. Verify tech stack union
    expected_tech = {"Python", "FastAPI", "Docker", "PostgreSQL", "AWS", "Redis"}
    assert set(merged_acme.tech_stack) == expected_tech

    # 5. Verify highest match score preservation
    assert merged_acme.match_score == 94.5

    # 6. Verify ATS direct apply_url priority (Comeet ATS > HireMeTech > AllJobs)
    assert "comeet.com" in (merged_acme.apply_url or "")

    # 7. Verify longest comprehensive description preservation
    assert merged_acme.description == job_comeet.description

    # 8. Verify department metadata preservation
    assert merged_acme.department == "Core Infrastructure"

    # 9. Verify distinct entity remained untouched
    beta_job = next(j for j in deduped if "beta" in j.company.lower())
    assert beta_job.job_id == "cmt-job-404"
    assert beta_job.sources == ["comeet"]


@pytest.mark.asyncio
async def test_e2e_cli_visual_runner_insight_cards(test_lifespan_context):
    """Test 5: E2E Visual CLI runner Job Insight Cards rendering and inspect mode default.

    Validates:
    - Default inspect mode produces Rich Job Insight Cards for top-tier and strong matches.
    - Insight cards display title, company, source badges, score breakdown, summary, and apply URLs.
    - Disabling inspect mode (--no-inspect) omits detailed insight cards while keeping summary tables.
    """
    import io
    from rich.console import Console
    from scripts.run_mock_llm_pipeline import build_parser, execute_cli_pipeline, render_summary_dashboard

    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context

    # Realistic mock jobs
    jobs = [
        Job(
            job_id="e2e-top-1",
            title="AI Agent Engineer (FastAPI & LangGraph)",
            company="CommIT Dynamics",
            source="comeet",
            sources=["comeet", "hiremetech"],
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "FastAPI", "Docker", "LangGraph", "React"],
            description="Developing agentic AI systems and FastAPI microservices in hybrid Tel Aviv office.",
            salary_range="$150,000 - $170,000",
            location="Tel Aviv",
            apply_url="https://app.comeet.com/jobs/commit/e2e-top-1/apply",
        ),
        Job(
            job_id="e2e-strong-2",
            title="Backend Python Developer",
            company="AlphaCloud",
            source="alljobs",
            sources=["alljobs"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "PostgreSQL"],
            description="Backend developer building cloud services with Python and PostgreSQL.",
            salary_range="₪35,000 / mo",
            location="Remote",
            url="https://www.alljobs.co.il/job/e2e-strong-2",
        ),
    ]
    cache.update(jobs)

    parser = build_parser()
    args = parser.parse_args(["--mode", "mock", "--no-auto-apply"])
    assert args.inspect is True

    output_stream = io.StringIO()
    console = Console(file=output_stream, force_terminal=False, color_system=None, width=120)

    # 1. Execute CLI pipeline with default inspect=True
    with patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock):
        result = await execute_cli_pipeline(args, console=console)

    assert result.success is True
    output = output_stream.getvalue()

    # 2. Verify Job Insight Cards rendering
    assert "Detailed Job Insight Cards" in output
    assert "AI Application Engineer (FastAPI & LangGraph)" in output
    assert "TechNova Labs" in output
    assert "Cognitive Dynamics" in output
    assert "Comeet (Direct ATS)" in output
    assert "HireMeTech" in output
    assert "Score:" in output
    assert "Level:" in output
    assert "Work Mode: Remote" in output
    assert "Location: Tel Aviv" in output
    assert "Python" in output
    assert "FastAPI" in output
    assert "https://hireme.tech/jobs/hmt-top-1" in output
    assert "CloudWorks Software" in output
    assert "InnoTech Solutions" in output
    assert "AllJobs" in output

    # 3. Verify --no-inspect disables cards
    no_inspect_stream = io.StringIO()
    no_inspect_console = Console(file=no_inspect_stream, force_terminal=False, color_system=None, width=120)
    render_summary_dashboard(
        no_inspect_console,
        result,
        top_tier_threshold=85,
        strong_match_threshold=70,
        inspect_jobs=False,
    )
    no_inspect_output = no_inspect_stream.getvalue()
    assert "Pipeline Execution Summary Dashboard" in no_inspect_output
    assert "AI Application Engineer" in no_inspect_output
    assert "TechNova Labs" in no_inspect_output
    assert "Detailed Job Insight Cards" not in no_inspect_output


@pytest.mark.asyncio
async def test_e2e_dynamic_cv_driven_mock_pipeline(test_lifespan_context):
    """Test 6: Purely dynamic CV-driven MockLLMAgent autonomous pipeline.

    Validates:
    - Running agent with cv_path and NO explicit tech_stack or exclude_keywords flags.
    - Automatic CandidateProfile extraction with detected seniority, top skills, search queries, and suggested exclusions.
    - Propagation of dynamic queries and adaptive exclusions to tool executions.
    - Autonomous top-tier staging & confirmation, strong match bookmarking, and disqualified cleanup.
    """
    ctx, cache, session_mgr, registry, aggregator = test_lifespan_context

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(
            "Junior AI Software Developer Resume:\n"
            "Technical Skills: Python, FastAPI, LangGraph, React, TypeScript, Docker, Linux, Git.\n"
            "Interests: Agentic AI, Autonomous Workflows, LLM Microservices.\n"
        )
        cv_path = f.name

    jobs = [
        Job(
            job_id="dyn-top-1",
            title="Junior Python & AI Agent Developer",
            company="CommIT Dynamics",
            source="comeet",
            sources=["comeet", "hiremetech"],
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "FastAPI", "LangGraph", "Docker", "React", "TypeScript", "Linux", "Git"],
            description="Developing agentic AI systems and FastAPI microservices.",
            salary_range="$145,000 - $160,000",
            location="Tel Aviv",
            apply_url="https://app.comeet.com/jobs/commit/dyn-top-1/apply",
        ),
        Job(
            job_id="dyn-strong-2",
            title="Python Developer",
            company="CloudWorks",
            source="comeet",
            sources=["comeet"],
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "PostgreSQL"],
            description="Python developer with FastAPI.",
            salary_range="$125,000",
            apply_url="https://app.comeet.com/jobs/cloudworks/dyn-strong-2",
        ),
        Job(
            job_id="dyn-disq-senior",
            title="Senior Enterprise Python Architect",
            company="LegacyCorp",
            source="hiremetech",
            sources=["hiremetech"],
            work_mode=WorkMode.ONSITE,
            tech_stack=["Python", "FastAPI"],
            description="Senior architect with 10+ years experience required.",
            salary_range="$190,000",
        ),
    ]
    cache.update(jobs)

    with patch("job_mcp.main.browser_bookmark_job", new_callable=AsyncMock) as mock_bookmark, \
         patch("job_mcp.main.browser_preview_application", new_callable=AsyncMock) as mock_preview, \
         patch("job_mcp.main.browser_execute_application", new_callable=AsyncMock) as mock_execute, \
         patch("job_mcp.main.browser_delete_job", new_callable=AsyncMock) as mock_delete, \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_http_post:

        mock_http_post.return_value = httpx.Response(200, json={"status": "received", "application_id": "app-123"})

        mock_preview.return_value = ApplicationPreview(
            job_id="dyn-top-1",
            job_title="Junior Python & AI Agent Developer",
            company="CommIT Dynamics",
            application_method="1-Click Apply",
            fields_to_submit={"name": "Candidate"},
            warnings=[],
        )

        agent = MockLLMAgent(cv_path=cv_path, context=ctx)
        # Calling with NO tech_stack, NO exclude_keywords, NO target_roles, NO keywords
        result = await agent.run_pipeline(
            cv_path=cv_path,
            auto_apply=True,
        )

        assert result.success is True
        assert result.profile is not None
        assert result.profile.seniority_level == "Junior"
        assert len(result.profile.primary_stack) > 0
        assert any("python" in s.lower() for s in result.profile.primary_stack)
        assert "Senior" in result.profile.suggested_exclusions
        assert "dyn-top-1" in result.bookmarked_job_ids
        assert "dyn-top-1" in result.confirmed_apply_ids

        # Verify dynamic effective tech stack was propagated to get_job_matches and filter_jobs_by_preferences
        matches_step = next(s for s in result.steps if s.tool_name == "get_job_matches")
        assert matches_step.arguments.get("tech_stack") == list(result.profile.primary_stack or result.profile.top_skills or result.profile.skills)
        filter_step = next(s for s in result.steps if s.tool_name == "filter_jobs_by_preferences")
        assert filter_step.arguments.get("tech_stack") == list(result.profile.primary_stack or result.profile.top_skills or result.profile.skills)


