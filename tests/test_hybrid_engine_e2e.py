"""End-to-End Test Suite for Hybrid Autonomous Application Engine.

Validates:
1. FastMCP tool invocation for auto_apply_job, confirm_auto_apply, and get_application_history.
2. Safety guardrails fail-closed behavior (AUTO_APPLY_ENABLED=false, match score, location, daily cap).
3. Duplicate submission prevention even with force=True.
4. Strategy preview and execution routing across API, Easy Apply, and Browser strategies.
5. End-to-end integration with LLM gateway caching and semantic form mapper.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastmcp import Context

from job_mcp.core.api_client import JobCache
from job_mcp.core.application import (
    ApplicationLedger,
    HybridApplicationDispatcher,
    SemanticFormMapper,
)
from job_mcp.core.application.strategies import (
    ApiPostStrategy,
    BrowserPlaywrightStrategy,
    EasyApplyStrategy,
)
from job_mcp.core.auth import SessionManager
from job_mcp.core.llm import LLMCache, ResilientLLMGateway
from job_mcp.main import (
    _pending_applications,
    auto_apply_job,
    confirm_auto_apply,
    get_application_history,
    run_job_scout,
)

from job_mcp.models.ledger import ApplicationMethod, ApplicationStatus
from job_mcp.models.schemas import (
    ApplicationPreview,
    CandidateProfile,
    Job,
    WorkMode,
)
from job_mcp.sources import JobAggregator, SourceRegistry
from job_mcp.sources.hiremetech import HireMeTechSource


@pytest.fixture
def isolated_engine_context():
    """Fixture providing isolated lifespan context with in-memory ledger and dispatcher."""
    _pending_applications.clear()
    cache = JobCache(ttl_minutes=15)
    ledger = ApplicationLedger(db_path=":memory:")
    session_mgr = AsyncMock(spec=SessionManager)
    session_mgr._initialized = True
    session_mgr.context = MagicMock()
    session_mgr.is_healthy = AsyncMock(return_value=True)

    mock_locator = MagicMock()
    mock_locator.first = MagicMock()
    mock_locator.first.count = AsyncMock(return_value=0)
    mock_locator.first.is_visible = AsyncMock(return_value=False)
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.is_visible = AsyncMock(return_value=False)
    mock_page = MagicMock()
    mock_page.url = "https://app.hireme.tech/dashboard"
    mock_page.locator.return_value = mock_locator
    mock_page.wait_for_timeout = AsyncMock()
    session_mgr.get_page = AsyncMock(return_value=mock_page)

    dispatcher = HybridApplicationDispatcher(
        ledger=ledger,
        session_manager=session_mgr,
        max_daily_applications=5,
        min_match_score=85.0,
    )

    registry = SourceRegistry()
    registry.register(HireMeTechSource(session_manager=session_mgr))
    aggregator = JobAggregator(registry=registry, cache=cache)

    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {
        "session": session_mgr,
        "cache": cache,
        "registry": registry,
        "aggregator": aggregator,
        "ledger": ledger,
        "dispatcher": dispatcher,
    }

    yield ctx, cache, ledger, dispatcher, session_mgr
    ledger.close()


@pytest.fixture
def sample_candidate_profile() -> CandidateProfile:
    """Fixture providing a standard candidate profile."""
    return CandidateProfile(
        full_name="Gal Cohen",
        email="gal.cohen@example.com",
        phone="+972-54-1234567",
        skills=["Python", "FastAPI", "Docker", "PostgreSQL", "React", "TypeScript"],
        top_skills=["Python", "FastAPI", "Docker"],
        primary_stack=["Python", "PostgreSQL"],
        seniority_level="Senior",
        target_roles=["Senior Python Engineer", "Backend Tech Lead"],
    )


# =========================================================================
# 1. FastMCP Tool Execution (auto_apply_job, confirm_auto_apply, get_application_history)
# =========================================================================

@pytest.mark.asyncio
async def test_fastmcp_two_step_apply_and_history_e2e(
    isolated_engine_context,
    sample_candidate_profile: CandidateProfile,
    tmp_path: Path,
):
    """Verify 2-step apply workflow stages preview, confirms submission, and persists in ledger history."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    cv_file = tmp_path / "resume.txt"
    cv_file.write_text("Gal Cohen\nSkills: Python, FastAPI, Docker, PostgreSQL\nEmail: gal.cohen@example.com\n")

    job = Job(
        job_id="direct_senior_101",
        title="Senior Python Backend Engineer",
        company="InnoTech IL",
        source="direct_tech",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        tech_stack=["Python", "FastAPI", "Docker"],
        match_score=94.0,
    )
    cache.update([job])

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}), \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:

        mock_post.return_value = httpx.Response(
            200,
            json={"status": "success", "submission_id": "api_sub_101", "message": "Application received"},
        )

        # Step 1: Preview Application
        preview_resp = await auto_apply_job(
            job_id="direct_senior_101",
            cv_path=str(cv_file),
            ctx=ctx,
        )

        assert preview_resp["success"] is True
        assert "Application preview generated" in preview_resp["message"]
        assert "direct_senior_101" in _pending_applications
        preview_data = preview_resp["data"]
        assert preview_data["job_id"] == "direct_senior_101"
        assert preview_data["company"] == "InnoTech IL"
        assert preview_data["application_method"] == ApplicationMethod.API.value
        assert "applicant_name" in preview_data["fields_to_submit"]

        # Step 2: Confirm Application
        confirm_resp = await confirm_auto_apply(
            job_id="direct_senior_101",
            ctx=ctx,
        )

        assert confirm_resp["success"] is True
        assert "Successfully submitted application" in confirm_resp["message"]
        assert confirm_resp["data"]["submitted"] is True
        assert confirm_resp["data"]["job_id"] == "direct_senior_101"
        assert confirm_resp["data"]["method"] == ApplicationMethod.API.value
        assert "direct_senior_101" not in _pending_applications

        # Step 3: Inspect History via get_application_history
        history_resp = await get_application_history(limit=10, ctx=ctx)
        assert history_resp["success"] is True
        history_data = history_resp["data"]
        assert history_data["total"] == 1
        assert len(history_data["applications"]) == 1

        app_record = history_data["applications"][0]
        assert app_record["job_id"] == "direct_senior_101"
        assert app_record["company"] == "InnoTech IL"
        assert app_record["status"] == ApplicationStatus.SUCCESS.value
        assert app_record["method"] == ApplicationMethod.API.value
        assert app_record["match_score"] == 94.0


@pytest.mark.asyncio
async def test_confirm_auto_apply_without_preview_fails_safely(isolated_engine_context):
    """Verify confirm_auto_apply requires prior auto_apply_job preview."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    resp = await confirm_auto_apply(job_id="nonexistent_preview_job", ctx=ctx)
    assert resp["success"] is False
    assert resp["error_code"] == "NO_PENDING_PREVIEW"
    assert "No pending application preview found" in resp["message"]


# =========================================================================
# 2. Fail-Closed Guardrails Behavior
# =========================================================================

@pytest.mark.asyncio
async def test_guardrail_fail_closed_auto_apply_disabled(isolated_engine_context):
    """Verify fail-closed behavior when AUTO_APPLY_ENABLED=false blocks execution and logs to ledger."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    job = Job(
        job_id="job_guardrail_test_1",
        title="Python Engineer",
        company="SecureTech",
        source="comeet",
        location="Tel Aviv",
        work_mode=WorkMode.HYBRID,
        match_score=90.0,
    )
    cache.update([job])

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}):
        # Preview stages application with guardrail warning
        preview_res = await auto_apply_job(job_id="job_guardrail_test_1", ctx=ctx)
        assert preview_res["success"] is True
        warnings = preview_res["data"]["warnings"]
        assert any("AUTO_APPLY_ENABLED=false" in w for w in warnings)

        # Standard confirm without force fails closed
        confirm_res = await confirm_auto_apply(job_id="job_guardrail_test_1", force=False, ctx=ctx)
        assert confirm_res["success"] is False
        assert confirm_res["error_code"] == "AUTO_APPLY_DISABLED"
        assert "AUTO_APPLY_ENABLED=false" in confirm_res["message"]

        # Ledger records the blocked event
        history = await get_application_history(status="blocked", ctx=ctx)
        assert history["success"] is True
        assert history["data"]["total"] == 1
        assert history["data"]["applications"][0]["job_id"] == "job_guardrail_test_1"


@pytest.mark.asyncio
async def test_guardrail_bypass_with_force(isolated_engine_context):
    """Verify force=True bypasses AUTO_APPLY_ENABLED and match score guardrails."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    job = Job(
        job_id="job_force_override_1",
        title="Junior Developer",
        company="StartupTech",
        source="comeet",
        location="Tel Aviv",
        work_mode=WorkMode.ONSITE,
        match_score=65.0,  # Below standard 85.0 threshold
    )
    cache.update([job])

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}), \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:

        mock_post.return_value = httpx.Response(200, json={"status": "ok"})

        await auto_apply_job(job_id="job_force_override_1", ctx=ctx)
        confirm_res = await confirm_auto_apply(job_id="job_force_override_1", force=True, ctx=ctx)

        assert confirm_res["success"] is True
        assert confirm_res["data"]["submitted"] is True
        assert ledger.is_applied("job_force_override_1")


@pytest.mark.asyncio
async def test_guardrail_score_and_location_rejections(isolated_engine_context):
    """Verify match score and location constraints block submission when force=False."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    # 1. Low match score job
    low_score_job = Job(
        job_id="job_low_score",
        title="Go Developer",
        company="GoLang Co",
        source="comeet",
        location="Tel Aviv, Israel",
        match_score=50.0,
    )
    # 2. Foreign non-remote location job
    foreign_loc_job = Job(
        job_id="job_foreign_loc",
        title="Senior Python Engineer",
        company="EuroCorp",
        source="comeet",
        location="Berlin, Germany",
        work_mode=WorkMode.ONSITE,
        match_score=95.0,
    )
    cache.update([low_score_job, foreign_loc_job])

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        # Low score blocked
        await auto_apply_job(job_id="job_low_score", ctx=ctx)
        res_score = await confirm_auto_apply(job_id="job_low_score", force=False, ctx=ctx)
        assert res_score["success"] is False
        assert res_score["error_code"] == "LOW_MATCH_SCORE"

        # Foreign location blocked
        await auto_apply_job(job_id="job_foreign_loc", ctx=ctx)
        res_loc = await confirm_auto_apply(job_id="job_foreign_loc", force=False, ctx=ctx)
        assert res_loc["success"] is False
        assert res_loc["error_code"] == "LOCATION_CONSTRAINT_FAILED"


# =========================================================================
# 3. Duplicate Prevention Guardrail
# =========================================================================

@pytest.mark.asyncio
async def test_guardrail_duplicate_prevention_strict(isolated_engine_context):
    """Verify duplicate submission is strictly blocked even when force=True is passed."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    job = Job(
        job_id="job_dedup_strict_1",
        title="Senior AI Engineer",
        company="CognitiveLab",
        source="comeet",
        location="Herzliya, Israel",
        work_mode=WorkMode.HYBRID,
        match_score=95.0,
    )
    cache.update([job])

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}), \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:

        mock_post.return_value = httpx.Response(200, json={"status": "ok"})

        # Initial successful submission
        await auto_apply_job(job_id="job_dedup_strict_1", ctx=ctx)
        res1 = await confirm_auto_apply(job_id="job_dedup_strict_1", ctx=ctx)
        assert res1["success"] is True
        assert ledger.is_applied("job_dedup_strict_1")

        # Second attempt preview detects existing application
        preview_res = await auto_apply_job(job_id="job_dedup_strict_1", ctx=ctx)
        assert preview_res["success"] is True
        assert any("Duplicate Guardrail" in w for w in preview_res["data"]["warnings"])

        # Second attempt confirm (even with force=True) is strictly blocked
        res2 = await confirm_auto_apply(job_id="job_dedup_strict_1", force=True, ctx=ctx)
        assert res2["success"] is False
        assert res2["error_code"] == "DUPLICATE_APPLICATION"
        assert "already been applied to" in res2["message"]


# =========================================================================
# 4. Multi-Source Strategy Routing Integration
# =========================================================================

@pytest.mark.asyncio
async def test_multi_source_strategy_routing_e2e(isolated_engine_context):
    """Verify end-to-end strategy routing across API (Direct Tech), Easy Apply (LinkedIn), and Browser (Workday)."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    jobs = [
        Job(
            job_id="direct_eng_1",
            title="Backend Engineer",
            company="DirectCorp",
            source="direct_tech",
            location="Tel Aviv",
            match_score=90.0,
        ),
        Job(
            job_id="linkedin_eng_2",
            title="Fullstack Engineer",
            company="LinkedInCorp",
            source="linkedin",
            location="Remote",
            work_mode=WorkMode.REMOTE,
            match_score=92.0,
        ),
        Job(
            job_id="workday_eng_3",
            title="DevOps Engineer",
            company="EnterpriseWorkday",
            source="workday",
            location="Haifa, Israel",
            match_score=88.0,
        ),
    ]
    cache.update(jobs)

    preview_workday = ApplicationPreview(
        job_id="workday_eng_3",
        job_title="DevOps Engineer",
        company="EnterpriseWorkday",
        application_method=ApplicationMethod.BROWSER.value,
        fields_to_submit={"applicant_name": "Candidate"},
        warnings=[],
    )

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}), \
         patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post, \
         patch("job_mcp.core.browser.execute_application", new_callable=AsyncMock, return_value=True), \
         patch("job_mcp.core.browser.preview_application", new_callable=AsyncMock, return_value=preview_workday):

        mock_post.return_value = httpx.Response(200, json={"status": "ok"})

        # 1. API Strategy (Direct Tech)
        await auto_apply_job(job_id="direct_eng_1", ctx=ctx)
        res_api = await confirm_auto_apply(job_id="direct_eng_1", ctx=ctx)
        assert res_api["success"] is True
        assert res_api["data"]["method"] == ApplicationMethod.API.value

        # 2. Easy Apply Strategy (LinkedIn)
        await auto_apply_job(job_id="linkedin_eng_2", ctx=ctx)
        res_easy = await confirm_auto_apply(job_id="linkedin_eng_2", ctx=ctx)
        assert res_easy["success"] is True
        assert res_easy["data"]["method"] == ApplicationMethod.EASY_APPLY.value

        # 3. Browser Strategy (Workday)
        await auto_apply_job(job_id="workday_eng_3", ctx=ctx)
        res_browser = await confirm_auto_apply(job_id="workday_eng_3", ctx=ctx)
        assert res_browser["success"] is True
        assert res_browser["data"]["method"] == ApplicationMethod.BROWSER.value

    # Validate ledger records reflect correct methods
    entries = ledger.list_applications(limit=10, status=ApplicationStatus.SUCCESS)
    assert len(entries) == 3
    method_by_id = {e.job_id: e.method for e in entries}
    assert method_by_id["direct_eng_1"] == ApplicationMethod.API.value
    assert method_by_id["linkedin_eng_2"] == ApplicationMethod.EASY_APPLY.value
    assert method_by_id["workday_eng_3"] == ApplicationMethod.BROWSER.value


# =========================================================================
# 5. LLM Gateway & Semantic Form Mapper Caching Integration
# =========================================================================

@pytest.mark.asyncio
async def test_llm_gateway_caching_and_mapper_integration(tmp_path: Path):
    """Verify SemanticFormMapper with ResilientLLMGateway caches field resolutions and handles retries."""
    cache_db = tmp_path / "llm_cache.db"
    llm_cache = LLMCache(db_path=str(cache_db))
    gateway = ResilientLLMGateway(cache=llm_cache)

    profile = {
        "full_name": "Noa Ben-David",
        "email": "noa.bd@example.com",
        "phone": "+972-52-9876543",
        "skills": ["Python", "Go", "Kubernetes", "AWS"],
        "top_skills": ["Python", "Go"],
    }

    # 1. Deterministic heuristic mappings
    mapper = SemanticFormMapper(llm_gateway=gateway)
    name_res = await mapper.resolve_field("applicant_name", "Your Full Name", "text", profile=profile)
    assert name_res == "Noa Ben-David"

    email_res = await mapper.resolve_field("email_address", "Primary Email Address", "email", profile=profile)
    assert email_res == "noa.bd@example.com"

    auth_res = await mapper.resolve_field("israel_work_auth", "Are you legally authorized to work in Israel?", "radio", profile=profile)
    assert auth_res in ("Yes", "true", "Authorized")

    # 2. LLM Gateway Caching Verification
    with patch.object(gateway, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = "5+ years of production Python & Kubernetes experience"

        question = "How many years of Python experience do you have?"
        res1 = await gateway.ask_question(question=question, cv_context="Python, Kubernetes")
        assert "5+ years" in res1

        # Second identical prompt hits cache without calling _execute_with_retry
        mock_exec.reset_mock()
        res2 = await gateway.ask_question(question=question, cv_context="Python, Kubernetes")
        assert res2 == res1
        mock_exec.assert_not_called()

        assert llm_cache.get_cached_answer(question) == res1


@pytest.mark.asyncio
async def test_run_job_scout_hybrid_dispatch(isolated_engine_context):
    """Verify run_job_scout routes autonomous applications through HybridApplicationDispatcher."""
    ctx, cache, ledger, dispatcher, session_mgr = isolated_engine_context

    top_job = Job(

        job_id="comeet_top_ai_1",
        title="Senior AI Engineer",
        company="AI Labs",
        location="Tel Aviv, Israel",
        tech_stack=["Python", "FastAPI", "PyTorch"],
        source="comeet",
        description="Build LLM and RAG agents.",
        match_score=92.0,
    )
    cache.update([top_job])

    # 1. With AUTO_APPLY_ENABLED=false, run_job_scout stages and defers application
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}):
        res = await run_job_scout(
            sources=["comeet"],
            auto_apply=True,
            top_tier_threshold=85,
            ctx=ctx,
        )
        assert res["success"] is True
        data = res["data"]
        assert "comeet_top_ai_1" in data["deferred"]
        assert len(data["submitted"]) == 0

    # 2. With AUTO_APPLY_ENABLED=true, run_job_scout executes submission through strategy
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        with patch("job_mcp.core.application.strategies.api.httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = httpx.Response(
                200,
                json={"status": "ok", "application_id": "app_top_123"},
                request=httpx.Request("POST", "https://api.comeet.me/v1/apply"),
            )
            res2 = await run_job_scout(
                sources=["comeet"],
                auto_apply=True,
                top_tier_threshold=85,
                ctx=ctx,
            )
            assert res2["success"] is True
            data2 = res2["data"]
            assert "comeet_top_ai_1" in data2["submitted"]
            assert ledger.is_applied("comeet_top_ai_1") is True

