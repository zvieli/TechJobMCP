"""Unit and integration tests for HybridApplicationDispatcher and safety guardrails."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from job_mcp.core.application.dispatcher import HybridApplicationDispatcher
from job_mcp.core.application.ledger_service import ApplicationLedger
from job_mcp.models.ledger import ApplicationEntry, ApplicationMethod, ApplicationStatus
from job_mcp.models.schemas import CandidateProfile, Job, WorkMode


@pytest.fixture
def memory_ledger() -> ApplicationLedger:
    """Create an isolated in-memory ApplicationLedger instance."""
    ledger = ApplicationLedger(db_path=":memory:")
    yield ledger
    ledger.close()


@pytest.fixture
def sample_valid_job() -> Job:
    """A valid Israeli Job listing with match score meeting thresholds."""
    return Job(
        job_id="job_il_101",
        title="Senior Python Backend Engineer",
        company="Startup Nation Tech",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="comeet",
        match_score=90.0,
    )


@pytest.fixture
def sample_profile() -> CandidateProfile:
    """Candidate profile matching senior Python roles."""
    return CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        top_skills=["Python", "FastAPI"],
        primary_stack=["Python", "PostgreSQL"],
        seniority_level="Senior",
        target_roles=["Senior Python Engineer"],
    )


# ---------------------------------------------------------
# Guardrail 1: Duplicate Submission
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_duplicate_submission_blocked(
    memory_ledger: ApplicationLedger,
    sample_valid_job: Job,
    sample_profile: CandidateProfile,
):
    """Verify duplicate submission is blocked if job is already marked SUCCESS in ledger."""
    # Seed ledger with prior successful submission
    memory_ledger.record_application(
        ApplicationEntry(
            job_id=sample_valid_job.job_id,
            company=sample_valid_job.company,
            job_title=sample_valid_job.title,
            source=sample_valid_job.source,
            method=ApplicationMethod.API,
            status=ApplicationStatus.SUCCESS,
        )
    )

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        # Attempt standard apply
        result = await dispatcher.execute_application(sample_valid_job, sample_profile)
        assert result["success"] is False
        assert result["status"] == "blocked"
        assert result["error_code"] == "DUPLICATE_APPLICATION"
        assert "already been applied to" in result["message"]

        # Attempt with force=True (duplicate guardrail cannot be bypassed)
        result_force = await dispatcher.execute_application(sample_valid_job, sample_profile, force=True)
        assert result_force["success"] is False
        assert result_force["status"] == "blocked"
        assert result_force["error_code"] == "DUPLICATE_APPLICATION"


# ---------------------------------------------------------
# Guardrail 2: AUTO_APPLY_ENABLED Fail-Closed
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_auto_apply_disabled_by_default(
    memory_ledger: ApplicationLedger,
    sample_valid_job: Job,
    sample_profile: CandidateProfile,
):
    """Verify fail-closed behavior when AUTO_APPLY_ENABLED is unset or false."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)
        result = await dispatcher.execute_application(sample_valid_job, sample_profile)

        assert result["success"] is False
        assert result["status"] == "blocked"
        assert result["error_code"] == "AUTO_APPLY_DISABLED"
        assert "AUTO_APPLY_ENABLED=false" in result["message"]

        # Ledger records the blocked event
        entries = memory_ledger.list_applications(status=ApplicationStatus.BLOCKED)
        assert len(entries) == 1
        assert entries[0].job_id == sample_valid_job.job_id


@pytest.mark.asyncio
async def test_guardrail_auto_apply_overridden_with_force(
    memory_ledger: ApplicationLedger,
    sample_valid_job: Job,
    sample_profile: CandidateProfile,
):
    """Verify force=True allows submission even when AUTO_APPLY_ENABLED=false."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)
        result = await dispatcher.execute_application(sample_valid_job, sample_profile, force=True)

        assert result["success"] is True
        assert result["status"] == "success"
        assert memory_ledger.is_applied(sample_valid_job.job_id)


# ---------------------------------------------------------
# Guardrail 3: Daily Application Cap
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_daily_application_cap(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify daily cap prevents exceeding configured daily application limits."""
    dispatcher = HybridApplicationDispatcher(
        ledger=memory_ledger,
        max_daily_applications=2,
    )

    # Seed 2 successful applications today
    for i in range(2):
        memory_ledger.record_application(
            ApplicationEntry(
                job_id=f"prior_job_{i}",
                company="Company",
                job_title="Engineer",
                method=ApplicationMethod.API,
                status=ApplicationStatus.SUCCESS,
            )
        )

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        target_job = Job(
            job_id="job_cap_exceeded",
            title="Python Dev",
            company="Israel Cyber",
            location="Herzliya, Israel",
            match_score=95.0,
            source="comeet",
        )
        result = await dispatcher.execute_application(target_job, sample_profile)

        assert result["success"] is False
        assert result["status"] == "blocked"
        assert result["error_code"] == "DAILY_CAP_REACHED"
        assert "Daily application cap reached (2/2)" in result["message"]

        # With force=True, daily cap is bypassed
        result_force = await dispatcher.execute_application(target_job, sample_profile, force=True)
        assert result_force["success"] is True


# ---------------------------------------------------------
# Guardrail 4: Match Score Threshold
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_match_score_threshold(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify jobs with match score below 85.0 or missing score are blocked."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger, min_match_score=85.0)

        # 1. Job with match_score 78.0 (< 85.0)
        low_score_job = Job(
            job_id="job_low_score",
            title="Junior Dev",
            company="Startup IL",
            location="Haifa, Israel",
            match_score=78.0,
            source="comeet",
        )
        res_low = await dispatcher.execute_application(low_score_job, sample_profile)
        assert res_low["success"] is False
        assert res_low["error_code"] == "LOW_MATCH_SCORE"

        # 2. Job with missing match_score (None)
        no_score_job = Job(
            job_id="job_no_score",
            title="Junior Dev",
            company="Startup IL",
            location="Haifa, Israel",
            match_score=None,
            source="comeet",
        )
        res_no_score = await dispatcher.execute_application(no_score_job, sample_profile)
        assert res_no_score["success"] is False
        assert res_no_score["error_code"] == "LOW_MATCH_SCORE"

        # 3. Job with 0-1 scale (0.90 -> 90.0) passes
        scaled_score_job = Job(
            job_id="job_scaled_score",
            title="Senior Dev",
            company="Startup IL",
            location="Haifa, Israel",
            match_score=0.90,
            source="comeet",
        )
        res_scaled = await dispatcher.execute_application(scaled_score_job, sample_profile)
        assert res_scaled["success"] is True

        # 4. Low score job bypassed with force=True
        res_force = await dispatcher.execute_application(low_score_job, sample_profile, force=True)
        assert res_force["success"] is True


# ---------------------------------------------------------
# Guardrail 5: Location Constraint
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_location_constraint(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify non-Israeli / non-Remote locations are blocked."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger, max_daily_applications=100)

        # Allowed locations
        valid_locations = [
            ("loc_1", "Tel Aviv, Israel", WorkMode.ONSITE),
            ("loc_2", "Herzliya Pituach", WorkMode.HYBRID),
            ("loc_3", "Jerusalem, IL", WorkMode.ONSITE),
            ("loc_4", "Haifa", WorkMode.HYBRID),
            ("loc_5", "Remote, Worldwide", WorkMode.REMOTE),
            ("loc_6", "Beer Sheva", WorkMode.ONSITE),
            ("loc_7", "Yokneam Illit", WorkMode.HYBRID),
            ("loc_8", "Petah Tikva", WorkMode.ONSITE),
        ]
        for jid, loc, wm in valid_locations:
            job = Job(
                job_id=jid,
                title="Python Developer",
                company="Israeli Tech",
                location=loc,
                work_mode=wm,
                match_score=90.0,
                source="comeet",
            )
            res = await dispatcher.execute_application(job, sample_profile)
            assert res["success"] is True, f"Failed for valid location: {loc}"

        # Blocked non-Israeli locations
        blocked_locations = [
            ("block_1", "New York, NY, USA", WorkMode.ONSITE),
            ("block_2", "London, United Kingdom", WorkMode.ONSITE),
            ("block_3", "Berlin, Germany", WorkMode.HYBRID),
            ("block_4", "", WorkMode.ONSITE),
        ]
        for jid, loc, wm in blocked_locations:
            job = Job(
                job_id=jid,
                title="Python Developer",
                company="Foreign Corp",
                location=loc,
                work_mode=wm,
                match_score=90.0,
                source="comeet",
            )
            res = await dispatcher.execute_application(job, sample_profile)
            assert res["success"] is False, f"Did not block invalid location: '{loc}'"
            assert res["error_code"] == "LOCATION_CONSTRAINT_FAILED"

            # force=True bypasses location constraint
            res_force = await dispatcher.execute_application(job, sample_profile, force=True)
            assert res_force["success"] is True


# ---------------------------------------------------------
# Preview Application Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_preview_application(
    memory_ledger: ApplicationLedger,
    sample_valid_job: Job,
    sample_profile: CandidateProfile,
):
    """Test preview_application generates preview model and aggregates guardrail warnings."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "false"}):
        dispatcher = HybridApplicationDispatcher(
            ledger=memory_ledger,
            max_daily_applications=0,
            min_match_score=95.0,  # Higher than sample_valid_job.match_score (90.0)
        )
        preview = await dispatcher.preview_application(sample_valid_job, sample_profile)

        assert preview.job_id == sample_valid_job.job_id
        assert preview.application_method == ApplicationMethod.BROWSER.value
        # Warnings should contain auto apply disabled, daily cap, and score threshold
        assert any("AUTO_APPLY_ENABLED=false" in w for w in preview.warnings)
        assert any("Daily application cap reached" in w for w in preview.warnings)
        assert any("Match score (90.0) is below required threshold" in w for w in preview.warnings)


# ---------------------------------------------------------
# Strategy Routing & Execution in Dispatcher
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_executes_strategies_and_records_ledger(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Test dispatching across Direct Tech (API), LinkedIn (EasyApply), and Workday (Browser)."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        # 1. Direct Tech -> API Strategy
        api_job = Job(
            job_id="api_dispatch_1",
            title="Backend Engineer",
            company="DirectTechCompany",
            location="Tel Aviv, Israel",
            match_score=90.0,
            source="direct_tech",
        )
        res_api = await dispatcher.execute_application(api_job, sample_profile)
        assert res_api["success"] is True
        assert res_api["method"] == ApplicationMethod.API.value
        entry_api = memory_ledger.get_application(api_job.job_id)
        assert entry_api is not None
        assert entry_api.method == ApplicationMethod.API.value
        assert entry_api.status == ApplicationStatus.SUCCESS.value

        # 2. LinkedIn -> Easy Apply Strategy
        li_job = Job(
            job_id="li_dispatch_2",
            title="Fullstack Developer",
            company="LinkedInPartner",
            location="Herzliya, Israel",
            match_score=88.0,
            source="linkedin",
        )
        res_ea = await dispatcher.execute_application(li_job, sample_profile)
        assert res_ea["success"] is True
        assert res_ea["method"] == ApplicationMethod.EASY_APPLY.value
        entry_ea = memory_ledger.get_application(li_job.job_id)
        assert entry_ea is not None
        assert entry_ea.method == ApplicationMethod.EASY_APPLY.value
        assert entry_ea.status == ApplicationStatus.SUCCESS.value

        # 3. Workday -> Browser Playwright Strategy
        wd_job = Job(
            job_id="wd_dispatch_3",
            title="DevOps Architect",
            company="EnterpriseWorkday",
            location="Petah Tikva, Israel",
            match_score=91.0,
            source="workday",
        )
        res_wd = await dispatcher.execute_application(wd_job, sample_profile)
        assert res_wd["success"] is True
        assert res_wd["method"] == ApplicationMethod.BROWSER.value
        entry_wd = memory_ledger.get_application(wd_job.job_id)
        assert entry_wd is not None
        assert entry_wd.method == ApplicationMethod.BROWSER.value
        assert entry_wd.status == ApplicationStatus.SUCCESS.value


@pytest.mark.asyncio
async def test_dispatcher_strategy_exception_handling(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify unexpected strategy exceptions are recorded as FAILED in ledger."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        crash_job = Job(
            job_id="job_crash_1",
            title="Crash Engineer",
            company="BuggyCorp",
            location="Tel Aviv, Israel",
            match_score=90.0,
            source="direct_tech",
        )

        with patch("job_mcp.core.application.strategies.api.ApiPostStrategy.apply", side_effect=RuntimeError("Fatal strategy crash")):
            result = await dispatcher.execute_application(crash_job, sample_profile)
            assert result["success"] is False
            assert result["status"] == "failed"
            assert result["error_code"] == "EXECUTION_ERROR"
            assert "Fatal strategy crash" in result["message"]

            entry = memory_ledger.get_application(crash_job.job_id)
            assert entry is not None
            assert entry.status == ApplicationStatus.FAILED.value
            assert "Fatal strategy crash" in (entry.error_message or "")
