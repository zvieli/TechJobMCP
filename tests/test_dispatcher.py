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


@pytest.mark.asyncio
async def test_guardrail_match_score_fallback_when_none(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify that jobs with match_score=None calculate skill overlap fallback score instead of being blocked."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger, min_match_score=85.0)

        # 1. Job with match_score=None but matching skills in title/tech_stack
        matching_job = Job(
            job_id="job_none_score_matching",
            title="Senior Python Backend Developer",
            company="Comeet Startup",
            location="Tel Aviv, Israel",
            tech_stack=["Python", "FastAPI"],
            description="We are looking for a Python engineer with FastAPI expertise.",
            match_score=None,
            source="comeet",
        )
        # Direct _validate_match_score check
        assert dispatcher._validate_match_score(matching_job, sample_profile) is True
        assert matching_job.match_score is not None
        assert matching_job.match_score >= 85.0

        # Autonomous execution should succeed rather than being blocked
        res = await dispatcher.execute_application(matching_job, sample_profile)
        assert res["success"] is True
        assert res["status"] == "success"

        # Check ledger recorded the calculated fallback score
        entry = memory_ledger.get_application(matching_job.job_id)
        assert entry is not None
        assert entry.match_score is not None
        assert entry.match_score >= 85.0

        # 2. Job with match_score=None but completely irrelevant skills should fail match validation
        irrelevant_job = Job(
            job_id="job_none_score_irrelevant",
            title="Rust Systems Engineer",
            company="Embedded Corp",
            location="Tel Aviv, Israel",
            tech_stack=["Rust", "C++"],
            description="Developing bare metal firmware in Rust and C++.",
            match_score=None,
            source="comeet",
        )
        assert dispatcher._validate_match_score(irrelevant_job, sample_profile) is False

        res_irrelevant = await dispatcher.execute_application(irrelevant_job, sample_profile)
        assert res_irrelevant["success"] is False
        assert res_irrelevant["error_code"] == "LOW_MATCH_SCORE"

        # 3. Validating match score without profile should return False when score is None
        unscored_job = Job(
            job_id="job_no_profile",
            title="Python Developer",
            company="Startup",
            location="Tel Aviv",
            match_score=None,
        )
        assert dispatcher._validate_match_score(unscored_job, None) is False

        # 4. Normalization of 0-1 scale to 0-100 scale updates job.match_score
        scaled_job = Job(
            job_id="job_scaled_0_1",
            title="Engineer",
            company="Startup",
            location="Tel Aviv",
            match_score=0.92,
        )
        assert dispatcher._validate_match_score(scaled_job) is True
        assert scaled_job.match_score == 92.0


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


@pytest.mark.asyncio
async def test_guardrail_location_constraint_hebrew_locations(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify Israeli cities and regions specified in Hebrew pass location validation."""
    dispatcher = HybridApplicationDispatcher(ledger=memory_ledger, max_daily_applications=100)

    hebrew_locations = [
        "תל אביב - יפו",
        "מרכז",
        "חיפה",
        "ישראל",
        "הרצליה",
        "רמת גן",
        "פתח תקווה",
        "פתח תקוה",
        "גוש דן",
        "שרון",
        "באר שבע",
        "ירושלים",
        "נתניה",
        "רעננה",
        "חולון",
        "ראשון לציון",
        "רחובות",
        "כפר סבא",
        "הוד השרון",
        "בת ים",
        "מודיעין",
        "גבעתיים",
        "נס ציונה",
        "בני ברק",
    ]

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        for idx, loc in enumerate(hebrew_locations):
            job = Job(
                job_id=f"job_hebrew_{idx}",
                title="Fullstack Developer",
                company="Israeli High-Tech",
                location=loc,
                work_mode=WorkMode.HYBRID,
                match_score=90.0,
                source="comeet",
            )
            # Direct _validate_location check
            assert dispatcher._validate_location(job) is True, f"Failed _validate_location for Hebrew location: {loc}"

            # Full execute_application check
            res = await dispatcher.execute_application(job, sample_profile)
            assert res["success"] is True, f"Failed execute_application for Hebrew location: {loc}"
            assert res["status"] == "success"



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


@pytest.mark.asyncio
async def test_dispatcher_fallback_from_api_to_browser_on_endpoint_not_an_api(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify dispatcher gracefully falls back from API to Browser when endpoint is not an API."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        api_job = Job(
            job_id="job_fallback_api_1",
            title="Senior Backend Engineer",
            company="FallbackCorp",
            location="Tel Aviv, Israel",
            match_score=92.0,
            source="direct_tech",
            apply_url="https://fallbackcorp.com/careers/job1",
        )

        api_mock_return = {
            "success": False,
            "job_id": api_job.job_id,
            "method": ApplicationMethod.API.value,
            "status": "failed",
            "error_code": "ENDPOINT_NOT_AN_API",
            "error": "ATS endpoint returned HTML content (200): target appears to be a frontend web page",
        }

        browser_mock_return = {
            "success": True,
            "job_id": api_job.job_id,
            "method": ApplicationMethod.BROWSER.value,
            "status": "success",
            "submission_id": "pw_sub_fallback123",
            "response": {"message": "Submitted via browser fallback"},
        }

        with patch(
            "job_mcp.core.application.strategies.api.ApiPostStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_api_apply, patch(
            "job_mcp.core.application.strategies.browser.BrowserPlaywrightStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_browser_apply:
            mock_api_apply.return_value = api_mock_return
            mock_browser_apply.return_value = browser_mock_return

            result = await dispatcher.execute_application(api_job, sample_profile)

            mock_api_apply.assert_called_once()
            mock_browser_apply.assert_called_once_with(api_job, sample_profile, cv_path=None)

            assert result["success"] is True
            assert result["method"] == ApplicationMethod.BROWSER.value
            assert result["status"] == "success"

            # Check ledger entry
            entry = memory_ledger.get_application(api_job.job_id)
            assert entry is not None
            assert entry.status == ApplicationStatus.SUCCESS.value
            assert entry.method == ApplicationMethod.BROWSER.value
            assert "fallback" in (entry.notes or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["HTTP_405", "HTTP_301"])
async def test_dispatcher_fallback_from_api_to_browser_on_http_redirect_or_not_allowed(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
    error_code: str,
):
    """Verify dispatcher gracefully falls back from API to Browser on HTTP 405 or 301."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        api_job = Job(
            job_id=f"job_fallback_{error_code.lower()}",
            title="Cloud Architect",
            company="HttpErrorCorp",
            location="Remote",
            work_mode=WorkMode.REMOTE,
            match_score=89.0,
            source="direct_tech",
            apply_url="https://httperrorcorp.com/apply",
        )

        api_mock_return = {
            "success": False,
            "job_id": api_job.job_id,
            "method": ApplicationMethod.API.value,
            "status": "failed",
            "error_code": error_code,
            "error": f"ATS endpoint returned {error_code}",
        }

        browser_mock_return = {
            "success": True,
            "job_id": api_job.job_id,
            "method": ApplicationMethod.BROWSER.value,
            "status": "success",
            "submission_id": f"pw_sub_{error_code}",
            "response": {"message": "Success via browser"},
        }

        with patch(
            "job_mcp.core.application.strategies.api.ApiPostStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_api_apply, patch(
            "job_mcp.core.application.strategies.browser.BrowserPlaywrightStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_browser_apply:
            mock_api_apply.return_value = api_mock_return
            mock_browser_apply.return_value = browser_mock_return

            result = await dispatcher.execute_application(api_job, sample_profile)

            mock_api_apply.assert_called_once()
            mock_browser_apply.assert_called_once_with(api_job, sample_profile, cv_path=None)

            assert result["success"] is True
            assert result["method"] == ApplicationMethod.BROWSER.value

            entry = memory_ledger.get_application(api_job.job_id)
            assert entry is not None
            assert entry.status == ApplicationStatus.SUCCESS.value
            assert entry.method == ApplicationMethod.BROWSER.value


@pytest.mark.asyncio
async def test_dispatcher_records_receipt_details_in_ledger(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify that structured receipt_details from strategy is recorded in the ApplicationEntry ledger."""
    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        job = Job(
            job_id="job_receipt_test_1",
            title="Senior Backend Engineer",
            company="ReceiptCorp",
            location="Tel Aviv, Israel",
            match_score=90.0,
            source="greenhouse",
            apply_url="https://boards.greenhouse.io/receiptcorp/1",
        )

        mock_receipt = {
            "confirmed": True,
            "confirmation_type": "DOM_CONFIRMATION",
            "receipt_text": "Confirmation message: 'thank you for applying'",
            "confirmation_url": "https://boards.greenhouse.io/receiptcorp/1",
            "screenshot_path": "data/screenshots/receipt.png",
            "http_receipts": [{"url": "https://boards-api.greenhouse.io/apply", "status": 200}],
        }

        mock_apply_return = {
            "success": True,
            "job_id": job.job_id,
            "method": ApplicationMethod.BROWSER.value,
            "status": "success",
            "receipt_details": mock_receipt,
            "receipt": mock_receipt["receipt_text"],
        }

        with patch(
            "job_mcp.core.application.strategies.browser.BrowserPlaywrightStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_browser_apply:
            mock_browser_apply.return_value = mock_apply_return
            res = await dispatcher.execute_application(job, sample_profile)

        assert res["success"] is True
        entry = memory_ledger.get_application(job.job_id)
        assert entry is not None
        assert entry.receipt_details is not None
        assert entry.receipt_details["confirmation_type"] == "DOM_CONFIRMATION"
        assert entry.receipt_details["screenshot_path"] == "data/screenshots/receipt.png"


@pytest.mark.asyncio
async def test_dispatcher_auto_tailor_invokes_system2(
    memory_ledger: ApplicationLedger,
    sample_profile: CandidateProfile,
):
    """Verify that auto_tailor=True invokes System 2 application package generation for high match scores."""
    from job_mcp.core.application.tailoring import ApplicationPackage, InterviewQuestionPrep

    with patch.dict(os.environ, {"AUTO_APPLY_ENABLED": "true"}):
        dispatcher = HybridApplicationDispatcher(ledger=memory_ledger)

        job = Job(
            job_id="job_tailor_test_1",
            title="Senior AI Engineer",
            company="AILabs",
            location="Tel Aviv, Israel",
            match_score=88.0,
            source="comeet",
            apply_url="https://comeet.com/jobs/ailabs/1",
        )

        mock_pkg = ApplicationPackage(
            job_id=job.job_id,
            job_title=job.title,
            company=job.company,
            tailored_cv_highlights=[
                "Proven expertise in Python & Agentic systems",
                "Hands-on production delivery with fast retrieval architectures",
            ],
            custom_cover_letter="Dear AILabs team, I am eager to contribute...",
            recruiter_pitch="Experienced engineer with strong background in LLMs.",
            interview_prep_questions=[
                InterviewQuestionPrep(
                    question="How do you handle multi-agent orchestration?",
                    topic="Agent Architecture",
                    recommended_strategy="Highlight hands-on Antigravity CLI work",
                ),
                InterviewQuestionPrep(
                    question="How do you evaluate system 1 vs system 2 models?",
                    topic="Model Evaluation",
                    recommended_strategy="Discuss A/B benchmarking and precision capping",
                ),
            ],
        )

        with patch(
            "job_mcp.core.application.tailoring.generate_application_package",
            new_callable=AsyncMock,
        ) as mock_tailor, patch(
            "job_mcp.core.application.strategies.browser.BrowserPlaywrightStrategy.apply",
            new_callable=AsyncMock,
        ) as mock_browser_apply:
            mock_tailor.return_value = mock_pkg
            mock_browser_apply.return_value = {
                "success": True,
                "job_id": job.job_id,
                "method": ApplicationMethod.BROWSER.value,
                "status": "success",
            }

            res = await dispatcher.execute_application(job, sample_profile, auto_tailor=True)

        assert res["success"] is True
        mock_tailor.assert_called_once()
        # Ensure strategy received profile with customized cover letter
        called_args, called_kwargs = mock_browser_apply.call_args
        called_profile = called_args[1]
        assert called_profile.cover_letter == "Dear AILabs team, I am eager to contribute..."
