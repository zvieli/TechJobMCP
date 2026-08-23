"""Unit and integration tests for BrowserPlaywrightStrategy dynamic DOM interaction and ATS submission."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from playwright.async_api import async_playwright

from job_mcp.core.application.mapper import SemanticFormMapper
from job_mcp.core.application.strategies.browser import BrowserPlaywrightStrategy
from job_mcp.core.application.strategy import get_application_strategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job, WorkMode


class MockSessionManager:
    """Mock session manager providing an active Playwright page."""

    def __init__(self, page):
        self.page = page

    async def get_page(self):
        return self.page


@pytest.fixture
def sample_profile() -> CandidateProfile:
    """CandidateProfile fixture."""
    return CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        top_skills=["Python", "FastAPI"],
        primary_stack=["Python", "PostgreSQL"],
        seniority_level="Senior",
        target_roles=["Senior Backend Engineer", "Lead Developer"],
    )


@pytest.fixture
def sample_profile_dict() -> dict:
    """Detailed profile dictionary."""
    return {
        "first_name": "Lior",
        "last_name": "Zvieli",
        "full_name": "Lior Zvieli",
        "email": "lior@example.com",
        "phone": "+972-50-1234567",
        "linkedin": "https://www.linkedin.com/in/liorzvieli",
        "location": "Tel Aviv, Israel",
        "seniority_level": "Senior",
        "skills": ["Python", "FastAPI"],
    }


@pytest.fixture
def dummy_cv_file(tmp_path: Path) -> Path:
    """Create a temporary dummy CV file."""
    cv_file = tmp_path / "lior_resume.pdf"
    cv_file.write_text("%PDF-1.4 mock cv binary content")
    return cv_file


@pytest.fixture
def sample_job() -> Job:
    """Sample ATS Job listing without remote apply_url for local DOM tests."""
    return Job(
        job_id="job-ats-501",
        title="Senior Python Backend Engineer",
        company="CyberTech Labs",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="comeet",
        match_score=95.0,
    )


# ---------------------------------------------------------------------------
# 1. Dynamic DOM Preview Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_preview_dynamic_dom(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test dynamic DOM preview extracting form schema from active page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <form id="application-form">
            <label for="applicant_name">Full Name *</label>
            <input type="text" id="applicant_name" name="name" required />

            <label for="applicant_email">Email Address *</label>
            <input type="email" id="applicant_email" name="email" required />

            <label for="applicant_phone">Phone Number</label>
            <input type="tel" id="applicant_phone" name="phone" />

            <label for="cv_file">Upload CV / Resume *</label>
            <input type="file" id="cv_file" name="resume" required />

            <label for="exp_years">Years of Experience</label>
            <select id="exp_years" name="experience">
                <option value="junior">0-2 Years</option>
                <option value="mid">3-5 Years</option>
                <option value="senior">6+ Years</option>
            </select>

            <button type="submit" id="submit_btn">Submit Application</button>
        </form>
    </body>
    </html>
    """

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content)

        session_mgr = MockSessionManager(page)
        strategy = BrowserPlaywrightStrategy(session_manager=session_mgr)

        preview = await strategy.preview(sample_job, sample_profile, cv_path=str(dummy_cv_file))

        assert isinstance(preview, ApplicationPreview)
        assert preview.job_id == sample_job.job_id
        assert preview.application_method == ApplicationMethod.BROWSER.value
        assert "applicant_name" in preview.fields_to_submit
        assert preview.fields_to_submit["applicant_name"]["type"] == "text"
        assert preview.fields_to_submit["applicant_name"]["value"] != ""

        assert "applicant_email" in preview.fields_to_submit
        assert "@" in preview.fields_to_submit["applicant_email"]["value"]

        assert "cv_file" in preview.fields_to_submit
        assert preview.fields_to_submit["cv_file"]["type"] == "file"
        assert preview.fields_to_submit["resume_file_path"] == str(dummy_cv_file.resolve())

        assert any("Dynamic DOM form schema extracted" in w for w in preview.warnings)
        assert any("Detected submit button" in w for w in preview.warnings)

        await browser.close()


# ---------------------------------------------------------------------------
# 2. Dynamic DOM Apply Tests (Full form interaction)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_apply_dynamic_dom(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test dynamic DOM apply filling all form fields and clicking submit button."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <form id="ats-form" onsubmit="event.preventDefault(); window.formSubmitted = true;">
            <label for="full_name">Candidate Name</label>
            <input type="text" id="full_name" name="name" />

            <label for="email_in">Candidate Email</label>
            <input type="email" id="email_in" name="email" />

            <label for="phone_in">Phone</label>
            <input type="tel" id="phone_in" name="phone" />

            <label for="resume_in">Resume File</label>
            <input type="file" id="resume_in" name="resume" />

            <label for="experience_sel">Experience Level</label>
            <select id="experience_sel" name="experience">
                <option value="1-3">1-3 Years</option>
                <option value="4-6">4-6 Years</option>
                <option value="7+">7+ Years</option>
            </select>

            <label for="notes_ta">Why do you want this role?</label>
            <textarea id="notes_ta" name="cover_letter"></textarea>

            <label>
                <input type="checkbox" id="work_auth_cb" name="work_auth" />
                Authorized to work in Israel
            </label>

            <button type="submit" id="submit_application_btn" class="btn-primary">
                Send Application
            </button>
        </form>
    </body>
    </html>
    """

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content)

        session_mgr = MockSessionManager(page)
        strategy = BrowserPlaywrightStrategy(session_manager=session_mgr)

        result = await strategy.apply(sample_job, sample_profile, cv_path=str(dummy_cv_file))

        assert result["success"] is True
        assert result["status"] == "success"
        assert result["method"] == ApplicationMethod.BROWSER.value
        assert "submission_id" in result
        assert result["submit_clicked"] is True

        # Verify fields were filled in DOM
        name_val = await page.locator("#full_name").input_value()
        assert len(name_val) > 0

        email_val = await page.locator("#email_in").input_value()
        assert "@" in email_val

        phone_val = await page.locator("#phone_in").input_value()
        assert len(phone_val) > 0

        is_checked = await page.locator("#work_auth_cb").is_checked()
        assert is_checked is True

        # Verify form submit was triggered
        submitted = await page.evaluate("() => window.formSubmitted === true")
        assert submitted is True

        await browser.close()


# ---------------------------------------------------------------------------
# 3. Comeet / Embedded IFrame Application Flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_comeet_iframe_flow(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test dynamic form extraction and submission when ATS form is inside an embedded iframe."""
    main_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Company Careers Portal</title></head>
    <body>
        <div class="header"><h1>Join Our Engineering Team</h1></div>
        <iframe id="comeet-widget" srcdoc='
            <html>
            <body>
                <form id="comeet-apply-form" onsubmit="event.preventDefault(); window.comeetSubmitted = true;">
                    <label for="c_name">Full Name</label>
                    <input type="text" id="c_name" name="name" required />

                    <label for="c_email">Email</label>
                    <input type="email" id="c_email" name="email" required />

                    <label for="c_phone">Phone</label>
                    <input type="tel" id="c_phone" name="phone" />

                    <label for="c_cv">Attach CV</label>
                    <input type="file" id="c_cv" name="attachment" required />

                    <button type="submit" id="c_submit">Submit Application</button>
                </form>
            </body>
            </html>
        '></iframe>
    </body>
    </html>
    """

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(main_html)
        await page.wait_for_timeout(300)

        session_mgr = MockSessionManager(page)
        strategy = BrowserPlaywrightStrategy(session_manager=session_mgr)

        result = await strategy.apply(sample_job, sample_profile, cv_path=str(dummy_cv_file))

        assert result["success"] is True
        assert result["status"] == "success"
        assert len(result["fields_filled"]) >= 3
        assert result["submit_clicked"] is True

        # Check that iframe frame received the form submit
        iframe = page.frames[1]
        submitted = await iframe.evaluate("() => window.comeetSubmitted === true")
        assert submitted is True

        await browser.close()


# ---------------------------------------------------------------------------
# 4. Workday Modal / Apply Trigger Flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_workday_modal_trigger(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test clicking Apply trigger button when form is initially hidden in a modal."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <div id="job-header">
            <h2>Principal Architect</h2>
            <button type="button" id="apply_btn" class="apply" onclick="document.getElementById('modal').style.display='block'">Apply Now</button>
        </div>

        <div id="modal" style="display:none;">
            <form id="modal-form" onsubmit="event.preventDefault(); window.modalSubmitted = true;">
                <label for="cand_name">Applicant Name</label>
                <input type="text" id="cand_name" name="applicant_name" />

                <label for="cand_email">Applicant Email</label>
                <input type="email" id="cand_email" name="applicant_email" />

                <button type="submit" id="modal_submit" class="btn primary">Submit Application</button>
            </form>
        </div>
    </body>
    </html>
    """

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content)

        session_mgr = MockSessionManager(page)
        strategy = BrowserPlaywrightStrategy(session_manager=session_mgr)

        job_workday = Job(
            job_id="wd-808",
            title="Principal Architect",
            company="Enterprise Cloud",
            source="workday",
        )

        result = await strategy.apply(job_workday, sample_profile, cv_path=str(dummy_cv_file))

        assert result["success"] is True
        assert result["status"] == "success"
        assert result["submit_clicked"] is True

        submitted = await page.evaluate("() => window.modalSubmitted === true")
        assert submitted is True

        await browser.close()


# ---------------------------------------------------------------------------
# 5. Fallback Heuristic Execution (No Browser Page)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_fallback_without_page(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test simulated fallback execution when session manager is None or has no page."""
    strategy = BrowserPlaywrightStrategy(session_manager=None)

    preview = await strategy.preview(sample_job, sample_profile, cv_path=str(dummy_cv_file))
    assert preview.application_method == ApplicationMethod.BROWSER.value
    assert "applicant_name" in preview.fields_to_submit
    assert preview.fields_to_submit["resume_file_path"] == str(dummy_cv_file.resolve())

    result = await strategy.apply(sample_job, sample_profile, cv_path=str(dummy_cv_file))
    assert result["success"] is True
    assert result["method"] == ApplicationMethod.BROWSER.value
    assert result["status"] == "success"
    assert "Playwright Browser Automation" in result["response"]["portal"]


# ---------------------------------------------------------------------------
# 6. Error Handling during Page Interaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_strategy_apply_error_handling(sample_profile: CandidateProfile):
    """Test error handling when page evaluation throws an unhandled error."""
    mock_page = MagicMock()
    mock_page.goto = AsyncMock(side_effect=Exception("Connection refused by ATS endpoint"))
    mock_page.url = "about:blank"

    job_err = Job(
        job_id="job-ats-err",
        title="Senior Python Backend Engineer",
        company="CyberTech Labs",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="comeet",
        apply_url="https://careers.cybertech.com/jobs/501",
        match_score=95.0,
    )

    session_mgr = MockSessionManager(mock_page)
    strategy = BrowserPlaywrightStrategy(session_manager=session_mgr)

    result = await strategy.apply(job_err, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# 7. ATS Routing Verification
# ---------------------------------------------------------------------------

def test_ats_routing_to_browser_strategy():
    """Verify ATS sources route to BrowserPlaywrightStrategy."""
    sources = ["comeet", "workday", "eightfold", "greenhouse", "lever", "alljobs", "browser", "custom_unknown_ats"]
    for src in sources:
        strat = get_application_strategy(src)
        assert isinstance(strat, BrowserPlaywrightStrategy), f"Source '{src}' did not route to BrowserPlaywrightStrategy"
