"""Unit tests for ApplicationStrategy implementations and strategy routing factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from job_mcp.core.application.strategies import (
    ApiPostStrategy,
    BrowserPlaywrightStrategy,
    EasyApplyStrategy,
)
from job_mcp.core.application.strategy import (
    ApplicationStrategy,
    get_application_strategy,
    register_application_strategy,
)
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job, WorkMode


@pytest.fixture
def sample_job() -> Job:
    """Fixture returning a sample Job listing."""
    return Job(
        job_id="job-101",
        title="Senior Python Backend Engineer",
        company="TechCorp Israel",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="comeet",
        apply_url="https://api.comeet.me/v1/positions/job-101/apply",
        match_score=92.0,
    )


@pytest.fixture
def sample_profile() -> CandidateProfile:
    """Fixture returning a sample CandidateProfile."""
    return CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        top_skills=["Python", "FastAPI"],
        primary_stack=["Python", "PostgreSQL"],
        seniority_level="Senior",
        target_roles=["Senior Python Engineer", "Backend Tech Lead"],
    )


@pytest.fixture
def dummy_cv_file(tmp_path: Path) -> Path:
    """Create a temporary CV document file."""
    cv_file = tmp_path / "resume_test.pdf"
    cv_file.write_text("%PDF-1.4 sample resume content for testing")
    return cv_file


# ---------------------------------------------------------
# Strategy Routing & Registry Tests
# ---------------------------------------------------------

def test_get_application_strategy_routing():
    """Verify strategy routing maps ATS sources to appropriate Strategy instances."""
    # Direct API Post sources (strictly api, api_direct, api_post, direct_tech)
    assert isinstance(get_application_strategy("api_direct"), ApiPostStrategy)
    assert isinstance(get_application_strategy("api_post"), ApiPostStrategy)
    assert isinstance(get_application_strategy("direct_tech"), ApiPostStrategy)
    assert isinstance(get_application_strategy("api"), ApiPostStrategy)


    # Easy Apply sources
    assert isinstance(get_application_strategy("linkedin"), EasyApplyStrategy)
    assert isinstance(get_application_strategy("easy_apply"), EasyApplyStrategy)
    assert isinstance(get_application_strategy("quick_apply"), EasyApplyStrategy)

    # Dynamic ATS Browser Playwright sources & web platforms (including hiremetech, jobify)
    assert isinstance(get_application_strategy("hiremetech"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("jobify"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("comeet"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("comeet_12345"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("greenhouse"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("lever"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("workday"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("alljobs"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("eightfold"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("browser"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("playwright"), BrowserPlaywrightStrategy)
    assert isinstance(get_application_strategy("custom_unknown_ats"), BrowserPlaywrightStrategy)



def test_custom_strategy_registration():
    """Verify register_application_strategy allows dynamically plugging custom strategies."""
    class CustomPortalStrategy(ApplicationStrategy):
        method = "custom_portal"

        async def preview(self, job, profile, cv_path=None):
            return ApplicationPreview(
                job_id=job.job_id,
                job_title=job.title,
                company=job.company,
                application_method="custom_portal",
            )

        async def apply(self, job, profile, cv_path=None):
            return {"success": True, "method": "custom_portal"}

    register_application_strategy("custom_portal", CustomPortalStrategy)
    strategy = get_application_strategy("custom_portal_source")
    assert isinstance(strategy, CustomPortalStrategy)
    assert strategy.method == "custom_portal"


# ---------------------------------------------------------
# ApiPostStrategy Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_api_post_strategy_preview(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test ApiPostStrategy preview with valid CV and endpoint."""
    strategy = ApiPostStrategy()
    preview = await strategy.preview(sample_job, sample_profile, cv_path=str(dummy_cv_file))

    assert isinstance(preview, ApplicationPreview)
    assert preview.job_id == sample_job.job_id
    assert preview.company == sample_job.company
    assert preview.application_method == ApplicationMethod.API.value
    assert preview.fields_to_submit["endpoint_url"] == sample_job.apply_url
    assert preview.fields_to_submit["cv_filename"] == dummy_cv_file.name
    assert preview.fields_to_submit["seniority_level"] == "Senior"
    assert "Python" in preview.fields_to_submit["skills"]


@pytest.mark.asyncio
async def test_api_post_strategy_preview_warnings(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy preview generates warnings when CV or endpoint is missing."""
    job_no_url = Job(
        job_id="job-nourl",
        title="Software Engineer",
        company="Startup Corp",
        source="comeet",
    )
    strategy = ApiPostStrategy()
    preview = await strategy.preview(job_no_url, sample_profile, cv_path="/non/existent/cv.pdf")

    assert any("No explicit apply_url" in w for w in preview.warnings)
    assert any("does not exist on disk" in w for w in preview.warnings)


@pytest.mark.asyncio
async def test_api_post_strategy_apply_simulated(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test ApiPostStrategy simulated direct ATS API submission."""
    # When apply_url is empty, it uses the simulated ATS API gateway
    job_direct = Job(
        job_id="comeet_999",
        title="Backend Engineer",
        company="CyberTech",
        source="comeet",
    )
    strategy = ApiPostStrategy()
    result = await strategy.apply(job_direct, sample_profile, cv_path=str(dummy_cv_file))

    assert result["success"] is True
    assert result["job_id"] == "comeet_999"
    assert result["method"] == ApplicationMethod.API.value
    assert result["status"] == "success"
    assert "submission_id" in result
    assert result["response"]["source"] == "comeet"


@pytest.mark.asyncio
async def test_api_post_strategy_apply_http_success(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test ApiPostStrategy HTTP POST submission with mock httpx client (200 OK)."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "received", "application_id": "app-xyz"}
    mock_client.post.return_value = mock_resp

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile, cv_path=str(dummy_cv_file))

    assert result["success"] is True
    assert result["status"] == "success"
    assert result["status_code"] == 200
    assert result["response"]["application_id"] == "app-xyz"
    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_post_strategy_apply_http_rate_limit(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy HTTP 429 rate limit response."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"
    mock_client.post.return_value = mock_resp

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error_code"] == "RATE_LIMITED"
    assert "HTTP 429" in result["error"]


@pytest.mark.asyncio
async def test_api_post_strategy_apply_network_error(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy network exception handling."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Connection failed")

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error_code"] == "NETWORK_ERROR"
    assert "Connection failed" in result["error"]


@pytest.mark.asyncio
async def test_api_post_strategy_apply_http_405_method_not_allowed(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy handles HTTP 405 by reporting endpoint is not an API."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 405
    mock_resp.text = "Method Not Allowed"
    mock_resp.headers = {"content-type": "text/html"}
    mock_client.post.return_value = mock_resp

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error_code"] in ("ENDPOINT_NOT_AN_API", "HTTP_405")
    assert "not an API endpoint" in result["error"].lower() or "method not allowed" in result["error"].lower()


@pytest.mark.asyncio
async def test_api_post_strategy_apply_http_301_redirect(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy handles HTTP 301 by reporting endpoint is not an API."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 301
    mock_resp.text = "Moved Permanently"
    mock_resp.headers = {"content-type": "text/html"}
    mock_client.post.return_value = mock_resp

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error_code"] == "ENDPOINT_NOT_AN_API"
    assert "not an API endpoint" in result["error"].lower() or "redirect" in result["error"].lower()


@pytest.mark.asyncio
async def test_api_post_strategy_apply_html_content_type(sample_job: Job, sample_profile: CandidateProfile):
    """Test ApiPostStrategy rejects HTTP 200 responses that are actually HTML web pages."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = "<!DOCTYPE html><html><body>Job Listing Page</body></html>"
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_client.post.return_value = mock_resp

    strategy = ApiPostStrategy(client=mock_client)
    result = await strategy.apply(sample_job, sample_profile)

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error_code"] == "ENDPOINT_NOT_AN_API"
    assert "web page" in result["error"].lower() or "not an api" in result["error"].lower()


# ---------------------------------------------------------
# EasyApplyStrategy Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_easy_apply_strategy_preview(sample_job: Job, sample_profile: CandidateProfile, dummy_cv_file: Path):
    """Test EasyApplyStrategy preview."""
    job_li = Job(
        job_id="li_888",
        title="Full Stack Developer",
        company="Fintech IL",
        source="linkedin",
    )
    strategy = EasyApplyStrategy()
    preview = await strategy.preview(job_li, sample_profile, cv_path=str(dummy_cv_file))

    assert preview.application_method == ApplicationMethod.EASY_APPLY.value
    assert preview.fields_to_submit["apply_mode"] == "1_click_easy_apply"
    assert preview.fields_to_submit["cv_attached"] is True


@pytest.mark.asyncio
async def test_easy_apply_strategy_apply():
    """Test EasyApplyStrategy application execution."""
    job_li = Job(
        job_id="li_888",
        title="Full Stack Developer",
        company="Fintech IL",
        source="linkedin",
    )
    profile = CandidateProfile(skills=["TypeScript", "React"])
    strategy = EasyApplyStrategy()
    result = await strategy.apply(job_li, profile)

    assert result["success"] is True
    assert result["method"] == ApplicationMethod.EASY_APPLY.value
    assert result["status"] == "success"
    assert "submission_id" in result
    assert result["response"]["easy_apply_status"] == "submitted"


@pytest.mark.asyncio
async def test_easy_apply_strategy_apply_navigates_to_job_url():
    """Verify EasyApplyStrategy navigates to target URL when page is not on target URL."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://www.linkedin.com/feed/"
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.first = mock_locator
    mock_page.locator = MagicMock(return_value=mock_locator)

    job = Job(
        job_id="li-999",
        title="Backend Developer",
        company="InnoTech",
        source="linkedin",
        apply_url="https://www.linkedin.com/jobs/view/123456789/",
    )
    profile = CandidateProfile(skills=["Python"])

    strategy = EasyApplyStrategy(session_manager=mock_session_manager)
    result = await strategy.apply(job, profile)

    assert result["success"] is True
    mock_page.goto.assert_awaited_once_with(
        "https://www.linkedin.com/jobs/view/123456789/",
        wait_until="domcontentloaded",
    )


@pytest.mark.asyncio
async def test_easy_apply_strategy_apply_skips_navigation_if_already_on_url():
    """Verify EasyApplyStrategy does not call page.goto if page is already on target URL."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://www.linkedin.com/jobs/view/123456789/"
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.first = mock_locator
    mock_page.locator = MagicMock(return_value=mock_locator)

    job = Job(
        job_id="li-999",
        title="Backend Developer",
        company="InnoTech",
        source="linkedin",
        apply_url="https://www.linkedin.com/jobs/view/123456789/",
    )
    profile = CandidateProfile(skills=["Python"])

    strategy = EasyApplyStrategy(session_manager=mock_session_manager)
    result = await strategy.apply(job, profile)

    assert result["success"] is True
    mock_page.goto.assert_not_awaited()


@pytest.mark.asyncio
async def test_easy_apply_strategy_hebrew_buttons():
    """Verify EasyApplyStrategy queries and clicks Hebrew apply buttons."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://www.jobify.co.il/jobs/456"
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_apply_btn = AsyncMock()
    mock_apply_btn.count = AsyncMock(return_value=1)
    mock_apply_btn.is_visible = AsyncMock(return_value=True)

    mock_submit_btn = AsyncMock()
    mock_submit_btn.count = AsyncMock(return_value=1)
    mock_submit_btn.is_visible = AsyncMock(return_value=True)

    def locator_side_effect(selector):
        loc = MagicMock()
        # Distinguish between Easy Apply button and modal submit button
        if "הגשה מהירה" in selector or "Easy Apply" in selector:
            loc.first = mock_apply_btn
        elif "שלח" in selector or "Submit application" in selector:
            loc.first = mock_submit_btn
        else:
            loc.first = mock_apply_btn
        return loc

    mock_page.locator = MagicMock(side_effect=locator_side_effect)


    job = Job(
        job_id="jobify-456",
        title="Fullstack Developer",
        company="StartupIL",
        source="jobify",
        url="https://www.jobify.co.il/jobs/456",
    )
    profile = CandidateProfile(skills=["React", "Node"])

    strategy = EasyApplyStrategy(session_manager=mock_session_manager)
    result = await strategy.apply(job, profile)

    assert result["success"] is True
    mock_apply_btn.click.assert_awaited_once()
    mock_submit_btn.click.assert_awaited_once()



# ---------------------------------------------------------
# BrowserPlaywrightStrategy Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_playwright_strategy_preview(sample_job: Job, sample_profile: CandidateProfile):
    """Test BrowserPlaywrightStrategy preview without active browser page."""
    job_wd = Job(
        job_id="workday_111",
        title="DevOps Engineer",
        company="CloudScale",
        source="workday",
    )
    strategy = BrowserPlaywrightStrategy()
    preview = await strategy.preview(job_wd, sample_profile)

    assert preview.application_method == ApplicationMethod.BROWSER.value
    assert "applicant_name" in preview.fields_to_submit
    assert "resume_upload" in preview.fields_to_submit
    assert any("Browser automation strategy active" in w for w in preview.warnings)


@pytest.mark.asyncio
async def test_browser_playwright_strategy_apply():
    """Test BrowserPlaywrightStrategy execution."""
    job_wd = Job(
        job_id="workday_111",
        title="DevOps Engineer",
        company="CloudScale",
        source="workday",
    )
    profile = CandidateProfile(skills=["Kubernetes", "AWS"])
    strategy = BrowserPlaywrightStrategy()
    result = await strategy.apply(job_wd, profile)

    assert result["success"] is True
    assert result["method"] == ApplicationMethod.BROWSER.value
    assert result["status"] == "success"
    assert "submission_id" in result
    assert "Playwright Browser Automation" in result["response"]["portal"]


@pytest.mark.asyncio
async def test_browser_playwright_strategy_sso_wall_detected():
    """Verify BrowserPlaywrightStrategy detects Google/Apple SSO login wall and blocks with SSO_LOGIN_REQUIRED."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://accounts.google.com/v3/signin/identifier"
    mock_page.title = AsyncMock(return_value="Sign in - Google Accounts")
    mock_page.screenshot = AsyncMock()
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    job = Job(
        job_id="direct_google_123",
        title="Software Engineer, Search",
        company="Google",
        source="direct_tech",
        apply_url="https://www.google.com/about/careers/applications/signin?jobId=123",
    )
    profile = CandidateProfile(skills=["Python", "Algorithms"])

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)
    result = await strategy.apply(job, profile)

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error_code"] == "SSO_LOGIN_REQUIRED"
    assert "Google" in result["error"]
    assert result["screenshot_path"] is not None


@pytest.mark.asyncio
async def test_browser_playwright_strategy_apple_sso_wall_detected():
    """Verify BrowserPlaywrightStrategy detects Apple ID authentication wall."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://idmsa.apple.com/IDMSWebAuth/signin"
    mock_page.title = AsyncMock(return_value="Sign In - Apple")
    mock_page.screenshot = AsyncMock()
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    job = Job(
        job_id="direct_apple_456",
        title="Full Stack Developer : Agentic AI",
        company="Apple",
        source="direct_tech",
        apply_url="https://jobs.apple.com/en-il/details/200674773-1451/full-stack-developer-agentic-ai",
    )
    profile = CandidateProfile(skills=["Agentic AI", "Python"])

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)
    result = await strategy.apply(job, profile)

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error_code"] == "SSO_LOGIN_REQUIRED"
    assert "Apple ID" in result["error"]


@pytest.mark.asyncio
async def test_browser_playwright_strategy_no_submit_button_returns_incomplete():
    """Verify BrowserPlaywrightStrategy returns incomplete if form fields exist but submit button not found."""
    from job_mcp.core.application.dom_inspector import FormFieldSchema

    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://jobs.example.com/apply"
    mock_page.title = AsyncMock(return_value="Careers Application")
    mock_page.screenshot = AsyncMock()
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_loc = MagicMock()
    mock_loc.count = AsyncMock(return_value=1)
    mock_loc.fill = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_loc)

    job = Job(
        job_id="custom_portal_789",
        title="DevOps Engineer",
        company="StartupTech",
        source="browser",
        apply_url="https://jobs.example.com/apply",
    )
    profile = CandidateProfile(skills=["Kubernetes", "AWS"])

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=[
            FormFieldSchema(field_id="email", name="email", field_type="email", selector="input[name='email']")
        ])),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=None)),
    ):
        result = await strategy.apply(job, profile)

    assert result["success"] is False
    assert result["status"] == "incomplete"
    assert result["error_code"] == "NO_SUBMIT_BUTTON"


@pytest.mark.asyncio
async def test_browser_playwright_strategy_success_with_receipt():
    """Verify BrowserPlaywrightStrategy validates submit click and post-submission receipt."""
    from job_mcp.core.application.dom_inspector import FormFieldSchema, SubmitButtonInfo

    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://jobs.example.com/confirmation"
    mock_page.title = AsyncMock(return_value="Application Received")
    mock_page.content = AsyncMock(return_value="<div>Thank you for your application!</div>")
    mock_page.screenshot = AsyncMock()
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_btn_loc = MagicMock()
    mock_btn_loc.count = AsyncMock(return_value=1)
    mock_btn_loc.first = mock_btn_loc
    mock_btn_loc.click = AsyncMock()

    mock_field_loc = MagicMock()
    mock_field_loc.count = AsyncMock(return_value=1)
    mock_field_loc.fill = AsyncMock()

    def locator_mock(selector):
        if "submit" in selector:
            return mock_btn_loc
        return mock_field_loc

    mock_page.locator = MagicMock(side_effect=locator_mock)

    job = Job(
        job_id="custom_portal_999",
        title="Backend Engineer",
        company="StartupTech",
        source="browser",
        apply_url="https://jobs.example.com/apply",
    )
    profile = CandidateProfile(skills=["Python"])

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)

    mock_submit_info = SubmitButtonInfo(selector="button[type='submit']", text="Submit", confidence=0.95)

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=[
            FormFieldSchema(field_id="email", name="email", field_type="email", selector="input[name='email']")
        ])),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=mock_submit_info)),
    ):
        result = await strategy.apply(job, profile)

    assert result["success"] is True
    assert result["status"] == "success"
    assert result["submit_clicked"] is True
    assert result["confirmed"] is True
    assert "confirmation" in result["receipt"].lower() or "thank you" in result["receipt"].lower()
    assert result["screenshot_path"] is not None


# ---------------------------------------------------------
# New Source Routing & Flagging Tests (Task 4)
# ---------------------------------------------------------

def test_get_application_strategy_routes_new_sources():
    """Verify that gotfriends, greenhouse, lever, and comeet route to BrowserPlaywrightStrategy."""
    for src in ("gotfriends", "gotfriends_123", "greenhouse", "lever", "comeet", "comeet_ats"):
        strategy = get_application_strategy(src)
        assert isinstance(strategy, BrowserPlaywrightStrategy), f"Expected BrowserPlaywrightStrategy for source '{src}'"


def test_sources_support_auto_apply_flags():
    """Verify supports_auto_apply is True for GotFriends, Greenhouse, Lever, and Comeet sources."""
    from job_mcp.sources.public.comeet import ComeetSource
    from job_mcp.sources.public.gotfriends import GotFriendsSource
    from job_mcp.sources.public.greenhouse import GreenhouseSource
    from job_mcp.sources.public.lever import LeverSource

    assert GotFriendsSource().supports_auto_apply is True
    assert GreenhouseSource().supports_auto_apply is True
    assert LeverSource().supports_auto_apply is True
    assert ComeetSource().supports_auto_apply is True


@pytest.mark.asyncio
async def test_browser_strategy_preview_passes_job_to_mapper():
    """Verify BrowserPlaywrightStrategy.preview passes job=job to form_mapper.map_form_fields."""
    from job_mcp.core.application.dom_inspector import FormFieldSchema

    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://jobs.example.com/apply"
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_mapper = MagicMock()
    mock_mapper.map_form_fields = AsyncMock(return_value={"email": "candidate@example.com"})
    mock_mapper.llm_gateway = None

    job = Job(
        job_id="test-job-preview",
        title="Senior AI Engineer",
        company="AI Labs",
        source="gotfriends",
        apply_url="https://jobs.example.com/apply",
    )
    profile = CandidateProfile(skills=["Python", "PyTorch"])

    strategy = BrowserPlaywrightStrategy(
        session_manager=mock_session_manager,
        form_mapper=mock_mapper,
    )

    schema = [
        FormFieldSchema(field_id="email", name="email", field_type="email", selector="input[name='email']")
    ]

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=schema)),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=None)),
    ):
        await strategy.preview(job, profile)

    mock_mapper.map_form_fields.assert_awaited_once()
    _, kwargs = mock_mapper.map_form_fields.call_args
    assert kwargs.get("job") == job


@pytest.mark.asyncio
async def test_browser_strategy_apply_passes_job_to_mapper():
    """Verify BrowserPlaywrightStrategy.apply passes job=job to form_mapper.map_form_fields."""
    from job_mcp.core.application.dom_inspector import FormFieldSchema, SubmitButtonInfo

    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://jobs.example.com/apply"
    mock_page.title = AsyncMock(return_value="Careers Application")
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_loc = MagicMock()
    mock_loc.count = AsyncMock(return_value=1)
    mock_loc.first = mock_loc
    mock_loc.fill = AsyncMock()
    mock_loc.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_loc)

    mock_mapper = MagicMock()
    mock_mapper.map_form_fields = AsyncMock(return_value={"email": "candidate@example.com"})
    mock_mapper.llm_gateway = None

    job = Job(
        job_id="test-job-apply",
        title="Staff ML Engineer",
        company="TechCorp",
        source="lever",
        apply_url="https://jobs.example.com/apply",
    )
    profile = CandidateProfile(skills=["Python", "Deep Learning"])

    strategy = BrowserPlaywrightStrategy(
        session_manager=mock_session_manager,
        form_mapper=mock_mapper,
    )

    schema = [
        FormFieldSchema(field_id="email", name="email", field_type="email", selector="input[name='email']")
    ]
    mock_submit_info = SubmitButtonInfo(selector="button[type='submit']", text="Submit", confidence=0.95)

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=schema)),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=mock_submit_info)),
    ):
        await strategy.apply(job, profile)

    mock_mapper.map_form_fields.assert_awaited_once()
    _, kwargs = mock_mapper.map_form_fields.call_args
    assert kwargs.get("job") == job


