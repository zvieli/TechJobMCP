"""Unit tests for ApplicationStrategy implementations and strategy routing factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
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
    # API Post sources
    assert isinstance(get_application_strategy("comeet"), ApiPostStrategy)
    assert isinstance(get_application_strategy("comeet_12345"), ApiPostStrategy)
    assert isinstance(get_application_strategy("hiremetech"), ApiPostStrategy)
    assert isinstance(get_application_strategy("greenhouse"), ApiPostStrategy)
    assert isinstance(get_application_strategy("lever"), ApiPostStrategy)
    assert isinstance(get_application_strategy("api_direct"), ApiPostStrategy)
    assert isinstance(get_application_strategy("direct_tech"), ApiPostStrategy)

    # Easy Apply sources
    assert isinstance(get_application_strategy("linkedin"), EasyApplyStrategy)
    assert isinstance(get_application_strategy("easy_apply"), EasyApplyStrategy)
    assert isinstance(get_application_strategy("quick_apply"), EasyApplyStrategy)

    # Browser Playwright sources & fallback
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
