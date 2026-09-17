"""Tests for GreenhouseSource job source."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences
from job_mcp.sources.contracts import SourceCategory
from job_mcp.sources.public.greenhouse import (
    GREENHOUSE_COMPANIES,
    GreenhouseCompany,
    GreenhouseSource,
    parse_greenhouse_job,
)


class TestGreenhouseCompany:
    """Tests for GreenhouseCompany dataclass."""

    def test_company_has_required_fields(self):
        company = GreenhouseCompany(name="Test Corp", board_token="testcorp")
        assert company.name == "Test Corp"
        assert company.board_token == "testcorp"
        assert company.enabled is True

    def test_default_companies_not_empty(self):
        assert len(GREENHOUSE_COMPANIES) > 0

    def test_ai21_in_default_companies(self):
        assert "ai21labs" in GREENHOUSE_COMPANIES
        assert GREENHOUSE_COMPANIES["ai21labs"].board_token == "ai21labs"


class TestParseGreenhouseJob:
    """Tests for parse_greenhouse_job parser."""

    def test_parses_minimal_job(self):
        raw = {
            "id": 12345,
            "title": "AI Engineer",
            "location": {"name": "Tel Aviv, Israel"},
            "absolute_url": "https://boards.greenhouse.io/ai21labs/jobs/12345",
            "content": "<p>We are looking for an AI Engineer...</p>",
            "updated_at": "2026-09-15T10:00:00Z",
            "departments": [{"name": "Engineering"}],
        }
        job = parse_greenhouse_job(raw, "AI21 Labs")
        assert job.job_id == "greenhouse_12345"
        assert job.title == "AI Engineer"
        assert job.company == "AI21 Labs"
        assert "Tel Aviv" in job.location
        assert job.source == "greenhouse"
        assert job.url == "https://boards.greenhouse.io/ai21labs/jobs/12345"
        assert job.department == "Engineering"
        assert job.posted_date == "2026-09-15T10:00:00Z"

    def test_extracts_tech_stack(self):
        raw = {
            "id": 99,
            "title": "ML Engineer",
            "location": {"name": "Haifa"},
            "absolute_url": "https://boards.greenhouse.io/test/jobs/99",
            "content": "<p>Requirements: Python, PyTorch, LangChain, RAG experience</p>",
            "departments": [],
        }
        job = parse_greenhouse_job(raw, "TestCo")
        assert any("python" in s.lower() for s in job.tech_stack)

    def test_handles_missing_location(self):
        raw = {
            "id": 100,
            "title": "Developer",
            "location": None,
            "absolute_url": "https://boards.greenhouse.io/test/jobs/100",
            "content": "",
            "departments": [],
        }
        job = parse_greenhouse_job(raw, "TestCo")
        assert job.location == ""
        assert job.posted_date is None

    def test_handles_remote_in_location(self):
        raw = {
            "id": 101,
            "title": "Remote AI Dev",
            "location": {"name": "Remote - Israel"},
            "absolute_url": "https://boards.greenhouse.io/test/jobs/101",
            "content": "",
            "departments": [],
        }
        job = parse_greenhouse_job(raw, "TestCo")
        assert job.work_mode == "remote"

    def test_handles_hybrid_in_location_or_description(self):
        raw = {
            "id": 103,
            "title": "Full Stack Dev",
            "location": {"name": "Tel Aviv (Hybrid)"},
            "absolute_url": "https://boards.greenhouse.io/test/jobs/103",
            "content": "<p>We offer hybrid work flexibility.</p>",
            "departments": [],
        }
        job = parse_greenhouse_job(raw, "TestCo")
        assert job.work_mode == "hybrid"

    def test_handles_string_location(self):
        raw = {
            "id": 102,
            "title": "Backend Dev",
            "location": "Tel Aviv",
            "absolute_url": "https://boards.greenhouse.io/test/jobs/102",
            "content": "",
            "departments": [],
        }
        job = parse_greenhouse_job(raw, "TestCo")
        assert job.location == "Tel Aviv"


class TestGreenhouseSource:
    """Tests for GreenhouseSource fetch and health."""

    def test_source_metadata(self):
        source = GreenhouseSource()
        assert source.source_id == "greenhouse"
        assert source.display_name == "Greenhouse"
        meta = source.get_metadata()
        assert meta.source_id == "greenhouse"
        assert meta.category.value == "public"

    def test_request_headers_user_agent(self):
        from job_mcp.sources.public.greenhouse import REQUEST_HEADERS, REPO_URL
        assert "TechJobMCP/1.0" in REQUEST_HEADERS["User-Agent"]
        assert REPO_URL in REQUEST_HEADERS["User-Agent"]
        assert bytes.fromhex("7a7669656c69").decode() not in REQUEST_HEADERS["User-Agent"].lower()

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        source = GreenhouseSource()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": []}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            result = await source.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_no_companies(self):
        source = GreenhouseSource(companies={})
        result = await source.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        source = GreenhouseSource()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection failed")):
            result = await source.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_jobs_returns_jobs(self):
        source = GreenhouseSource()
        mock_jobs_response = {
            "jobs": [
                {
                    "id": 1,
                    "title": "Junior AI Engineer",
                    "location": {"name": "Tel Aviv"},
                    "absolute_url": "https://boards.greenhouse.io/ai21labs/jobs/1",
                    "content": "<p>Python, LLM, RAG</p>",
                    "departments": [{"name": "AI"}],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_jobs_response
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) >= 1
        assert jobs[0].source == "greenhouse"

    @pytest.mark.asyncio
    async def test_fetch_jobs_respects_limit(self):
        source = GreenhouseSource()
        mock_jobs = [
            {
                "id": i,
                "title": f"Job {i}",
                "location": {"name": "Tel Aviv"},
                "absolute_url": f"https://boards.greenhouse.io/test/jobs/{i}",
                "content": "<p>Python</p>",
                "departments": [],
            }
            for i in range(100)
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": mock_jobs}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            jobs = await source.fetch_jobs(limit=5)
        assert len(jobs) <= 5

    @pytest.mark.asyncio
    async def test_fetch_jobs_no_enabled_companies(self):
        source = GreenhouseSource(companies={"test": GreenhouseCompany("Test", "test", enabled=False)})
        jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_http_error(self):
        source = GreenhouseSource(companies={"test": GreenhouseCompany("Test", "test", enabled=True)})
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        req = httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/test/jobs")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.HTTPStatusError("Not found", request=req, response=mock_resp)):
            jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences_filter(self):
        source = GreenhouseSource(companies={"ai21": GreenhouseCompany("AI21", "ai21labs", enabled=True)})
        mock_jobs_response = {
            "jobs": [
                {
                    "id": 1,
                    "title": "Python AI Engineer",
                    "location": {"name": "Tel Aviv"},
                    "absolute_url": "https://boards.greenhouse.io/ai21labs/jobs/1",
                    "content": "<p>Python, PyTorch</p>",
                    "departments": [],
                },
                {
                    "id": 2,
                    "title": "Accountant",
                    "location": {"name": "Tel Aviv"},
                    "absolute_url": "https://boards.greenhouse.io/ai21labs/jobs/2",
                    "content": "<p>Excel, bookkeeping</p>",
                    "departments": [],
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_jobs_response
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            prefs = JobPreferences(exclude_keywords=["Accountant"])
            jobs = await source.fetch_jobs(preferences=prefs, limit=10)
        assert len(jobs) == 1
        assert jobs[0].title == "Python AI Engineer"

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_unexpected_exception(self):
        source = GreenhouseSource(companies={"test": GreenhouseCompany("Test", "test", enabled=True)})
        with patch.object(source, "_fetch_company_jobs", side_effect=RuntimeError("Unexpected board crash")):
            jobs = await source.fetch_jobs(limit=10)
        assert jobs == []


class TestGreenhouseBackwardCompatAndRegistry:
    """Tests for backward-compat shim and registry integration."""

    def test_backward_compat_shim_import(self):
        from job_mcp.sources.greenhouse import (
            GREENHOUSE_COMPANIES,
            GreenhouseCompany,
            GreenhouseSource,
            parse_greenhouse_job,
        )
        assert GreenhouseSource.source_id == "greenhouse"
        assert GREENHOUSE_COMPANIES is not None
        assert GreenhouseCompany is not None
        assert parse_greenhouse_job is not None

    def test_sources_public_init_exports(self):
        import job_mcp.sources.public as public_pkg
        assert hasattr(public_pkg, "GreenhouseSource")
        assert hasattr(public_pkg, "GreenhouseCompany")
        assert hasattr(public_pkg, "GREENHOUSE_COMPANIES")

    def test_sources_init_exports(self):
        import job_mcp.sources as sources_pkg
        assert hasattr(sources_pkg, "GreenhouseSource")

    def test_registry_integration(self):
        from job_mcp.sources.registry import (
            create_default_registry,
            get_registered_providers,
            reset_builtin_providers,
        )
        reset_builtin_providers()
        providers = get_registered_providers()
        assert "greenhouse" in providers
        provider = providers["greenhouse"]
        assert provider.default_enabled is True
        assert provider.env_var == "ENABLE_GREENHOUSE"
        assert provider.category == SourceCategory.PUBLIC

        # Enabled in default registry
        reg = create_default_registry()
        assert "greenhouse" in reg

        # Can disable explicitly
        reg_disabled = create_default_registry(enable_greenhouse=False)
        assert "greenhouse" not in reg_disabled
