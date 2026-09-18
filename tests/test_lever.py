"""Tests for LeverSource job source."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences
from job_mcp.sources.contracts import SourceCategory
from job_mcp.sources.public.lever import (
    LEVER_COMPANIES,
    LeverCompany,
    LeverSource,
    parse_lever_job,
)


class TestLeverCompany:
    """Tests for LeverCompany dataclass."""

    def test_company_has_required_fields(self):
        company = LeverCompany(name="DriveNets", slug="drivenets")
        assert company.name == "DriveNets"
        assert company.slug == "drivenets"
        assert company.enabled is True

    def test_default_companies_not_empty(self):
        assert len(LEVER_COMPANIES) > 0

    def test_drivenets_in_default_companies(self):
        assert "drivenets" in LEVER_COMPANIES
        assert LEVER_COMPANIES["drivenets"].slug == "drivenets"

    def test_redis_in_default_companies(self):
        assert "redis" in LEVER_COMPANIES
        assert LEVER_COMPANIES["redis"].slug == "redis"


class TestParseLeverJob:
    """Tests for parse_lever_job parser."""

    def test_parses_minimal_job(self):
        raw = {
            "id": "abc-123",
            "text": "Senior Python Engineer",
            "categories": {
                "location": "Tel Aviv",
                "team": "Engineering",
                "commitment": "Full-time",
            },
            "descriptionPlain": "We are looking for a Senior Python Engineer with FastAPI and Kubernetes experience.",
            "hostedUrl": "https://jobs.lever.co/drivenets/abc-123",
            "applyUrl": "https://jobs.lever.co/drivenets/abc-123/apply",
            "createdAt": 1726400000000,
            "workplaceType": "hybrid",
        }
        job = parse_lever_job(raw, "DriveNets")
        assert job.job_id == "lever_abc-123"
        assert job.title == "Senior Python Engineer"
        assert job.company == "DriveNets"
        assert job.location == "Tel Aviv"
        assert job.source == "lever"
        assert job.url == "https://jobs.lever.co/drivenets/abc-123"
        assert job.apply_url == "https://jobs.lever.co/drivenets/abc-123/apply"
        assert job.department == "Engineering"
        assert job.posted_date == "1726400000000"
        assert job.work_mode == "hybrid"

    def test_extracts_tech_stack(self):
        raw = {
            "id": "xyz-789",
            "text": "ML Engineer",
            "categories": {"location": "Tel Aviv"},
            "descriptionPlain": "Requirements: Python, PyTorch, Docker, AWS experience",
            "hostedUrl": "https://jobs.lever.co/test/xyz-789",
        }
        job = parse_lever_job(raw, "TestCo")
        assert any("python" in s.lower() for s in job.tech_stack)

    def test_lever_sections_and_tech_stack_isolation(self):
        raw = {
            "id": "lev-sec-1",
            "text": "Frontend Developer",
            "categories": {"location": "Tel Aviv", "team": "Frontend"},
            "description": "<h3>About Us</h3><p>We are a high-scale data platform using Go, Spark, and Cassandra.</p>",
            "lists": [
                {
                    "text": "The Role",
                    "content": "<ul><li>Develop web interfaces with Vue and JavaScript</li></ul>",
                },
                {
                    "text": "Requirements",
                    "content": "<ul><li>3+ years Vue experience and HTML/CSS</li></ul>",
                },
            ],
            "hostedUrl": "https://jobs.lever.co/test/lev-sec-1",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.company_overview is not None and "high-scale data platform" in job.company_overview
        assert job.responsibilities is not None and "Vue and JavaScript" in job.responsibilities
        assert job.requirements is not None and "Vue experience" in job.requirements
        assert "Vue" in job.tech_stack
        assert "JavaScript" in job.tech_stack
        # Company overview boilerplate keywords MUST NOT bleed into tech stack
        assert "Go" not in job.tech_stack
        assert "Spark" not in job.tech_stack
        assert "Cassandra" not in job.tech_stack

    def test_handles_missing_categories_and_location(self):
        raw = {
            "id": "no-cat-1",
            "text": "Software Engineer",
            "categories": None,
            "descriptionPlain": "Backend engineer role",
            "hostedUrl": "https://jobs.lever.co/test/no-cat-1",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.location == ""
        assert job.department is None
        assert job.posted_date is None

    def test_handles_remote_workplace_type(self):
        raw = {
            "id": "rem-1",
            "text": "Frontend Developer",
            "categories": {"location": "Israel"},
            "workplaceType": "remote",
            "descriptionPlain": "Remote position",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.work_mode == "remote"

    def test_handles_remote_in_location_when_unspecified(self):
        raw = {
            "id": "rem-2",
            "text": "Full Stack Dev",
            "categories": {"location": "Remote - Tel Aviv"},
            "workplaceType": "unspecified",
            "descriptionPlain": "Develop web applications",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.work_mode == "remote"

    def test_handles_hybrid_in_description_when_unspecified(self):
        raw = {
            "id": "hyb-1",
            "text": "DevOps Engineer",
            "categories": {"location": "Tel Aviv"},
            "workplaceType": "unspecified",
            "descriptionPlain": "This is a hybrid role with 2 days from home.",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.work_mode == "hybrid"

    def test_handles_html_fallback_description(self):
        raw = {
            "id": "html-desc",
            "text": "Backend Dev",
            "descriptionPlain": None,
            "description": "<p>Experience with <b>Python</b> and <i>PostgreSQL</i>.</p>",
        }
        job = parse_lever_job(raw, "TestCo")
        assert "<p>" not in job.description
        assert "Experience with Python and PostgreSQL." in job.description

    def test_handles_empty_title_fallback(self):
        raw = {
            "id": "empty-title",
            "text": "",
        }
        job = parse_lever_job(raw, "TestCo")
        assert job.title == "Untitled"


class TestLeverSource:
    """Tests for LeverSource fetch and health."""

    def test_source_metadata(self):
        source = LeverSource()
        assert source.source_id == "lever"
        assert source.display_name == "Lever"
        meta = source.get_metadata()
        assert meta.source_id == "lever"
        assert meta.category.value == "public"
        assert meta.supports_auto_apply is True

    def test_request_headers_user_agent(self):
        from job_mcp.sources.public.lever import REQUEST_HEADERS, REPO_URL
        assert "TechJobMCP/1.0" in REQUEST_HEADERS["User-Agent"]
        assert REPO_URL in REQUEST_HEADERS["User-Agent"]
        assert bytes.fromhex("7a7669656c69").decode() not in REQUEST_HEADERS["User-Agent"].lower()

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        source = LeverSource()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            result = await source.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_no_companies(self):
        source = LeverSource(companies={})
        result = await source.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        source = LeverSource()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection failed")):
            result = await source.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_jobs_returns_jobs(self):
        source = LeverSource()
        mock_jobs_response = [
            {
                "id": "job-1",
                "text": "Junior AI Engineer",
                "categories": {"location": "Tel Aviv", "team": "AI"},
                "hostedUrl": "https://jobs.lever.co/drivenets/job-1",
                "applyUrl": "https://jobs.lever.co/drivenets/job-1/apply",
                "descriptionPlain": "Python, LLM, RAG experience required",
                "createdAt": 1726400000000,
                "workplaceType": "hybrid",
            }
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_jobs_response
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) >= 1
        assert jobs[0].source == "lever"
        assert jobs[0].job_id == "lever_job-1"

    @pytest.mark.asyncio
    async def test_fetch_jobs_respects_limit(self):
        source = LeverSource()
        mock_jobs = [
            {
                "id": f"job-{i}",
                "text": f"Job {i}",
                "categories": {"location": "Tel Aviv"},
                "hostedUrl": f"https://jobs.lever.co/test/job-{i}",
                "descriptionPlain": "Python",
            }
            for i in range(100)
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_jobs
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            jobs = await source.fetch_jobs(limit=5)
        assert len(jobs) <= 5

    @pytest.mark.asyncio
    async def test_fetch_jobs_no_enabled_companies(self):
        source = LeverSource(companies={"test": LeverCompany("Test", "test", enabled=False)})
        jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_http_error(self):
        source = LeverSource(companies={"test": LeverCompany("Test", "test", enabled=True)})
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        req = httpx.Request("GET", "https://api.lever.co/v0/postings/test?mode=json&limit=100")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.HTTPStatusError("Not found", request=req, response=mock_resp)):
            jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_non_list_response(self):
        source = LeverSource(companies={"test": LeverCompany("Test", "test", enabled=True)})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences_filter(self):
        source = LeverSource(companies={"drivenets": LeverCompany("DriveNets", "drivenets", enabled=True)})
        mock_jobs_response = [
            {
                "id": "job-1",
                "text": "Python AI Engineer",
                "categories": {"location": "Tel Aviv"},
                "hostedUrl": "https://jobs.lever.co/drivenets/job-1",
                "descriptionPlain": "Python, PyTorch",
            },
            {
                "id": "job-2",
                "text": "Accountant",
                "categories": {"location": "Tel Aviv"},
                "hostedUrl": "https://jobs.lever.co/drivenets/job-2",
                "descriptionPlain": "Excel, bookkeeping",
            },
        ]
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
        source = LeverSource(companies={"test": LeverCompany("Test", "test", enabled=True)})
        with patch.object(source, "_fetch_company_jobs", side_effect=RuntimeError("Unexpected board crash")):
            jobs = await source.fetch_jobs(limit=10)
        assert jobs == []


class TestLeverBackwardCompatAndRegistry:
    """Tests for backward-compat shim and registry integration."""

    def test_backward_compat_shim_import(self):
        from job_mcp.sources.lever import (
            LEVER_COMPANIES,
            LeverCompany,
            LeverSource,
            parse_lever_job,
        )
        assert LeverSource.source_id == "lever"
        assert LEVER_COMPANIES is not None
        assert LeverCompany is not None
        assert parse_lever_job is not None

    def test_sources_public_init_exports(self):
        import job_mcp.sources.public as public_pkg
        assert hasattr(public_pkg, "LeverSource")
        assert hasattr(public_pkg, "LeverCompany")
        assert hasattr(public_pkg, "LEVER_COMPANIES")
        assert hasattr(public_pkg, "parse_lever_job")

    def test_sources_init_exports(self):
        import job_mcp.sources as sources_pkg
        assert hasattr(sources_pkg, "LeverSource")

    def test_registry_integration(self):
        from job_mcp.sources.registry import (
            create_default_registry,
            get_registered_providers,
            reset_builtin_providers,
        )
        reset_builtin_providers()
        providers = get_registered_providers()
        assert "lever" in providers
        provider = providers["lever"]
        assert provider.default_enabled is True
        assert provider.env_var == "ENABLE_LEVER"
        assert provider.category == SourceCategory.PUBLIC

        # Enabled in default registry
        reg = create_default_registry()
        assert "lever" in reg

        # Can disable explicitly
        reg_disabled = create_default_registry(enable_lever=False)
        assert "lever" not in reg_disabled
