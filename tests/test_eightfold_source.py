"""Unit tests for Eightfold AI ATS job source, position parser, company presets, concurrency, and caching."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources import BaseJobSource, SourceRegistry
from job_mcp.sources.eightfold import (
    DEFAULT_EIGHTFOLD_COMPANIES,
    EIGHTFOLD_COMPANIES,
    EightfoldAISource,
    EightfoldCompany,
    parse_eightfold_position,
)

SAMPLE_EIGHTFOLD_POSTING_FULL = {
    "id": "700123",
    "name": "Senior Deep Learning Infrastructure Engineer",
    "positionUrl": "/careers?pid=700123",
    "locations": ["Yokne'am Illit, Northern District, Israel"],
    "workplace_type": "Hybrid",
    "department": "Engineering & Architecture",
    "posted_date": "2026-08-10",
    "job_description": "<p>We are seeking an engineer experienced in <strong>Python, CUDA, PyTorch, C++, and Docker</strong>.</p>",
    "skills": ["Python", "CUDA", "PyTorch", "C++", "Docker"],
}

SAMPLE_EIGHTFOLD_POSTING_REMOTE = {
    "id": "700456",
    "name": "Staff AI Researcher - Remote",
    "positionUrl": "/careers/job/700456",
    "locations": ["Remote, Israel"],
    "workplace_type": "Remote",
    "department": "AI Research",
    "posted_date": "2026-08-15",
    "job_description": "Working with Transformers, LLMs, FastAPI, and Kubernetes in a distributed environment.",
    "skills": ["Transformers", "FastAPI", "Kubernetes"],
}

SAMPLE_EIGHTFOLD_POSTING_MINIMAL = {
    "id": "700789",
    "name": "Backend Software Developer",
    "positionUrl": "/careers?pid=700789",
}


class TestEightfoldPositionParser:
    """Tests for parse_eightfold_position."""

    def test_parse_full_position(self) -> None:
        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
            locations=["Yokne'am Illit"],
        )
        job = parse_eightfold_position(SAMPLE_EIGHTFOLD_POSTING_FULL, company=company)
        assert job.job_id == "eightfold_nvidia_700123"
        assert job.title == "Senior Deep Learning Infrastructure Engineer"
        assert job.company == "NVIDIA"
        assert job.source == "eightfold"
        assert job.sources == ["eightfold"]
        assert "Yokne'am Illit" in job.location
        assert job.work_mode == WorkMode.HYBRID
        assert job.posted_date == "2026-08-10"
        assert job.url == "https://nvidia.eightfold.ai/careers?pid=700123"
        assert job.apply_url == job.url
        assert "Python" in job.tech_stack
        assert "PyTorch" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "CUDA, PyTorch" in job.description or "Python, CUDA" in job.description
        assert job.department == "Engineering & Architecture"

    def test_parse_remote_position(self) -> None:
        company = EightfoldCompany(
            name="Intel",
            hostname="intel.eightfold.ai",
            domain="intel.com",
            locations=["Israel"],
        )
        job = parse_eightfold_position(SAMPLE_EIGHTFOLD_POSTING_REMOTE, company=company)
        assert job.job_id == "eightfold_intel_700456"
        assert job.title == "Staff AI Researcher - Remote"
        assert job.company == "Intel"
        assert job.work_mode == WorkMode.REMOTE
        assert "Kubernetes" in job.tech_stack
        assert "FastAPI" in job.tech_stack

    def test_parse_minimal_and_fallback_fields(self) -> None:
        company = EightfoldCompany(
            name="Elbit Systems",
            hostname="elbitsystems.eightfold.ai",
            domain="elbitsystems.com",
        )
        job = parse_eightfold_position(SAMPLE_EIGHTFOLD_POSTING_MINIMAL, company=company)
        assert job.job_id == "eightfold_elbit_systems_700789"
        assert job.title == "Backend Software Developer"
        assert job.company == "Elbit Systems"
        assert job.location == ""
        assert job.work_mode is None

    def test_parse_fallback_id_without_id_field(self) -> None:
        company = EightfoldCompany(
            name="Micron",
            hostname="micron.eightfold.ai",
            domain="micron.com",
        )
        raw = {
            "name": "DevOps Engineer",
            "positionUrl": "/careers?pid=888999",
            "job_description": "CI/CD and Linux administration.",
        }
        job = parse_eightfold_position(raw, company=company)
        assert job.job_id == "eightfold_micron_888999"

    def test_parse_absolute_and_relative_urls(self) -> None:
        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
        )
        # Absolute URL
        raw1 = {
            "id": "111",
            "name": "Firmware Engineer",
            "positionUrl": "https://nvidia.eightfold.ai/careers?pid=111",
        }
        job1 = parse_eightfold_position(raw1, company=company)
        assert job1.url == "https://nvidia.eightfold.ai/careers?pid=111"

        # Relative without leading slash
        raw2 = {
            "id": "222",
            "name": "Firmware Engineer",
            "positionUrl": "careers?pid=222",
        }
        job2 = parse_eightfold_position(raw2, company=company)
        assert job2.url == "https://nvidia.eightfold.ai/careers?pid=222"

    def test_parse_skills_and_tech_stack_merging(self) -> None:
        company = EightfoldCompany(
            name="PayPal",
            hostname="paypal.eightfold.ai",
            domain="paypal.com",
        )
        raw = {
            "id": "333",
            "name": "Software Engineer",
            "skills": ["Go", "Kafka", "PostgreSQL"],
            "job_description": "We build payment systems.",
        }
        job = parse_eightfold_position(raw, company=company)
        assert "Go" in job.tech_stack
        assert "Kafka" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack


class TestEightfoldSourceMetadataAndPresets:
    """Tests for metadata, company presets, and registration."""

    def test_source_attributes(self) -> None:
        source = EightfoldAISource()
        assert source.source_id == "eightfold"
        assert "Eightfold" in source.display_name
        assert "Eightfold" in source.description
        assert source.is_authenticated is False
        assert source.supports_bookmarks is False
        assert source.supports_auto_apply is False

        meta = source.get_metadata()
        assert meta.source_id == "eightfold"
        assert meta.is_authenticated is False

    def test_eightfold_companies_presets_coverage(self) -> None:
        assert isinstance(EIGHTFOLD_COMPANIES, dict)
        expected_companies = ["nvidia", "intel", "elbit_systems", "micron"]
        for comp_key in expected_companies:
            assert comp_key in EIGHTFOLD_COMPANIES, f"Missing preset for {comp_key}"
            comp = EIGHTFOLD_COMPANIES[comp_key]
            assert comp.name
            assert comp.hostname
            assert comp.domain
            assert comp.get_search_url().startswith("https://")
            assert "/api/pcsx/search" in comp.get_search_url()

    def test_add_and_remove_company(self) -> None:
        source = EightfoldAISource(companies=[])
        assert len(source.get_companies()) == 0

        company = EightfoldCompany(
            name="Custom Corp",
            hostname="customcorp.eightfold.ai",
            domain="customcorp.com",
        )
        source.add_company(company)
        assert len(source.get_companies()) == 1
        assert source.get_companies()[0].name == "Custom Corp"

        removed = source.remove_company("customcorp.com")
        assert removed is not None
        assert removed.name == "Custom Corp"
        assert len(source.get_companies()) == 0

    def test_registry_integration(self) -> None:
        reg = SourceRegistry()
        source = EightfoldAISource()
        reg.register(source)
        assert "eightfold" in reg
        assert reg.get("eightfold") is source


class TestEightfoldSourceFetchingAndCaching:
    """Tests for EightfoldAISource fetch_jobs, caching, error handling, and concurrency."""

    @pytest.mark.asyncio
    async def test_fetch_jobs_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "positions": [SAMPLE_EIGHTFOLD_POSTING_FULL],
                "total": 1,
            },
        }
        mock_client.get.return_value = mock_response

        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
            locations=["Yokne'am Illit"],
        )
        source = EightfoldAISource(companies=[company], client=mock_client)

        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].job_id == "eightfold_nvidia_700123"
        assert jobs[0].company == "NVIDIA"
        assert jobs[0].source == "eightfold"
        assert jobs[0].sources == ["eightfold"]
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_flat_positions_format(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "positions": [SAMPLE_EIGHTFOLD_POSTING_FULL],
        }
        mock_client.get.return_value = mock_response

        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
        )
        source = EightfoldAISource(companies=[company], client=mock_client)

        jobs = await source.fetch_jobs()
        assert len(jobs) == 1
        assert jobs[0].job_id == "eightfold_nvidia_700123"

    @pytest.mark.asyncio
    async def test_fetch_jobs_keyword_search_params(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"positions": [SAMPLE_EIGHTFOLD_POSTING_FULL]},
        }
        mock_client.get.return_value = mock_response

        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
            locations=["Yokne'am Illit"],
        )
        source = EightfoldAISource(companies=[company], client=mock_client)

        prefs = JobPreferences(keywords=["deep", "learning"], location="Israel")
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1

        called_args, called_kwargs = mock_client.get.call_args
        params = called_kwargs.get("params", {})
        assert params.get("domain") == "nvidia.com"
        assert params.get("query") == "deep learning"
        assert params.get("location") == "Israel"

    @pytest.mark.asyncio
    async def test_ttl_caching(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"positions": [SAMPLE_EIGHTFOLD_POSTING_FULL]},
        }
        mock_client.get.return_value = mock_response

        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
        )
        source = EightfoldAISource(companies=[company], client=mock_client, cache_ttl_seconds=3600)

        # 1st call hits HTTP endpoint
        jobs1 = await source.fetch_jobs()
        assert len(jobs1) == 1
        assert mock_client.get.call_count == 1

        # 2nd call hits cache
        jobs2 = await source.fetch_jobs()
        assert len(jobs2) == 1
        assert mock_client.get.call_count == 1

        # Clear cache and verify fresh fetch
        source.clear_cache()
        jobs3 = await source.fetch_jobs()
        assert len(jobs3) == 1
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrency_semaphore_throttling(self) -> None:
        max_concurrency = 2
        active_requests = 0
        peak_requests = 0

        async def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal active_requests, peak_requests
            active_requests += 1
            if active_requests > peak_requests:
                peak_requests = active_requests
            await asyncio.sleep(0.05)
            active_requests -= 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "data": {"positions": [SAMPLE_EIGHTFOLD_POSTING_FULL]},
            }
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get

        companies = [
            EightfoldCompany(
                name=f"Company {i}",
                hostname=f"comp{i}.eightfold.ai",
                domain=f"comp{i}.com",
            )
            for i in range(6)
        ]
        source = EightfoldAISource(
            companies=companies,
            max_concurrency=max_concurrency,
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        assert len(jobs) == 6
        assert peak_requests <= max_concurrency
        assert mock_client.get.call_count == 6

    @pytest.mark.asyncio
    async def test_error_resilience_partial_failure(self) -> None:
        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            if "failcorp" in url:
                resp = MagicMock()
                resp.status_code = 500
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "data": {"positions": [SAMPLE_EIGHTFOLD_POSTING_FULL]},
            }
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get

        source = EightfoldAISource(
            companies=[
                EightfoldCompany(name="Good Corp", hostname="goodcorp.eightfold.ai", domain="goodcorp.com"),
                EightfoldCompany(name="Fail Corp", hostname="failcorp.eightfold.ai", domain="failcorp.com"),
            ],
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        # Failing company does not block good company
        assert len(jobs) == 1
        assert jobs[0].company == "Good Corp"

    @pytest.mark.asyncio
    async def test_empty_companies_list(self) -> None:
        source = EightfoldAISource(companies=[])
        jobs = await source.fetch_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_preferences_filtering_and_limit(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "positions": [
                    SAMPLE_EIGHTFOLD_POSTING_FULL,
                    SAMPLE_EIGHTFOLD_POSTING_REMOTE,
                ]
            }
        }
        mock_client.get.return_value = mock_response

        company = EightfoldCompany(
            name="NVIDIA",
            hostname="nvidia.eightfold.ai",
            domain="nvidia.com",
        )
        source = EightfoldAISource(companies=[company], client=mock_client)

        prefs = JobPreferences(work_mode=WorkMode.REMOTE)
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1
        assert jobs[0].title == "Staff AI Researcher - Remote"

        all_jobs = await source.fetch_jobs(limit=1)
        assert len(all_jobs) == 1


class TestEightfoldSourceHealthCheck:
    """Tests for EightfoldAISource check_health."""

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        source = EightfoldAISource(client=mock_client)
        assert await source.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_non_200(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get.return_value = mock_response

        source = EightfoldAISource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Network error")

        source = EightfoldAISource(client=mock_client)
        assert await source.check_health() is False


class TestCreateDefaultRegistryWithEightfold:
    """Tests for create_default_registry with ENABLE_EIGHTFOLD env var."""

    def test_default_registration_without_eightfold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from job_mcp.sources import create_default_registry

        monkeypatch.delenv("ENABLE_EIGHTFOLD", raising=False)
        reg = create_default_registry()
        assert "eightfold" not in reg

    @pytest.mark.parametrize("env_val", ["true", "1", "yes", "TRUE", "True"])
    def test_registration_with_enable_eightfold(self, monkeypatch: pytest.MonkeyPatch, env_val: str) -> None:
        from job_mcp.sources import create_default_registry

        monkeypatch.setenv("ENABLE_EIGHTFOLD", env_val)
        reg = create_default_registry()
        assert "eightfold" in reg
        assert isinstance(reg.get("eightfold"), EightfoldAISource)
