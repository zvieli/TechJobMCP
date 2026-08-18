"""Unit tests for Comeet ATS job source, position parser, concurrency, and caching."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hireme_mcp.models.schemas import JobPreferences, WorkMode
from hireme_mcp.sources import BaseJobSource, SourceRegistry
from hireme_mcp.sources.comeet import (
    DEFAULT_COMEET_COMPANIES,
    ComeetCompany,
    ComeetSource,
    parse_comeet_position,
)


# Sample Comeet API payloads
SAMPLE_COMEET_POSITION_FULL = {
    "uid": "POS.101",
    "position_uid": "POS.101",
    "name": "Senior Python Backend Engineer",
    "department": "Engineering",
    "url_comeet_hosted_page": "https://www.comeet.com/jobs/acme/12.345/python-dev/POS.101",
    "url_recruit_hosted_page": "https://www.comeet.com/jobs/acme/12.345/python-dev/POS.101",
    "url_active_page": "https://www.acme.com/careers#POS.101",
    "workplace_type": "Hybrid",
    "location": {
        "name": "Tel Aviv-Yafo",
        "city": "Tel Aviv",
        "country": "Israel",
        "is_remote": False,
    },
    "details": [
        {
            "name": "Description",
            "value": "<p>We are seeking a senior engineer with expertise in <strong>FastAPI, PostgreSQL, Docker, and Kubernetes</strong>.</p>",
            "order": 1,
        },
        {
            "name": "Requirements",
            "value": "<p>5+ years of experience with Python, AWS, and Microservices.</p>",
            "order": 2,
        },
    ],
}

SAMPLE_COMEET_POSITION_REMOTE = {
    "uid": "POS.202",
    "name": "DevOps Engineer - Remote",
    "department": "Infrastructure",
    "url_comeet_hosted_page": "https://www.comeet.com/jobs/acme/12.345/devops/POS.202",
    "workplace_type": "Remote",
    "location": {
        "name": "Remote, Israel",
        "city": "Remote",
        "country": "Israel",
        "is_remote": True,
    },
    "details": {
        "description": "Kubernetes and Terraform expert needed for cloud infrastructure.",
    },
}

SAMPLE_COMEET_POSITION_MINIMAL = {
    "name": "Frontend Developer",
    "details": "React, TypeScript, CSS",
}


class TestComeetPositionParser:
    """Tests for parse_comeet_position."""

    def test_parse_full_position(self) -> None:
        job = parse_comeet_position(SAMPLE_COMEET_POSITION_FULL, company_name="Acme Tech")
        assert job.job_id == "comeet_POS.101"
        assert job.title == "Senior Python Backend Engineer"
        assert job.company == "Acme Tech"
        assert job.source == "comeet"
        assert job.sources == ["comeet"]
        assert job.department == "Engineering"
        assert job.url == "https://www.comeet.com/jobs/acme/12.345/python-dev/POS.101"
        assert job.apply_url == "https://www.comeet.com/jobs/acme/12.345/python-dev/POS.101"
        assert "Tel Aviv" in job.location
        assert job.work_mode == WorkMode.HYBRID
        assert "FastAPI" in job.tech_stack or "Python" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "Kubernetes" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack
        assert "FastAPI" in job.description or "Python" in job.description

    def test_parse_remote_position(self) -> None:
        job = parse_comeet_position(SAMPLE_COMEET_POSITION_REMOTE, company_name="Cloud Corp")
        assert job.job_id == "comeet_POS.202"
        assert job.title == "DevOps Engineer - Remote"
        assert job.company == "Cloud Corp"
        assert job.work_mode == WorkMode.REMOTE
        assert "Kubernetes" in job.tech_stack
        assert "Terraform" in job.tech_stack

    def test_parse_minimal_and_missing_fields(self) -> None:
        job = parse_comeet_position(SAMPLE_COMEET_POSITION_MINIMAL, company_name="Startup Inc")
        assert job.job_id.startswith("comeet_")
        assert job.title == "Frontend Developer"
        assert job.company == "Startup Inc"
        assert job.location == ""
        assert "React" in job.tech_stack
        assert "TypeScript" in job.tech_stack

    def test_parse_location_formats(self) -> None:
        # String location
        raw1 = {"name": "QA Engineer", "location": "Haifa, Israel"}
        job1 = parse_comeet_position(raw1, "Haifa Tech")
        assert "Haifa" in job1.location

        # Location dict with only name
        raw2 = {"name": "QA Engineer", "location": {"name": "Jerusalem"}}
        job2 = parse_comeet_position(raw2, "JLM Tech")
        assert "Jerusalem" in job2.location

        # Location dict with is_remote=True
        raw3 = {"name": "QA Engineer", "location": {"city": "Tel Aviv", "is_remote": True}}
        job3 = parse_comeet_position(raw3, "TLV Tech")
        assert job3.work_mode == WorkMode.REMOTE

    def test_parse_fallback_urls(self) -> None:
        raw = {
            "uid": "123",
            "name": "Product Manager",
            "url_recruit_hosted_page": "https://recruit.comeet.co/pm/123",
        }
        job = parse_comeet_position(raw, "PM Corp")
        assert job.url == "https://recruit.comeet.co/pm/123"
        assert job.apply_url == "https://recruit.comeet.co/pm/123"


class TestComeetSourceMetadataAndDirectory:
    """Tests for metadata, company directory, and company registration."""

    def test_source_attributes(self) -> None:
        source = ComeetSource()
        assert source.source_id == "comeet"
        assert source.display_name == "Comeet (Direct ATS)"
        assert "Comeet ATS" in source.description
        assert source.is_authenticated is False
        assert source.supports_bookmarks is False
        assert source.supports_auto_apply is False

        meta = source.get_metadata()
        assert meta.source_id == "comeet"
        assert meta.display_name == "Comeet (Direct ATS)"
        assert meta.is_authenticated is False

    def test_default_companies_contains_commit(self) -> None:
        source = ComeetSource()
        companies = source.get_companies()
        commit_entry = next((c for c in companies if c.name == "Comm-IT"), None)
        assert commit_entry is not None
        assert commit_entry.uid == "76.008"
        assert commit_entry.token == "67826D067833C0CF002D48020581368"

    def test_add_and_remove_company(self) -> None:
        source = ComeetSource(companies=[])
        assert len(source.get_companies()) == 0

        source.add_company(uid="11.222", name="CyberArk", token="TOKEN123")
        companies = source.get_companies()
        assert len(companies) == 1
        assert companies[0].name == "CyberArk"
        assert companies[0].uid == "11.222"

        removed = source.remove_company("11.222")
        assert removed is not None
        assert removed.name == "CyberArk"
        assert len(source.get_companies()) == 0

    def test_registry_integration(self) -> None:
        reg = SourceRegistry()
        source = ComeetSource()
        reg.register(source)
        assert "comeet" in reg
        assert reg.get("comeet") is source


class TestComeetSourceFetchingAndCaching:
    """Tests for ComeetSource fetch_jobs, caching, error resilience, and semaphore."""

    @pytest.mark.asyncio
    async def test_fetch_jobs_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [SAMPLE_COMEET_POSITION_FULL]
        mock_client.get.return_value = mock_response

        source = ComeetSource(
            companies=[{"uid": "76.008", "name": "Comm-IT", "token": "TOKEN"}],
            client=mock_client,
        )

        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].job_id == "comeet_POS.101"
        assert jobs[0].company == "Comm-IT"
        assert jobs[0].source == "comeet"
        assert jobs[0].sources == ["comeet"]
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_ttl_caching(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [SAMPLE_COMEET_POSITION_FULL]
        mock_client.get.return_value = mock_response

        source = ComeetSource(
            companies=[{"uid": "76.008", "name": "Comm-IT", "token": "TOKEN"}],
            client=mock_client,
            cache_ttl_seconds=3600,
        )

        # 1st call triggers HTTP request
        jobs1 = await source.fetch_jobs()
        assert len(jobs1) == 1
        assert mock_client.get.call_count == 1

        # 2nd call hits cache, no new HTTP request
        jobs2 = await source.fetch_jobs()
        assert len(jobs2) == 1
        assert mock_client.get.call_count == 1

        # Invalidate cache
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
            resp.json.return_value = [SAMPLE_COMEET_POSITION_FULL]
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get

        # 6 companies with max_concurrency = 2
        test_companies = [
            {"uid": f"uid_{i}", "name": f"Company {i}", "token": f"token_{i}"}
            for i in range(6)
        ]
        source = ComeetSource(
            companies=test_companies,
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
            if "fail_uid" in url:
                resp = MagicMock()
                resp.status_code = 500
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = [SAMPLE_COMEET_POSITION_FULL]
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get

        source = ComeetSource(
            companies=[
                {"uid": "good_uid", "name": "Good Corp", "token": "TOKEN1"},
                {"uid": "fail_uid", "name": "Failing Corp", "token": "TOKEN2"},
            ],
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        # Failing company does not prevent good company from returning jobs
        assert len(jobs) == 1
        assert jobs[0].company == "Good Corp"

    @pytest.mark.asyncio
    async def test_preferences_filtering_and_limit(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            SAMPLE_COMEET_POSITION_FULL,
            SAMPLE_COMEET_POSITION_REMOTE,
        ]
        mock_client.get.return_value = mock_response

        source = ComeetSource(
            companies=[{"uid": "76.008", "name": "Comm-IT", "token": "TOKEN"}],
            client=mock_client,
        )

        # Filter by remote
        prefs = JobPreferences(work_mode=WorkMode.REMOTE)
        remote_jobs = await source.fetch_jobs(preferences=prefs)
        assert len(remote_jobs) == 1
        assert remote_jobs[0].title == "DevOps Engineer - Remote"

        # Limit test
        all_jobs = await source.fetch_jobs(limit=1)
        assert len(all_jobs) == 1


class TestComeetSourceHealthCheck:
    """Tests for ComeetSource check_health."""

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        source = ComeetSource(client=mock_client)
        assert await source.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_non_200(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_client.get.return_value = mock_response

        source = ComeetSource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Network unreachable")

        source = ComeetSource(client=mock_client)
        assert await source.check_health() is False
