"""Unit tests for Workday ATS job source, position parser, company presets, concurrency, and caching."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources import BaseJobSource, SourceRegistry
from job_mcp.sources.workday import (
    WORKDAY_COMPANIES,
    WorkdayCompany,
    WorkdaySource,
    parse_workday_position,
)

SAMPLE_WORKDAY_POSTING_FULL = {
    "title": "Senior Cloud Infrastructure Engineer",
    "externalPath": "/job/Haifa/Senior-Cloud-Infrastructure-Engineer_R-100234",
    "locationsText": "Haifa, Israel",
    "postedOn": "Posted 2 Days Ago",
    "bulletFields": ["R-100234"],
    "timeType": "Full time",
    "workplaceType": "Hybrid",
    "jobPostingInfo": {
        "jobReqId": "R-100234",
        "jobDescription": "<p>We are seeking an engineer experienced in <strong>Python, Docker, Kubernetes, and AWS</strong>.</p>",
        "location": "Haifa, Israel",
        "timeType": "Full time",
        "subfunction": "Software Engineering",
    },
}

SAMPLE_WORKDAY_POSTING_REMOTE = {
    "title": "Staff Backend Engineer - Remote",
    "externalPath": "/job/Tel-Aviv/Staff-Backend-Engineer_R-55555",
    "locationsText": "Remote, Israel",
    "postedOn": "Posted 5 Days Ago",
    "bulletFields": ["R-55555"],
    "workplaceType": "Remote",
    "jobPostingInfo": {
        "jobReqId": "R-55555",
        "jobDescription": "FastAPI, PostgreSQL, Microservices, and Redis.",
        "location": "Remote",
    },
}

SAMPLE_WORKDAY_POSTING_MINIMAL = {
    "title": "Frontend React Developer",
    "externalPath": "/job/Frontend-React-Developer_R-999",
    "bulletFields": ["R-999"],
}


class TestWorkdayPositionParser:
    """Tests for parse_workday_position."""

    def test_parse_full_position(self) -> None:
        company = WorkdayCompany(
            name="Cisco",
            wd_company="cisco",
            wd_version=5,
            wd_suffix="Cisco_Careers",
            wd_locations=["loc123"],
        )
        job = parse_workday_position(SAMPLE_WORKDAY_POSTING_FULL, company=company)
        assert job.job_id == "workday_cisco_R-100234"
        assert job.title == "Senior Cloud Infrastructure Engineer"
        assert job.company == "Cisco"
        assert job.source == "workday"
        assert job.sources == ["workday"]
        assert job.location == "Haifa, Israel"
        assert job.work_mode == WorkMode.HYBRID
        assert job.posted_date == "Posted 2 Days Ago"
        assert job.url == "https://cisco.wd5.myworkdayjobs.com/en-US/Cisco_Careers/job/Haifa/Senior-Cloud-Infrastructure-Engineer_R-100234"
        assert job.apply_url == job.url
        assert "Python" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "Kubernetes" in job.tech_stack
        assert "AWS" in job.tech_stack
        assert "Python, Docker, Kubernetes, and AWS" in job.description
        assert job.department == "Software Engineering"

    def test_parse_remote_position(self) -> None:
        company = WorkdayCompany(
            name="Philips",
            wd_company="philips",
            wd_version=3,
            wd_suffix="jobs-and-careers",
        )
        job = parse_workday_position(SAMPLE_WORKDAY_POSTING_REMOTE, company=company)
        assert job.job_id == "workday_philips_R-55555"
        assert job.title == "Staff Backend Engineer - Remote"
        assert job.company == "Philips"
        assert job.work_mode == WorkMode.REMOTE
        assert "FastAPI" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack
        assert "Redis" in job.tech_stack

    def test_parse_minimal_and_fallback_fields(self) -> None:
        company = WorkdayCompany(
            name="NVIDIA",
            wd_company="nvidia",
            wd_version=5,
            wd_suffix="NVIDIAExternalCareerSite",
        )
        job = parse_workday_position(SAMPLE_WORKDAY_POSTING_MINIMAL, company=company)
        assert job.job_id == "workday_nvidia_R-999"
        assert job.title == "Frontend React Developer"
        assert job.company == "NVIDIA"
        assert job.location == ""
        assert "React" in job.tech_stack

    def test_parse_fallback_id_without_bullet_fields(self) -> None:
        company = WorkdayCompany(name="Acme", wd_company="acme")
        raw = {
            "title": "DevOps Engineer",
            "externalPath": "/job/DevOps_REQ-1234",
            "jobPostingInfo": {"jobReqId": "REQ-1234"},
        }
        job = parse_workday_position(raw, company=company)
        assert job.job_id == "workday_acme_REQ-1234"

    def test_parse_absolute_and_relative_urls(self) -> None:
        company = WorkdayCompany(
            name="Qualcomm",
            wd_company="qualcomm",
            wd_version=5,
            wd_suffix="External",
        )
        # Already absolute
        raw1 = {
            "title": "Firmware Engineer",
            "externalPath": "https://qualcomm.wd5.myworkdayjobs.com/en-US/External/job/Haifa/123",
            "bulletFields": ["123"],
        }
        job1 = parse_workday_position(raw1, company=company)
        assert job1.url == "https://qualcomm.wd5.myworkdayjobs.com/en-US/External/job/Haifa/123"

        # Relative without leading slash
        raw2 = {
            "title": "Firmware Engineer",
            "externalPath": "job/Haifa/123",
            "bulletFields": ["123"],
        }
        job2 = parse_workday_position(raw2, company=company)
        assert job2.url == "https://qualcomm.wd5.myworkdayjobs.com/en-US/External/job/Haifa/123"


class TestWorkdaySourceMetadataAndPresets:
    """Tests for metadata, company presets, and registration."""

    def test_source_attributes(self) -> None:
        source = WorkdaySource()
        assert source.source_id == "workday"
        assert "Workday" in source.display_name
        assert "Workday" in source.description
        assert source.is_authenticated is False
        assert source.supports_bookmarks is False
        assert source.supports_auto_apply is False

        meta = source.get_metadata()
        assert meta.source_id == "workday"
        assert meta.is_authenticated is False

    def test_workday_companies_presets_coverage(self) -> None:
        assert isinstance(WORKDAY_COMPANIES, dict)
        expected_companies = ["microsoft", "cisco", "philips", "qualcomm", "ptc", "nvidia"]
        for comp_key in expected_companies:
            assert comp_key in WORKDAY_COMPANIES, f"Missing preset for {comp_key}"
            comp = WORKDAY_COMPANIES[comp_key]
            if isinstance(comp, WorkdayCompany):
                assert comp.name
                assert comp.wd_company
                assert comp.wd_version > 0
                assert comp.wd_suffix
            elif isinstance(comp, dict):
                assert "name" in comp
                assert "wd_company" in comp

    def test_add_and_remove_company(self) -> None:
        source = WorkdaySource(companies=[])
        assert len(source.get_companies()) == 0

        source.add_company(
            WorkdayCompany(
                name="Custom Corp",
                wd_company="customcorp",
                wd_version=2,
                wd_suffix="Careers",
            )
        )
        assert len(source.get_companies()) == 1
        assert source.get_companies()[0].name == "Custom Corp"

        removed = source.remove_company("customcorp")
        assert removed is not None
        assert removed.name == "Custom Corp"
        assert len(source.get_companies()) == 0

    def test_registry_integration(self) -> None:
        reg = SourceRegistry()
        source = WorkdaySource()
        reg.register(source)
        assert "workday" in reg
        assert reg.get("workday") is source


class TestWorkdaySourceFetchingAndCaching:
    """Tests for WorkdaySource fetch_jobs, caching, error handling, and concurrency."""

    @pytest.mark.asyncio
    async def test_fetch_jobs_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "jobPostings": [SAMPLE_WORKDAY_POSTING_FULL],
        }
        mock_client.post.return_value = mock_response

        company = WorkdayCompany(
            name="Cisco",
            wd_company="cisco",
            wd_version=5,
            wd_suffix="Cisco_Careers",
            wd_locations=["loc123"],
        )
        source = WorkdaySource(companies=[company], client=mock_client)

        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].job_id == "workday_cisco_R-100234"
        assert jobs[0].company == "Cisco"
        assert jobs[0].source == "workday"
        assert jobs[0].sources == ["workday"]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_keyword_search_payload(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "jobPostings": [SAMPLE_WORKDAY_POSTING_FULL],
        }
        mock_client.post.return_value = mock_response

        company = WorkdayCompany(
            name="Cisco",
            wd_company="cisco",
            wd_version=5,
            wd_suffix="Cisco_Careers",
            wd_locations=["loc123"],
        )
        source = WorkdaySource(companies=[company], client=mock_client)

        prefs = JobPreferences(keywords=["python", "cloud"])
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1

        called_args, called_kwargs = mock_client.post.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("searchText") == "python cloud"
        assert payload.get("appliedFacets", {}).get("locations") == ["loc123"]

    @pytest.mark.asyncio
    async def test_ttl_caching(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "jobPostings": [SAMPLE_WORKDAY_POSTING_FULL],
        }
        mock_client.post.return_value = mock_response

        company = WorkdayCompany(name="Cisco", wd_company="cisco")
        source = WorkdaySource(companies=[company], client=mock_client, cache_ttl_seconds=3600)

        # 1st call hits HTTP endpoint
        jobs1 = await source.fetch_jobs()
        assert len(jobs1) == 1
        assert mock_client.post.call_count == 1

        # 2nd call hits cache
        jobs2 = await source.fetch_jobs()
        assert len(jobs2) == 1
        assert mock_client.post.call_count == 1

        # Clear cache and verify fresh fetch
        source.clear_cache()
        jobs3 = await source.fetch_jobs()
        assert len(jobs3) == 1
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrency_semaphore_throttling(self) -> None:
        max_concurrency = 2
        active_requests = 0
        peak_requests = 0

        async def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal active_requests, peak_requests
            active_requests += 1
            if active_requests > peak_requests:
                peak_requests = active_requests
            await asyncio.sleep(0.05)
            active_requests -= 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "total": 1,
                "jobPostings": [SAMPLE_WORKDAY_POSTING_FULL],
            }
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = mock_post

        companies = [
            WorkdayCompany(name=f"Company {i}", wd_company=f"comp_{i}")
            for i in range(6)
        ]
        source = WorkdaySource(
            companies=companies,
            max_concurrency=max_concurrency,
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        assert len(jobs) == 6
        assert peak_requests <= max_concurrency
        assert mock_client.post.call_count == 6

    @pytest.mark.asyncio
    async def test_error_resilience_partial_failure(self) -> None:
        async def mock_post(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            if "failcorp" in url:
                resp = MagicMock()
                resp.status_code = 500
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "total": 1,
                "jobPostings": [SAMPLE_WORKDAY_POSTING_FULL],
            }
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = mock_post

        source = WorkdaySource(
            companies=[
                WorkdayCompany(name="Good Corp", wd_company="goodcorp"),
                WorkdayCompany(name="Fail Corp", wd_company="failcorp"),
            ],
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        # Failing company does not block good company
        assert len(jobs) == 1
        assert jobs[0].company == "Good Corp"

    @pytest.mark.asyncio
    async def test_empty_companies_list(self) -> None:
        source = WorkdaySource(companies=[])
        jobs = await source.fetch_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_preferences_filtering_and_limit(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 2,
            "jobPostings": [
                SAMPLE_WORKDAY_POSTING_FULL,
                SAMPLE_WORKDAY_POSTING_REMOTE,
            ],
        }
        mock_client.post.return_value = mock_response

        company = WorkdayCompany(name="Cisco", wd_company="cisco")
        source = WorkdaySource(companies=[company], client=mock_client)

        prefs = JobPreferences(work_mode=WorkMode.REMOTE)
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1
        assert jobs[0].title == "Staff Backend Engineer - Remote"

        all_jobs = await source.fetch_jobs(limit=1)
        assert len(all_jobs) == 1


class TestWorkdaySourceHealthCheck:
    """Tests for WorkdaySource check_health."""

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response

        source = WorkdaySource(client=mock_client)
        assert await source.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_non_200(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.post.return_value = mock_response

        source = WorkdaySource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Network timeout")

        source = WorkdaySource(client=mock_client)
        assert await source.check_health() is False
