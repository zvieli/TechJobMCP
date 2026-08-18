"""Unit tests for Direct Tech Company Sources (Google, Amazon, Apple, IBM), parsers, presets, concurrency, and caching."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources import BaseJobSource, SourceRegistry, create_default_registry
from job_mcp.sources.direct_tech import (
    DEFAULT_DIRECT_TECH_COMPANIES,
    DIRECT_TECH_COMPANIES,
    DirectTechCompany,
    DirectTechSource,
    parse_amazon_position,
    parse_amazon_positions,
    parse_apple_position,
    parse_apple_positions,
    parse_google_job,
    parse_google_positions,
    parse_ibm_position,
    parse_ibm_positions,
)

SAMPLE_GOOGLE_HTML = """
<!doctype html>
<html>
<head><title>Google Careers</title></head>
<body>
<script>
window.WIZ_global_data = {};
// ds:1 data blob
'ds:1', hash: '12345', data:[[
  "73886960954811078",
  "Software Engineer III, Cloud Infrastructure",
  "https://www.google.com/about/careers/applications/signin?jobId=123",
  [null, "<ul><li>Design distributed systems with Python, Go, and Kubernetes.</li></ul>"],
  [null, "<ul><li>Bachelor's degree in CS.</li><li>Experience with Docker and Cloud.</li></ul>"],
  "projects/gweb-careers-proto/tenants/123",
  null,
  "Google",
  "en-US",
  [
    ["Haifa, Israel", ["Andrei Sakharov St 9, Haifa"], "Haifa", "3508409", "Haifa District", "IL"],
    ["Tel Aviv, Israel", ["Tel Aviv-Yafo"], "Tel Aviv", null, "Tel Aviv District", "IL"]
  ],
  [null, "We are looking for a Software Engineer to build scalable cloud systems with Python, Go, and Kubernetes."],
  [2],
  [1766399300, 732000000]
]],null
</script>
</body>
</html>
"""

SAMPLE_AMAZON_HIT = {
    "fields": {
        "icimsJobId": ["AMZ-998877"],
        "title": ["Software Development Engineer - AWS"],
        "description": ["Join AWS team building distributed services using Java, Python, and AWS."],
        "basicQualifications": ["Bachelor's in Computer Science", "2+ years Python experience"],
        "preferredQualifications": ["Experience with Docker and Kubernetes"],
        "city": ["Haifa"],
        "country": ["ISR"],
        "location": ["Haifa, Israel"],
        "jobCategory": ["Software Development"],
        "postedDate": ["October 15, 2025"],
    }
}

SAMPLE_AMAZON_RESPONSE = {
    "searchHits": [SAMPLE_AMAZON_HIT],
    "hits": 1,
}

SAMPLE_APPLE_JOB = {
    "id": "114438158",
    "postingTitle": "AI/ML Systems Engineer",
    "transformedPostingTitle": "ai-ml-systems-engineer",
    "jobSummary": "Develop cutting-edge machine learning and PyTorch infrastructure in Swift and C++.",
    "locations": [{"name": "Herzliya, Tel Aviv, Israel"}],
    "postDateInFormat": "Sep 20, 2025",
    "team": {"teamName": "Machine Learning and AI"},
}

SAMPLE_APPLE_RESPONSE = {
    "res": {
        "searchResults": [SAMPLE_APPLE_JOB],
        "totalRecords": 1,
    }
}

SAMPLE_IBM_HIT = {
    "_id": "ibm_hit_4455",
    "_source": {
        "title": "Quantum Software Developer - Student Position",
        "url": "https://www.ibm.com/careers/search?jobId=IBM-REQ-889900&lang=en",
        "description": "Research and build quantum algorithms using Qiskit, Python, and Linux.",
        "field_keyword_05": "Israel",
        "field_keyword_08": "Givatayim",
    },
}

SAMPLE_IBM_RESPONSE = {
    "hits": {
        "hits": [SAMPLE_IBM_HIT],
        "total": {"value": 1},
    }
}


class TestDirectTechPositionParsers:
    """Tests for direct career response parsers across Google, Amazon, Apple, and IBM."""

    def test_parse_google_positions_html(self) -> None:
        jobs = parse_google_positions(SAMPLE_GOOGLE_HTML)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.job_id == "direct_google_73886960954811078"
        assert job.title == "Software Engineer III, Cloud Infrastructure"
        assert job.company == "Google"
        assert job.source == "direct_tech"
        assert job.sources == ["direct_tech"]
        assert "Haifa, Israel" in job.location
        assert "Tel Aviv, Israel" in job.location
        assert job.url == "https://www.google.com/about/careers/applications/jobs/results/73886960954811078"
        assert job.apply_url == "https://www.google.com/about/careers/applications/signin?jobId=123"
        assert "Python" in job.tech_stack
        assert "Go" in job.tech_stack
        assert "Kubernetes" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "scalable cloud systems" in job.description

    def test_parse_google_malformed_html(self) -> None:
        jobs = parse_google_positions("<html><body>No jobs here</body></html>")
        assert jobs == []

    def test_parse_amazon_position(self) -> None:
        job = parse_amazon_position(SAMPLE_AMAZON_HIT)
        assert job.job_id == "direct_amazon_AMZ-998877"
        assert job.title == "Software Development Engineer - AWS"
        assert job.company == "Amazon"
        assert job.source == "direct_tech"
        assert job.sources == ["direct_tech"]
        assert job.location == "Haifa, Israel"
        assert job.url == "https://amazon.jobs/jobs/AMZ-998877"
        assert job.apply_url == job.url
        assert "Python" in job.tech_stack
        assert "AWS" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert job.department == "Software Development"
        assert job.posted_date == "October 15, 2025"

    def test_parse_amazon_positions_list(self) -> None:
        jobs = parse_amazon_positions(SAMPLE_AMAZON_RESPONSE)
        assert len(jobs) == 1
        assert jobs[0].company == "Amazon"
        assert jobs[0].job_id == "direct_amazon_AMZ-998877"

    def test_parse_apple_position(self) -> None:
        job = parse_apple_position(SAMPLE_APPLE_JOB)
        assert job.job_id == "direct_apple_114438158"
        assert job.title == "AI/ML Systems Engineer"
        assert job.company == "Apple"
        assert job.source == "direct_tech"
        assert job.sources == ["direct_tech"]
        assert job.location == "Herzliya, Tel Aviv, Israel"
        assert job.url == "https://jobs.apple.com/en-il/details/114438158/ai-ml-systems-engineer"
        assert job.apply_url == job.url
        assert "PyTorch" in job.tech_stack
        assert "C++" in job.tech_stack
        assert job.department == "Machine Learning and AI"
        assert job.posted_date == "Sep 20, 2025"

    def test_parse_apple_positions_dict(self) -> None:
        jobs = parse_apple_positions(SAMPLE_APPLE_RESPONSE)
        assert len(jobs) == 1
        assert jobs[0].company == "Apple"
        assert jobs[0].job_id == "direct_apple_114438158"

    def test_parse_ibm_position(self) -> None:
        job = parse_ibm_position(SAMPLE_IBM_HIT)
        assert job.job_id == "direct_ibm_IBM-REQ-889900"
        assert job.title == "Quantum Software Developer - Student Position"
        assert job.company == "IBM"
        assert job.source == "direct_tech"
        assert job.sources == ["direct_tech"]
        assert "Israel" in job.location
        assert job.url == "https://www.ibm.com/careers/search?jobId=IBM-REQ-889900&lang=en"
        assert job.apply_url == job.url
        assert "Python" in job.tech_stack
        assert "Linux" in job.tech_stack

    def test_parse_ibm_positions_dict(self) -> None:
        jobs = parse_ibm_positions(SAMPLE_IBM_RESPONSE)
        assert len(jobs) == 1
        assert jobs[0].company == "IBM"
        assert jobs[0].job_id == "direct_ibm_IBM-REQ-889900"


class TestDirectTechSourceMetadataAndPresets:
    """Tests for metadata, company presets, and registry integration."""

    def test_source_attributes(self) -> None:
        source = DirectTechSource()
        assert source.source_id == "direct_tech"
        assert "Direct Tech" in source.display_name
        assert "Google" in source.description
        assert source.is_authenticated is False
        assert source.supports_bookmarks is False
        assert source.supports_auto_apply is False

        meta = source.get_metadata()
        assert meta.source_id == "direct_tech"
        assert meta.is_authenticated is False

    def test_direct_tech_companies_presets_coverage(self) -> None:
        assert isinstance(DIRECT_TECH_COMPANIES, dict)
        expected_providers = ["google", "amazon", "apple", "ibm"]
        for prov in expected_providers:
            assert prov in DIRECT_TECH_COMPANIES, f"Missing preset for {prov}"
            comp = DIRECT_TECH_COMPANIES[prov]
            assert isinstance(comp, DirectTechCompany)
            assert comp.provider_id == prov
            assert comp.name
            assert comp.search_url
            assert comp.enabled is True

    def test_add_and_remove_company(self) -> None:
        source = DirectTechSource(companies=[])
        assert len(source.get_companies()) == 0

        source.add_company(
            DirectTechCompany(
                provider_id="custom_provider",
                name="Custom Provider",
                search_url="https://example.com/api/jobs",
            )
        )
        assert len(source.get_companies()) == 1
        assert source.get_companies()[0].name == "Custom Provider"

        removed = source.remove_company("custom_provider")
        assert removed is not None
        assert removed.name == "Custom Provider"
        assert len(source.get_companies()) == 0

    def test_registry_integration_and_env_toggle(self) -> None:
        reg = SourceRegistry()
        source = DirectTechSource()
        reg.register(source)
        assert "direct_tech" in reg
        assert reg.get("direct_tech") is source

        # Test ENABLE_DIRECT_TECH toggle in create_default_registry
        with patch.dict(os.environ, {"ENABLE_DIRECT_TECH": "true"}):
            reg_with = create_default_registry()
            assert "direct_tech" in reg_with

        with patch.dict(os.environ, {"ENABLE_DIRECT_TECH": "false"}):
            reg_without = create_default_registry()
            assert "direct_tech" not in reg_without


class TestDirectTechSourceFetchingAndCaching:
    """Tests for DirectTechSource fetch_jobs, caching, concurrency, and error handling."""

    @pytest.mark.asyncio
    async def test_fetch_jobs_google_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_GOOGLE_HTML
        mock_client.get.return_value = mock_response

        google_comp = DIRECT_TECH_COMPANIES["google"]
        source = DirectTechSource(companies=[google_comp], client=mock_client)

        jobs = await source.fetch_jobs()
        assert len(jobs) == 1
        assert jobs[0].company == "Google"
        assert jobs[0].source == "direct_tech"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_amazon_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_AMAZON_RESPONSE
        mock_client.post.return_value = mock_response

        amazon_comp = DIRECT_TECH_COMPANIES["amazon"]
        source = DirectTechSource(companies=[amazon_comp], client=mock_client)

        jobs = await source.fetch_jobs()
        assert len(jobs) == 1
        assert jobs[0].company == "Amazon"
        assert jobs[0].job_id == "direct_amazon_AMZ-998877"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_apple_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_APPLE_RESPONSE
        mock_client.post.return_value = mock_response

        apple_comp = DIRECT_TECH_COMPANIES["apple"]
        source = DirectTechSource(companies=[apple_comp], client=mock_client)

        jobs = await source.fetch_jobs()
        assert len(jobs) == 1
        assert jobs[0].company == "Apple"
        assert jobs[0].job_id == "direct_apple_114438158"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_ibm_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_IBM_RESPONSE
        mock_client.post.return_value = mock_response

        ibm_comp = DIRECT_TECH_COMPANIES["ibm"]
        source = DirectTechSource(companies=[ibm_comp], client=mock_client)

        jobs = await source.fetch_jobs()
        assert len(jobs) == 1
        assert jobs[0].company == "IBM"
        assert jobs[0].job_id == "direct_ibm_IBM-REQ-889900"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_jobs_parallel_all_providers(self) -> None:
        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.text = SAMPLE_GOOGLE_HTML
            return resp

        async def mock_post(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "amazon" in url:
                resp.json.return_value = SAMPLE_AMAZON_RESPONSE
            elif "apple" in url:
                resp.json.return_value = SAMPLE_APPLE_RESPONSE
            elif "ibm" in url:
                resp.json.return_value = SAMPLE_IBM_RESPONSE
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get
        mock_client.post.side_effect = mock_post

        source = DirectTechSource(client=mock_client)
        jobs = await source.fetch_jobs()

        assert len(jobs) == 4
        companies = {j.company for j in jobs}
        assert companies == {"Google", "Amazon", "Apple", "IBM"}
        for j in jobs:
            assert j.source == "direct_tech"
            assert "direct_tech" in j.sources

    @pytest.mark.asyncio
    async def test_ttl_caching(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_AMAZON_RESPONSE
        mock_client.post.return_value = mock_resp

        amazon_comp = DIRECT_TECH_COMPANIES["amazon"]
        source = DirectTechSource(companies=[amazon_comp], client=mock_client, cache_ttl_seconds=3600)

        # 1st call hits HTTP
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
            resp.json.return_value = SAMPLE_AMAZON_RESPONSE
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = mock_post

        companies = [
            DirectTechCompany(
                provider_id=f"amazon_{i}",
                name=f"Amazon {i}",
                search_url="https://www.amazon.jobs/api/jobs/search",
            )
            for i in range(6)
        ]
        source = DirectTechSource(
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
            if "amazon" in url:
                resp = MagicMock()
                resp.status_code = 500
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = SAMPLE_APPLE_RESPONSE
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = mock_post

        source = DirectTechSource(
            companies=[
                DIRECT_TECH_COMPANIES["amazon"],
                DIRECT_TECH_COMPANIES["apple"],
            ],
            client=mock_client,
        )

        jobs = await source.fetch_jobs()
        # Amazon failure does not crash or block Apple results
        assert len(jobs) == 1
        assert jobs[0].company == "Apple"

    @pytest.mark.asyncio
    async def test_empty_companies_list(self) -> None:
        source = DirectTechSource(companies=[])
        jobs = await source.fetch_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_keyword_search_payload(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_AMAZON_RESPONSE
        mock_client.post.return_value = mock_response

        amazon_comp = DIRECT_TECH_COMPANIES["amazon"]
        source = DirectTechSource(companies=[amazon_comp], client=mock_client)

        prefs = JobPreferences(keywords=["python", "cloud"])
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1

        called_args, called_kwargs = mock_client.post.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("query") == "python cloud"

    @pytest.mark.asyncio
    async def test_preferences_filtering_and_limit(self) -> None:
        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.text = SAMPLE_GOOGLE_HTML
            return resp

        async def mock_post(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "amazon" in url:
                resp.json.return_value = SAMPLE_AMAZON_RESPONSE
            elif "apple" in url:
                resp.json.return_value = SAMPLE_APPLE_RESPONSE
            elif "ibm" in url:
                resp.json.return_value = SAMPLE_IBM_RESPONSE
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = mock_get
        mock_client.post.side_effect = mock_post

        source = DirectTechSource(client=mock_client)

        # Filter by location 'Givatayim'
        prefs = JobPreferences(location="Givatayim")
        jobs = await source.fetch_jobs(preferences=prefs)
        assert len(jobs) == 1
        assert jobs[0].company == "IBM"

        # Exclude keyword 'AWS'
        prefs_exc = JobPreferences(exclude_keywords=["AWS"])
        jobs_exc = await source.fetch_jobs(preferences=prefs_exc)
        assert all("AWS" not in j.title and "AWS" not in j.description for j in jobs_exc)

        # Limit truncation
        all_jobs = await source.fetch_jobs(limit=2)
        assert len(all_jobs) == 2


class TestDirectTechSourceHealthCheck:
    """Tests for DirectTechSource check_health."""

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp
        mock_client.post.return_value = mock_resp

        source = DirectTechSource(client=mock_client)
        assert await source.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_failure_status(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client.get.return_value = mock_resp
        mock_client.post.return_value = mock_resp

        source = DirectTechSource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection timed out")
        mock_client.post.side_effect = httpx.ConnectError("Connection timed out")

        source = DirectTechSource(client=mock_client)
        assert await source.check_health() is False
