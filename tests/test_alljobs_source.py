"""Unit tests for AllJobsSource job source, category fetching, parsing, and error isolation."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources import (
    BaseJobSource,
    SourceRegistry,
)
from job_mcp.sources.alljobs import (
    ALLJOBS_BASE_URL,
    ALLJOBS_HEADERS,
    ALLJOBS_REQUEST_TIMEOUT,
    DEFAULT_TECH_CATEGORIES,
    AllJobsSource,
    parse_alljobs_position,
)


# Sample AllJobs API payloads
SAMPLE_ALLJOBS_JOB_FULL = {
    "JobID": 987654,
    "JobTitle": "Senior Backend Engineer - Python & FastAPI",
    "CompanyName": "CyberArmor Ltd",
    "JobCity": "Tel Aviv",
    "JobRegion": "Center",
    "JobDescription": "<p>Looking for an experienced backend engineer with expertise in <strong>Python, FastAPI, Docker, and PostgreSQL</strong>. Remote flexibility available.</p>",
    "Salary": "35,000 - 45,000 ₪",
    "Date": "2026-08-15",
    "IsRemote": True,
    "CategoryName": "תוכנה",
}

SAMPLE_ALLJOBS_JOB_MINIMAL = {
    "JobID": "555123",
    "JobTitle": "Junior Frontend Developer",
    "CompanyName": "Startup Nation",
    "JobCity": "Herzliya",
    "JobDescription": "React and TypeScript development.",
}

SAMPLE_ALLJOBS_JOB_FALLBACK = {
    "title": "DevOps Engineer",
    "company": "CloudTech",
    "location": "Haifa",
    "description": "Kubernetes, Terraform, AWS CI/CD pipeline management.",
}

SAMPLE_CATEGORIES_PAYLOAD = {
    "Categories": [
        {"CategoryID": 235, "CategoryName": "תוכנה / Software"},
        {"CategoryID": 1998, "CategoryName": "בינה מלאכותית / AI"},
        {"CategoryID": 357, "CategoryName": "חומרה ורשתות / Computers & Networks"},
        {"CategoryID": 237, "CategoryName": "בדיקות תוכנה / QA"},
        {"CategoryID": 1563, "CategoryName": "אינטרנט / Internet"},
    ]
}


class TestAllJobsHeadersAndMetadata:
    """Tests for browser-grade headers and source metadata."""

    def test_browser_headers_configuration(self) -> None:
        """Verify headers contain realistic browser emulation fields."""
        assert "Mozilla" in ALLJOBS_HEADERS["User-Agent"]
        assert "Chrome" in ALLJOBS_HEADERS["User-Agent"]
        assert ALLJOBS_HEADERS["Accept"] == "application/json, text/javascript, */*; q=0.01"
        assert "he-IL" in ALLJOBS_HEADERS["Accept-Language"]
        assert ALLJOBS_HEADERS["X-Requested-With"] == "XMLHttpRequest"
        assert ALLJOBS_HEADERS["Sec-Fetch-Dest"] == "empty"
        assert ALLJOBS_HEADERS["Sec-Fetch-Mode"] == "cors"
        assert ALLJOBS_HEADERS["Sec-Fetch-Site"] == "same-origin"

    def test_alljobs_source_metadata(self) -> None:
        """Verify source descriptor attributes and metadata model."""
        source = AllJobsSource()
        assert source.source_id == "alljobs"
        assert source.display_name == "AllJobs Israel"
        assert "largest" in source.description.lower()
        assert source.is_authenticated is False
        assert source.supports_bookmarks is False
        assert source.supports_auto_apply is False

        metadata = source.get_metadata()
        assert metadata.source_id == "alljobs"
        assert metadata.display_name == "AllJobs Israel"
        assert metadata.is_authenticated is False
        assert metadata.supports_bookmarks is False
        assert metadata.supports_auto_apply is False

    def test_default_request_timeout_constant(self) -> None:
        """Verify default fast request timeout constant is 2.0s."""
        assert ALLJOBS_REQUEST_TIMEOUT == 2.0

    def test_alljobs_source_default_timeout(self) -> None:
        """Verify AllJobsSource default timeout is set to 2.0s."""
        source = AllJobsSource()
        assert source.timeout == 2.0



class TestAllJobsPositionParser:
    """Tests for parse_alljobs_position."""

    def test_parse_full_position(self) -> None:
        job = parse_alljobs_position(SAMPLE_ALLJOBS_JOB_FULL)
        assert job.job_id == "alljobs_987654"
        assert job.title == "Senior Backend Engineer - Python & FastAPI"
        assert job.company == "CyberArmor Ltd"
        assert job.source == "alljobs"
        assert job.sources == ["alljobs"]
        assert "Tel Aviv" in job.location
        assert "Center" in job.location
        assert job.work_mode == WorkMode.REMOTE
        assert "Python" in job.tech_stack
        assert "FastAPI" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack
        assert job.salary_range == "35,000 - 45,000 ₪"
        assert job.posted_date == "2026-08-15"
        assert job.url == "https://www.alljobs.co.il/User/ShowJob.aspx?JobID=987654"
        assert "<p>" not in job.description
        assert "experienced backend engineer" in job.description

    def test_parse_minimal_position(self) -> None:
        job = parse_alljobs_position(SAMPLE_ALLJOBS_JOB_MINIMAL)
        assert job.job_id == "alljobs_555123"
        assert job.title == "Junior Frontend Developer"
        assert job.company == "Startup Nation"
        assert job.location == "Herzliya"
        assert "React" in job.tech_stack
        assert "TypeScript" in job.tech_stack
        assert job.url == "https://www.alljobs.co.il/User/ShowJob.aspx?JobID=555123"

    def test_parse_fallback_hash_id(self) -> None:
        job = parse_alljobs_position(SAMPLE_ALLJOBS_JOB_FALLBACK)
        assert job.job_id.startswith("alljobs_")
        assert job.title == "DevOps Engineer"
        assert job.company == "CloudTech"
        assert job.location == "Haifa"
        assert "Kubernetes" in job.tech_stack
        assert "Terraform" in job.tech_stack
        assert "AWS" in job.tech_stack


class TestAllJobsCategories:
    """Tests for category retrieval, parsing, and caching."""

    @pytest.mark.asyncio
    async def test_fetch_categories_success_and_caching(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD
        mock_client.get.return_value = mock_response

        source = AllJobsSource(client=mock_client)
        categories = await source.fetch_categories()

        assert len(categories) >= 5
        assert categories.get("תוכנה / Software") == 235
        assert categories.get("בינה מלאכותית / AI") == 1998

        # Verify call was made with proper URL and headers
        mock_client.get.assert_called_once()
        args, kwargs = mock_client.get.call_args
        assert "/SearchResultsMobile.ashx" in args[0]
        assert kwargs["params"] == {"action": "getSearchEngineData", "categories": "true"}
        assert kwargs["headers"] == ALLJOBS_HEADERS

        # Second call should use cache and not invoke client.get again
        cached_categories = await source.fetch_categories()
        assert cached_categories == categories
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_categories_fallback_on_error(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Network unreachable")

        source = AllJobsSource(client=mock_client)
        categories = await source.fetch_categories()

        # Should return fallback/default tech categories without raising exception
        assert isinstance(categories, dict)
        assert "software" in categories or "Software" in str(categories)

    @pytest.mark.asyncio
    async def test_fetch_categories_standalone_without_client_closes_properly(self) -> None:
        """Verify fetch_categories on uninitialized source creates and properly closes active_client."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_instance = AsyncMock()
            mock_client_cls.return_value = mock_instance
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = SAMPLE_CATEGORIES_PAYLOAD
            mock_instance.get.return_value = mock_resp

            source = AllJobsSource()
            categories = await source.fetch_categories()

            assert len(categories) >= 5
            mock_instance.aclose.assert_awaited_once()


class TestAllJobsFetchJobs:
    """Tests for job fetching, filtering, and pagination."""

    @pytest.mark.asyncio
    async def test_fetch_jobs_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        cat_response = MagicMock(spec=httpx.Response)
        cat_response.status_code = 200
        cat_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD

        jobs_response = MagicMock(spec=httpx.Response)
        jobs_response.status_code = 200
        jobs_response.json.return_value = {
            "Jobs": [SAMPLE_ALLJOBS_JOB_FULL, SAMPLE_ALLJOBS_JOB_MINIMAL]
        }

        # Mock GET to return cat_response on category call, jobs_response on feed calls
        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            params = kwargs.get("params", {})
            if params.get("categories") == "true":
                return cat_response
            return jobs_response

        mock_client.get.side_effect = mock_get

        source = AllJobsSource(client=mock_client)
        jobs = await source.fetch_jobs(limit=10)

        assert len(jobs) == 2
        assert jobs[0].job_id == "alljobs_987654"
        assert jobs[0].source == "alljobs"
        assert jobs[1].job_id == "alljobs_555123"

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences_filtering(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        cat_response = MagicMock(spec=httpx.Response)
        cat_response.status_code = 200
        cat_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD

        jobs_response = MagicMock(spec=httpx.Response)
        jobs_response.status_code = 200
        jobs_response.json.return_value = {
            "Jobs": [SAMPLE_ALLJOBS_JOB_FULL, SAMPLE_ALLJOBS_JOB_MINIMAL]
        }

        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            params = kwargs.get("params", {})
            if params.get("categories") == "true":
                return cat_response
            return jobs_response

        mock_client.get.side_effect = mock_get

        source = AllJobsSource(client=mock_client)

        # 1. Filter by work mode & exclusion
        prefs = JobPreferences(work_mode=WorkMode.REMOTE, exclude_keywords=["Frontend"])
        jobs = await source.fetch_jobs(preferences=prefs, limit=10)

        assert len(jobs) == 1
        assert jobs[0].job_id == "alljobs_987654"
        assert "Python" in jobs[0].tech_stack
        assert jobs[0].work_mode == WorkMode.REMOTE

        # 2. Ranking check by tech stack
        prefs_stack = JobPreferences(tech_stack=["Python", "FastAPI"])
        ranked_jobs = await source.fetch_jobs(preferences=prefs_stack, limit=10)
        assert len(ranked_jobs) == 2
        assert ranked_jobs[0].job_id == "alljobs_987654"
        assert ranked_jobs[0].match_score == 100.0

        # 3. Limit enforcement
        limited_jobs = await source.fetch_jobs(limit=1)
        assert len(limited_jobs) == 1

    @pytest.mark.asyncio
    async def test_feed_caching_and_clear_cache(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        cat_response = MagicMock(spec=httpx.Response)
        cat_response.status_code = 200
        cat_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD

        jobs_response = MagicMock(spec=httpx.Response)
        jobs_response.status_code = 200
        jobs_response.json.return_value = {
            "Jobs": [SAMPLE_ALLJOBS_JOB_FULL]
        }

        call_count = 0

        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            params = kwargs.get("params", {})
            if params.get("categories") == "true":
                return cat_response
            return jobs_response

        mock_client.get.side_effect = mock_get

        source = AllJobsSource(client=mock_client, cache_ttl_seconds=3600)
        jobs1 = await source.fetch_jobs(limit=10)
        assert len(jobs1) == 1
        initial_calls = call_count

        # Second call hits cache
        jobs2 = await source.fetch_jobs(limit=10)
        assert len(jobs2) == 1
        assert call_count == initial_calls

        # Clear cache and fetch again
        source.clear_cache()
        jobs3 = await source.fetch_jobs(limit=10)
        assert len(jobs3) == 1
        assert call_count > initial_calls

    @pytest.mark.asyncio
    async def test_fetch_jobs_runs_category_feeds_in_parallel(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        cat_response = MagicMock(spec=httpx.Response)
        cat_response.status_code = 200
        cat_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD

        async def mock_get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
            params = kwargs.get("params", {})
            if params.get("categories") == "true":
                return cat_response
            # Add a small delay to simulate network latency
            await asyncio.sleep(0.05)
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            cat_id = params.get("cat", "default")
            mock_resp.json.return_value = {
                "Jobs": [
                    {
                        "JobID": f"job_cat_{cat_id}",
                        "JobTitle": f"Engineer in Cat {cat_id}",
                        "CompanyName": "Tech Co",
                    }
                ]
            }
            return mock_resp

        mock_client.get.side_effect = mock_get
        source = AllJobsSource(client=mock_client)

        start = asyncio.get_event_loop().time()
        jobs = await source.fetch_jobs(limit=10)
        elapsed = asyncio.get_event_loop().time() - start

        assert len(jobs) == 3  # 3 primary category feeds (235, 1998, 357)
        # 3 parallel calls of 0.05s should take well under 0.12s total (sequential would be >= 0.15s)
        assert elapsed < 0.12, f"Expected concurrent execution under 0.12s, took {elapsed:.2f}s"


class TestAllJobsErrorIsolation:
    """Tests for graceful error isolation across network and HTTP failures."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "side_effect",
        [
            httpx.HTTPStatusError("403 Forbidden", request=MagicMock(), response=MagicMock(status_code=403)),
            httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=MagicMock(status_code=500)),
            httpx.TimeoutException("Read timed out"),
            httpx.ConnectError("Connection refused"),
            httpx.DecodingError("Invalid JSON payload"),
        ],
    )
    async def test_fetch_jobs_isolated_from_http_errors(self, side_effect: Exception) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = side_effect

        source = AllJobsSource(client=mock_client)
        # Should not raise; should return empty list gracefully
        jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_feed_json_decode_error_returns_empty_list(self) -> None:
        """Verify malformed JSON responses catch JSONDecodeError cleanly and return empty list."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_client.get.return_value = mock_response

        source = AllJobsSource(client=mock_client)
        result = await source._fetch_feed(mock_client, {"action": "getJobs", "cat": 235})
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_feed_uses_request_timeout(self) -> None:
        """Verify _fetch_feed passes the default 2.0s request timeout."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"Jobs": []}
        mock_client.get.return_value = mock_response

        source = AllJobsSource(client=mock_client)
        await source._fetch_feed(mock_client, {"action": "getJobs", "cat": 235})

        mock_client.get.assert_called_once()
        _, kwargs = mock_client.get.call_args
        assert kwargs["timeout"] == 2.0

    @pytest.mark.asyncio
    async def test_fetch_jobs_creates_client_with_fast_timeout(self) -> None:
        """Verify standalone fetch_jobs initializes httpx.AsyncClient with fast timeout=2.0."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_instance = AsyncMock()
            mock_client_cls.return_value = mock_instance
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"Jobs": [SAMPLE_ALLJOBS_JOB_MINIMAL]}
            mock_instance.get.return_value = mock_resp

            source = AllJobsSource()
            jobs = await source.fetch_jobs(limit=10)

            assert len(jobs) == 1
            mock_client_cls.assert_called_with(timeout=2.0)
            mock_instance.aclose.assert_awaited_once()


class TestAllJobsHealthCheck:
    """Tests for check_health() method."""

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_CATEGORIES_PAYLOAD
        mock_client.get.return_value = mock_response

        source = AllJobsSource(client=mock_client)
        is_healthy = await source.check_health()
        assert is_healthy is True

        # Verify 5s timeout used for health check
        mock_client.get.assert_called_once()
        _, kwargs = mock_client.get.call_args
        assert kwargs["timeout"] == 5.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,payload",
        [
            (403, {}),
            (500, {}),
            (200, {"Categories": []}),
            (200, {}),
        ],
    )
    async def test_check_health_unhealthy_responses(self, status_code: int, payload: Any) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.json.return_value = payload
        mock_client.get.return_value = mock_response

        source = AllJobsSource(client=mock_client)
        is_healthy = await source.check_health()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")

        source = AllJobsSource(client=mock_client)
        is_healthy = await source.check_health()
        assert is_healthy is False


class TestAllJobsRegistryIntegration:
    """Tests for SourceRegistry integration with AllJobsSource."""

    def test_registry_registration(self) -> None:
        registry = SourceRegistry()
        source = AllJobsSource()
        registry.register(source)

        assert "alljobs" in registry
        retrieved = registry.get("alljobs")
        assert retrieved is source
        assert retrieved.display_name == "AllJobs Israel"
