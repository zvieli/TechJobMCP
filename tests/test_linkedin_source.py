"""Unit tests for Lightweight LinkedIn job source, HTML parsers, search API, rate limiting, and caching."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources import BaseJobSource, SourceRegistry
from job_mcp.sources.linkedin import (
    LINKEDIN_HEADERS,
    LINKEDIN_JOB_DETAIL_URL,
    LINKEDIN_SEARCH_API_URL,
    LinkedInSource,
    parse_linkedin_job_card,
    parse_linkedin_job_details,
    parse_linkedin_search_results,
    search_linkedin_jobs_api,
)

SAMPLE_LINKEDIN_CARD_FULL = """
<li class="jobs-search-results__list-item">
  <div class="base-card relative w-full hover:no-underline focus:no-underline base-card--link base-search-card base-search-card--link job-search-card" data-entity-urn="urn:li:jobPosting:4152839402">
    <a class="base-card__full-link absolute top-0 right-0 bottom-0 left-0 p-0 z-[2]" href="https://www.linkedin.com/jobs/view/senior-python-engineer-at-acme-corp-4152839402?position=1&amp;pageNum=0&amp;refId=abc&amp;trackingId=xyz">
      <span class="sr-only">Senior Python Engineer</span>
    </a>
    <div class="base-search-card__info">
      <h3 class="base-search-card__title">
        Senior Python Engineer
      </h3>
      <h4 class="base-search-card__subtitle">
        <a class="hidden-nested-link" href="https://www.linkedin.com/company/acme-corp">Acme Corp</a>
      </h4>
      <div class="base-search-card__metadata">
        <span class="job-search-card__location">Tel Aviv, Israel (Hybrid)</span>
        <time class="job-search-card__listdate" datetime="2026-08-15">3 days ago</time>
      </div>
      <p class="job-search-card__snippet">
        We are building high-throughput microservices with FastAPI, PostgreSQL, Docker, and Kubernetes.
      </p>
    </div>
  </div>
</li>
"""

SAMPLE_LINKEDIN_CARD_REMOTE = """
<div class="job-search-card" data-job-id="4152839403">
  <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/4152839403">
    <h3 class="base-search-card__title">Lead Rust Developer</h3>
  </a>
  <h4 class="base-search-card__subtitle">CryptoNext</h4>
  <span class="job-search-card__location">Remote, Israel</span>
  <time class="job-search-card__listdate--new" datetime="2026-08-18">1 day ago</time>
  <p class="job-search-card__snippet">Expertise in Rust, Solidity, Web3, and Tokio.</p>
</div>
"""

SAMPLE_LINKEDIN_CARD_ONSITE = """
<div class="job-search-card" data-job-id="4152839404">
  <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/onsite-devops-4152839404">
    <h3 class="base-search-card__title">Site Reliability Engineer - Onsite</h3>
  </a>
  <h4 class="base-search-card__subtitle">IronDefense</h4>
  <span class="job-search-card__location">Herzliya, Tel Aviv, Israel</span>
</div>
"""

SAMPLE_LINKEDIN_CARD_MINIMAL = """
<div class="job-search-card">
  <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/4152839405">Junior QA Automation</a>
  <h4 class="base-search-card__subtitle">Testify</h4>
</div>
"""

SAMPLE_LINKEDIN_SEARCH_PAGE = f"""
<ul class="jobs-search__results-list">
  {SAMPLE_LINKEDIN_CARD_FULL}
  {SAMPLE_LINKEDIN_CARD_REMOTE}
  {SAMPLE_LINKEDIN_CARD_ONSITE}
</ul>
"""

SAMPLE_LINKEDIN_JOB_DETAILS_HTML = """
<div class="decorated-job-posting__details">
  <section class="top-card-layout">
    <h1 class="top-card-layout__title font-sans text-lg">Staff Cloud Architect</h1>
    <a class="topcard__org-name-link" href="https://www.linkedin.com/company/cloudscale">CloudScale Technologies</a>
    <span class="topcard__flavor topcard__flavor--bullet">Tel Aviv-Yafo, Tel Aviv District, Israel</span>
    <span class="posted-time-ago__text">2 weeks ago</span>
  </section>
  <div class="show-more-less-html__markup show-more-less-html__markup--clamp-after-5">
    <p>CloudScale is hiring a Staff Cloud Architect to lead infrastructure strategy.</p>
    <p>Key requirements:</p>
    <ul>
      <li>10+ years designing distributed cloud architectures with AWS, GCP, and Azure</li>
      <li>Strong expertise in Kubernetes, Terraform, Docker, Python, and Go</li>
      <li>Experience leading cross-functional engineering initiatives</li>
    </ul>
  </div>
  <div class="job-details-how-to-apply">
    <a class="apply-button" href="https://cloudscale.io/careers/apply/987654">Apply on company website</a>
  </div>
  <ul class="description__job-criteria-list">
    <li>
      <h3 class="description__job-criteria-subheader">Seniority level</h3>
      <span class="description__job-criteria-text">Mid-Senior level</span>
    </li>
    <li>
      <h3 class="description__job-criteria-subheader">Employment type</h3>
      <span class="description__job-criteria-text">Full-time</span>
    </li>
    <li>
      <h3 class="description__job-criteria-subheader">Job function</h3>
      <span class="description__job-criteria-text">Engineering and Information Technology</span>
    </li>
    <li>
      <h3 class="description__job-criteria-subheader">Industries</h3>
      <span class="description__job-criteria-text">Software Development</span>
    </li>
  </ul>
</div>
"""


class TestLinkedInParsers:
    """Tests for LinkedIn HTML parser functions."""

    def test_parse_full_card(self) -> None:
        job = parse_linkedin_job_card(SAMPLE_LINKEDIN_CARD_FULL)
        assert job is not None
        assert job.job_id == "linkedin_4152839402"
        assert job.title == "Senior Python Engineer"
        assert job.company == "Acme Corp"
        assert job.source == "linkedin"
        assert job.sources == ["linkedin"]
        assert "Tel Aviv" in job.location
        assert job.work_mode == WorkMode.HYBRID
        assert job.posted_date == "2026-08-15"
        assert "FastAPI" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "Kubernetes" in job.tech_stack
        assert "4152839402" in (job.url or "")
        assert not (job.url or "").endswith("refId=abc&trackingId=xyz")
        assert job.apply_url is not None

    def test_parse_remote_card(self) -> None:
        job = parse_linkedin_job_card(SAMPLE_LINKEDIN_CARD_REMOTE)
        assert job is not None
        assert job.job_id == "linkedin_4152839403"
        assert job.title == "Lead Rust Developer"
        assert job.company == "CryptoNext"
        assert job.work_mode == WorkMode.REMOTE
        assert "Rust" in job.tech_stack
        assert "Solidity" in job.tech_stack
        assert "Web3" in job.tech_stack

    def test_parse_onsite_card(self) -> None:
        job = parse_linkedin_job_card(SAMPLE_LINKEDIN_CARD_ONSITE)
        assert job is not None
        assert job.job_id == "linkedin_4152839404"
        assert job.title == "Site Reliability Engineer - Onsite"
        assert job.company == "IronDefense"
        assert job.work_mode == WorkMode.ONSITE

    def test_parse_minimal_card(self) -> None:
        job = parse_linkedin_job_card(SAMPLE_LINKEDIN_CARD_MINIMAL)
        assert job is not None
        assert job.job_id == "linkedin_4152839405"
        assert job.title == "Junior QA Automation"
        assert job.company == "Testify"
        assert job.source == "linkedin"

    def test_parse_malformed_card(self) -> None:
        assert parse_linkedin_job_card("") is None
        assert parse_linkedin_job_card("   \n\t  ") is None
        assert parse_linkedin_job_card("<div><span>Just some random text with no job</span></div>") is None

    def test_parse_search_results_multiple(self) -> None:
        jobs = parse_linkedin_search_results(SAMPLE_LINKEDIN_SEARCH_PAGE)
        assert len(jobs) == 3
        job_ids = [j.job_id for j in jobs]
        assert "linkedin_4152839402" in job_ids
        assert "linkedin_4152839403" in job_ids
        assert "linkedin_4152839404" in job_ids

    def test_parse_search_results_empty(self) -> None:
        assert parse_linkedin_search_results("") == []
        assert parse_linkedin_search_results("<html><body>No jobs found</body></html>") == []

    def test_parse_job_details(self) -> None:
        details = parse_linkedin_job_details(SAMPLE_LINKEDIN_JOB_DETAILS_HTML)
        assert details["title"] == "Staff Cloud Architect"
        assert details["company"] == "CloudScale Technologies"
        assert "Tel Aviv" in details["location"]
        assert "posted_date" in details
        assert details["apply_url"] == "https://cloudscale.io/careers/apply/987654"
        assert details["seniority_level"] == "Mid-Senior level"
        assert details["employment_type"] == "Full-time"
        assert "Engineering" in details["department"]
        assert "AWS" in details["tech_stack"]
        assert "Kubernetes" in details["tech_stack"]
        assert "Terraform" in details["tech_stack"]
        assert "Python" in details["tech_stack"]
        assert "Go" in details["tech_stack"]
        assert "CloudScale is hiring a Staff Cloud Architect" in details["description"]


class TestLinkedInSearchApi:
    """Tests for search_linkedin_jobs_api function with mock network calls."""

    @pytest.mark.asyncio
    async def test_search_api_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        jobs = await search_linkedin_jobs_api(
            keywords="Python",
            location="Israel",
            start=0,
            client=mock_client,
        )
        assert len(jobs) == 3
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["params"]["keywords"] == "Python"
        assert call_kwargs["params"]["location"] == "Israel"
        assert call_kwargs["params"]["start"] == 0

    @pytest.mark.asyncio
    async def test_search_api_work_mode_filter(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        jobs = await search_linkedin_jobs_api(
            keywords="FastAPI",
            work_mode=WorkMode.REMOTE,
            client=mock_client,
        )
        assert len(jobs) == 3
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["params"]["f_WT"] == "2"

    @pytest.mark.asyncio
    async def test_search_api_rate_limit_backoff(self) -> None:
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "0.01"}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.side_effect = [mock_response_429, mock_response_200]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            jobs = await search_linkedin_jobs_api(
                keywords="Python",
                client=mock_client,
                max_retries=2,
            )
            assert len(jobs) == 3
            assert mock_client.get.call_count == 2
            mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_search_api_rate_limit_exhausted(self) -> None:
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response_429

        with patch("asyncio.sleep", new_callable=AsyncMock):
            jobs = await search_linkedin_jobs_api(
                keywords="Python",
                client=mock_client,
                max_retries=2,
            )
            assert jobs == []
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_search_api_network_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        jobs = await search_linkedin_jobs_api(
            keywords="Python",
            client=mock_client,
            max_retries=1,
        )
        assert jobs == []


class TestLinkedInSource:
    """Tests for LinkedInSource BaseJobSource implementation."""

    def test_metadata(self) -> None:
        source = LinkedInSource()
        assert isinstance(source, BaseJobSource)
        assert source.source_id == "linkedin"
        assert source.display_name == "LinkedIn"
        meta = source.get_metadata()
        assert meta.source_id == "linkedin"
        assert meta.display_name == "LinkedIn"

    @pytest.mark.asyncio
    async def test_fetch_jobs_basic(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 3
        assert all(j.source == "linkedin" for j in jobs)

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        prefs = JobPreferences(
            keywords=["Python"],
            tech_stack=["FastAPI"],
            work_mode=WorkMode.HYBRID,
            location="Tel Aviv",
        )
        jobs = await source.fetch_jobs(preferences=prefs, limit=10)
        assert len(jobs) >= 1
        assert jobs[0].work_mode == WorkMode.HYBRID

    @pytest.mark.asyncio
    async def test_fetch_jobs_caching(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client, cache_ttl_seconds=3600)
        # First call -> hits mock client
        jobs1 = await source.fetch_jobs(limit=10)
        assert len(jobs1) == 3
        assert mock_client.get.call_count == 1

        # Second identical call -> hits cache
        jobs2 = await source.fetch_jobs(limit=10)
        assert len(jobs2) == 3
        assert mock_client.get.call_count == 1

        # Invalidate cache
        source.clear_cache()
        jobs3 = await source.fetch_jobs(limit=10)
        assert len(jobs3) == 3
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_jobs_pagination(self) -> None:
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.text = SAMPLE_LINKEDIN_CARD_FULL

        mock_client = AsyncMock()
        mock_client.get.side_effect = [mock_response_1, mock_response_2]

        source = LinkedInSource(client=mock_client)
        jobs = await source.fetch_jobs(limit=50)
        assert len(jobs) >= 3
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_job_details(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_JOB_DETAILS_HTML

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        details = await source.fetch_job_details("4152839402")
        assert details is not None
        assert details["title"] == "Staff Cloud Architect"
        assert details["seniority_level"] == "Mid-Senior level"

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_CARD_MINIMAL

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        assert await source.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectTimeout("Timeout")

        source = LinkedInSource(client=mock_client)
        assert await source.check_health() is False

    @pytest.mark.asyncio
    async def test_bookmark_job_without_session(self) -> None:
        source = LinkedInSource()
        assert not source.supports_bookmarks
        assert await source.bookmark_job("123") is False

    @pytest.mark.asyncio
    async def test_bookmark_job_with_session(self) -> None:
        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.bookmark_job = AsyncMock(return_value=True)

        source = LinkedInSource(session_manager=mock_session)
        assert source.supports_bookmarks is True
        res = await source.bookmark_job("linkedin_4152839402")
        assert res is True
        mock_session.bookmark_job.assert_called_once_with("linkedin_4152839402")

    @pytest.mark.asyncio
    async def test_bookmark_job_session_exception(self) -> None:
        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.bookmark_job = AsyncMock(side_effect=RuntimeError("Playwright error"))

        source = LinkedInSource(session_manager=mock_session)
        assert await source.bookmark_job("linkedin_123") is False

    @pytest.mark.asyncio
    async def test_fetch_job_details_caching_and_non_200(self) -> None:
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response_500

        source = LinkedInSource(client=mock_client)
        assert await source.fetch_job_details("linkedin_9999") is None

        # Empty job id
        assert await source.fetch_job_details("") is None

    @pytest.mark.asyncio
    async def test_fetch_job_details_network_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Network error")

        source = LinkedInSource(client=mock_client)
        assert await source.fetch_job_details("12345") is None

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_empty_preferences(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_LINKEDIN_SEARCH_PAGE

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        source = LinkedInSource(client=mock_client)
        prefs = JobPreferences()
        jobs = await source.fetch_jobs(preferences=prefs, limit=10)
        assert len(jobs) == 3


class TestLinkedInHelpers:
    """Tests for internal helper functions."""

    def test_clean_html_text(self) -> None:
        from job_mcp.sources.linkedin import _clean_html_text

        assert _clean_html_text(None) == ""
        assert _clean_html_text("") == ""
        assert _clean_html_text("<p>Hello &amp; <strong>World</strong> &lt;AI&gt;</p>") == "Hello & World <AI>"
        assert _clean_html_text("   Lots   of   \n\t whitespace   ") == "Lots of whitespace"

    def test_extract_job_id_from_text(self) -> None:
        from job_mcp.sources.linkedin import _extract_job_id_from_text

        # 1. URN
        assert _extract_job_id_from_text('<div data-entity-urn="urn:li:jobPosting:112233">') == "112233"
        # 2. data-job-id
        assert _extract_job_id_from_text('<div data-job-id="445566">') == "445566"
        # 3. data-id
        assert _extract_job_id_from_text('<div data-id="778899">') == "778899"
        # 4. href view URL
        assert _extract_job_id_from_text('<a href="https://www.linkedin.com/jobs/view/dev-123456?refId=1">') == "123456"
        # 5. href currentJobId
        assert _extract_job_id_from_text('<a href="https://www.linkedin.com/jobs/search?currentJobId=987654">') == "987654"
        # None
        assert _extract_job_id_from_text('<div>no id here</div>') is None

    def test_clean_linkedin_url(self) -> None:
        from job_mcp.sources.linkedin import _clean_linkedin_url

        assert _clean_linkedin_url(None, "12345") == "https://www.linkedin.com/jobs/view/12345"
        assert _clean_linkedin_url("https://www.linkedin.com/jobs/view/dev-123?trackingId=xyz&pos=1", None) == "https://www.linkedin.com/jobs/view/dev-123"
        assert _clean_linkedin_url(None, None) == ""


class TestLinkedInSourceRegistry:
    """Tests for LinkedInSource registration and environment toggle."""

    def test_manual_registration(self) -> None:
        reg = SourceRegistry()
        source = LinkedInSource()
        reg.register(source)
        assert "linkedin" in reg
        assert reg.get("linkedin") is source

    def test_default_registry_env_toggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from job_mcp.sources import create_default_registry

        monkeypatch.setenv("ENABLE_LINKEDIN", "true")
        reg = create_default_registry()
        assert "linkedin" in reg
        assert isinstance(reg.get("linkedin"), LinkedInSource)

        monkeypatch.setenv("ENABLE_LINKEDIN", "false")
        reg2 = create_default_registry()
        assert "linkedin" not in reg2

