"""Unit tests for Jobify job source, JSON-LD parsing, related URL crawling, and registry integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources import SourceRegistry, create_default_registry
from job_mcp.sources.jobify import (
    DEFAULT_JOBIFY_SEED_URLS,
    JOBIFY_BASE_URL,
    JOBIFY_HEADERS,
    JobifySource,
    extract_jsonld_job_postings,
    extract_related_job_urls,
    parse_jobify_position,
)


SAMPLE_JSONLD_JOBPOSTING = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "ראש צוות פיתוח מערכות מידע",
    "description": "הובלת פיתוח מערכת ניהול תיקי התביעה. דרוש ניסיון ב-Python, Docker, Kubernetes ו-PostgreSQL.",
    "identifier": {
        "@type": "PropertyValue",
        "name": "Jobify",
        "value": "191_302995",
    },
    "datePosted": "2026-08-24",
    "validThrough": "2026-10-16T06:49:52.833481Z",
    "employmentType": "FULL_TIME",
    "hiringOrganization": {
        "@type": "Organization",
        "name": "צבא ההגנה לישראל",
        "logo": "https://cdn.jobify360.co.il/idf.png",
    },
    "jobLocation": {
        "@type": "Place",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "תל אביב - יפו",
            "addressCountry": "IL",
        },
    },
}

SAMPLE_JSONLD_REMOTE = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "Senior Python Backend Developer (Remote)",
    "description": "<p>עבודה מלאה מהבית בפיתוח מערכות Backend מבוססות <strong>FastAPI, AWS, Redis</strong>.</p>",
    "identifier": {
        "@type": "PropertyValue",
        "name": "Jobify",
        "value": "8821105",
    },
    "datePosted": "2026-09-15",
    "jobLocationType": "TELECOMMUTE",
    "hiringOrganization": {
        "@type": "Organization",
        "name": "עידור מחשבים בע&quot;מ",
    },
    "jobLocation": {
        "@type": "Place",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "הרצליה",
            "addressCountry": "IL",
        },
    },
}

SAMPLE_HTML_PAGE_1 = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <title>ראש צוות פיתוח מערכות מידע</title>
    <script type="application/ld+json">
    {json.dumps(SAMPLE_JSONLD_JOBPOSTING, ensure_ascii=False)}
    </script>
    <script type="application/ld+json">
    {{
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": []
    }}
    </script>
</head>
<body>
    <h1>ראש צוות פיתוח מערכות מידע</h1>
    <div class="recommended-jobs">
        <a href="/jobs/8821105-aj">Senior Python Developer</a>
        <a href="https://jobify360.co.il/jobs/4465715150-in">Fullstack Engineer</a>
        <a href="/about">About Jobify</a>
    </div>
</body>
</html>
"""

SAMPLE_HTML_PAGE_2 = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <title>Senior Python Backend Developer</title>
    <script type="application/ld+json">
    {json.dumps(SAMPLE_JSONLD_REMOTE, ensure_ascii=False)}
    </script>
</head>
<body>
    <h1>Senior Python Backend Developer</h1>
    <div class="recommended-jobs">
        <a href="/jobs/191_302995-emp">Previous Job</a>
    </div>
</body>
</html>
"""


class TestJobifyPositionParser:
    """Tests for parse_jobify_position."""

    def test_parse_full_job_posting(self) -> None:
        url = "https://jobify360.co.il/jobs/191_302995-emp"
        job = parse_jobify_position(SAMPLE_JSONLD_JOBPOSTING, url=url)

        assert job.job_id == "jobify_191_302995"
        assert job.title == "ראש צוות פיתוח מערכות מידע"
        assert job.company == "צבא ההגנה לישראל"
        assert "תל אביב" in job.location
        assert job.source == "jobify"
        assert job.sources == ["jobify"]
        assert job.url == url
        assert job.apply_url == url
        assert job.posted_date == "2026-08-24"
        assert "Python" in job.tech_stack
        assert "Docker" in job.tech_stack
        assert "Kubernetes" in job.tech_stack
        assert "PostgreSQL" in job.tech_stack
        assert job.work_mode == WorkMode.ONSITE

    def test_jobify_sections_and_tech_stack_isolation(self) -> None:
        payload = {
            "@type": "JobPosting",
            "title": "React Frontend Engineer",
            "hiringOrganization": {"name": "SaaS Platform"},
            "jobLocation": {"address": {"addressLocality": "Tel Aviv"}},
            "description": """
                <h2>About the Company</h2>
                <p>We are a high-scale data platform utilizing Kubernetes, Go, and Kafka.</p>
                <h2>Responsibilities</h2>
                <p>Build and maintain responsive client-side apps with React.</p>
                <h2>Requirements</h2>
                <p>Strong experience in TypeScript, React, and GraphQL.</p>
            """,
        }
        job = parse_jobify_position(payload, url="https://jobify360.co.il/jobs/112233")
        assert job.company_overview is not None and "high-scale data platform" in job.company_overview
        assert job.responsibilities is not None and "client-side apps" in job.responsibilities
        assert job.requirements is not None and "TypeScript" in job.requirements
        assert "React" in job.tech_stack
        assert "TypeScript" in job.tech_stack
        assert "GraphQL" in job.tech_stack
        # Company overview boilerplate keywords MUST NOT bleed into tech stack
        assert "Go" not in job.tech_stack
        assert "Kubernetes" not in job.tech_stack
        assert "Kafka" not in job.tech_stack

    def test_parse_remote_and_unescape(self) -> None:
        url = "https://jobify360.co.il/jobs/8821105-aj"
        job = parse_jobify_position(SAMPLE_JSONLD_REMOTE, url=url)

        assert job.job_id == "jobify_8821105"
        assert job.title == "Senior Python Backend Developer (Remote)"
        # HTML entity unescaped
        assert job.company == 'עידור מחשבים בע"מ'
        assert job.work_mode == WorkMode.REMOTE
        assert "FastAPI" in job.tech_stack or "Python" in job.tech_stack
        assert "AWS" in job.tech_stack
        assert "Redis" in job.tech_stack
        # HTML tags stripped from description
        assert "<p>" not in job.description
        assert "<strong>" not in job.description
        assert "עבודה מלאה מהבית" in job.description

    def test_fallback_job_id_from_url(self) -> None:
        payload = {
            "@type": "JobPosting",
            "title": "Data Engineer",
            "hiringOrganization": {"name": "Tech Corp"},
            "description": "SQL and Python pipelines",
        }
        url = "https://jobify360.co.il/jobs/998877-custom"
        job = parse_jobify_position(payload, url=url)
        assert job.job_id == "jobify_998877-custom"

    def test_already_prefixed_job_id_not_duplicated(self) -> None:
        payload = {
            "@type": "JobPosting",
            "title": "DevOps Specialist",
            "identifier": {"value": "jobify_12345"},
            "hiringOrganization": "DevOps Ltd",
        }
        job = parse_jobify_position(payload)
        assert job.job_id == "jobify_12345"


class TestExtractJsonLdJobPostings:
    """Tests for extract_jsonld_job_postings."""

    def test_extract_single_job_posting(self) -> None:
        postings = extract_jsonld_job_postings(SAMPLE_HTML_PAGE_1)
        assert len(postings) == 1
        assert postings[0]["title"] == "ראש צוות פיתוח מערכות מידע"

    def test_extract_graph_structure(self) -> None:
        html = """
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@graph": [
                {"@type": "Organization", "name": "Org1"},
                {"@type": "JobPosting", "title": "Graph Job", "description": "Graph desc"}
            ]
        }
        </script>
        """
        postings = extract_jsonld_job_postings(html)
        assert len(postings) == 1
        assert postings[0]["title"] == "Graph Job"

    def test_extract_malformed_ignored(self) -> None:
        html = """
        <script type="application/ld+json">
        { invalid json content }
        </script>
        <script type="application/ld+json">
        {"@type": "JobPosting", "title": "Valid Job"}
        </script>
        """
        postings = extract_jsonld_job_postings(html)
        assert len(postings) == 1
        assert postings[0]["title"] == "Valid Job"

    def test_extract_empty_when_no_job_postings(self) -> None:
        html = """
        <html><body><p>No json-ld</p></body></html>
        """
        assert extract_jsonld_job_postings(html) == []


class TestExtractRelatedJobUrls:
    """Tests for extract_related_job_urls."""

    def test_extract_related_job_urls(self) -> None:
        urls = extract_related_job_urls(SAMPLE_HTML_PAGE_1, base_url="https://jobify360.co.il")
        assert "https://jobify360.co.il/jobs/8821105-aj" in urls
        assert "https://jobify360.co.il/jobs/4465715150-in" in urls
        # Non-job link ignored
        assert not any("/about" in u for u in urls)

    def test_extract_deduplicates_and_ignores_fragments(self) -> None:
        html = """
        <a href="/jobs/123-aj">Job 1</a>
        <a href="/jobs/123-aj#apply">Job 1 again</a>
        <a href="https://jobify360.co.il/jobs/123-aj">Job 1 absolute</a>
        <a href="/jobs/logo.png">Image</a>
        """
        urls = extract_related_job_urls(html)
        assert len(urls) == 1
        assert urls[0] == "https://jobify360.co.il/jobs/123-aj"


class TestJobifySource:
    """Tests for JobifySource fetch_jobs, check_health, and snowball crawler."""

    def test_metadata(self) -> None:
        src = JobifySource()
        assert src.source_id == "jobify"
        assert src.display_name == "Jobify"
        assert src.is_authenticated is False
        assert src.supports_bookmarks is False
        assert src.supports_auto_apply is False

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp

        src = JobifySource(client=mock_client)
        assert await src.check_health() is True
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_client.get.return_value = mock_resp

        src = JobifySource(client=mock_client)
        assert await src.check_health() is False

        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        assert await src.check_health() is False

    @pytest.mark.asyncio
    async def test_fetch_jobs_snowball_crawling(self) -> None:
        """Test crawling seed URL discovers related job and yields multiple jobs."""
        seed_url = "https://jobify360.co.il/jobs/191_302995-emp"
        related_url = "https://jobify360.co.il/jobs/8821105-aj"

        async def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "191_302995-emp" in url_str:
                return httpx.Response(200, text=SAMPLE_HTML_PAGE_1)
            elif "8821105-aj" in url_str:
                return httpx.Response(200, text=SAMPLE_HTML_PAGE_2)
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            src = JobifySource(
                seed_urls=[seed_url],
                client=client,
                max_crawl_pages=5,
            )
            jobs = await src.fetch_jobs(limit=10)

        assert len(jobs) >= 2
        job_ids = [j.job_id for j in jobs]
        assert "jobify_191_302995" in job_ids
        assert "jobify_8821105" in job_ids
        for j in jobs:
            assert j.source == "jobify"
            assert "jobify" in j.sources

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences(self) -> None:
        """Test preference filtering filters out non-matching positions."""
        seed_url = "https://jobify360.co.il/jobs/191_302995-emp"

        async def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "191_302995-emp" in url_str:
                return httpx.Response(200, text=SAMPLE_HTML_PAGE_1)
            elif "8821105-aj" in url_str:
                return httpx.Response(200, text=SAMPLE_HTML_PAGE_2)
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            src = JobifySource(seed_urls=[seed_url], client=client)
            prefs = JobPreferences(work_mode=WorkMode.REMOTE, tech_stack=["Redis"])
            jobs = await src.fetch_jobs(preferences=prefs, limit=10)

        assert len(jobs) == 1
        assert jobs[0].job_id == "jobify_8821105"

    @pytest.mark.asyncio
    async def test_fetch_jobs_respects_limit(self) -> None:
        seed_url = "https://jobify360.co.il/jobs/191_302995-emp"

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_HTML_PAGE_1)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            src = JobifySource(seed_urls=[seed_url], client=client)
            jobs = await src.fetch_jobs(limit=1)

        assert len(jobs) == 1


class TestRegistryIntegration:
    """Tests for Jobify integration in SourceRegistry and create_default_registry."""

    def test_default_registration_includes_jobify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENABLE_JOBIFY", raising=False)
        reg = create_default_registry()
        assert "jobify" in reg
        src = reg.get("jobify")
        assert isinstance(src, JobifySource)

    def test_registration_with_explicit_flag(self) -> None:
        reg_enabled = create_default_registry(enable_jobify=True)
        assert "jobify" in reg_enabled

        reg_disabled = create_default_registry(enable_jobify=False)
        assert "jobify" not in reg_disabled

    @pytest.mark.parametrize("env_val", ["false", "0", "no", "FALSE"])
    def test_registration_with_disabled_env(self, monkeypatch: pytest.MonkeyPatch, env_val: str) -> None:
        monkeypatch.setenv("ENABLE_JOBIFY", env_val)
        reg = create_default_registry()
        assert "jobify" not in reg
