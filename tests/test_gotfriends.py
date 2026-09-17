"""Tests for GotFriendsSource job source."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources.contracts import SourceCategory
from job_mcp.sources.public.gotfriends import (
    GOTFRIENDS_BASE_URL,
    GOTFRIENDS_CATEGORIES,
    GotFriendsSource,
    parse_gotfriends_job_item,
)

SAMPLE_JOB_HTML = """
<div class="item">
    <a href="/jobslobby/ai/ai-engineer/155031/" class="position p-27092">
        <h2 class="title">AI Senior Software Engineer משרות הובלה בסטארט-אפ רפואי &amp; חדשני</h2>
    </a>
    <div class="item_content">
        <div class="inner">
            <section class="info meta">
                <dl>
                    <dd>
                        <span class="info-label">מיקום:</span> <span class="info-data">
                            ת&quot;א והמרכז
                        </span>
                    </dd>
                </dl>
            </section>
            <div class="desc">
                <div class="title_c">תיאור המשרה:</div>
                החברה פיתחה פתרון פורץ דרך שעוזר בהצלת חיים ומחפשת מהנדס/ת עם ניסיון מעמיק ב-Python, PyTorch ו-Docker.
            </div>
        </div>
    </div>
</div>
"""

SAMPLE_REMOTE_JOB_HTML = """
<div class="item">
    <a href="/jobslobby/software/backend/155032/" class="position p-27093">
        <h2 class="title">Backend Developer</h2>
    </a>
    <div class="item_content">
        <div class="inner">
            <section class="info meta">
                <dl>
                    <dd>
                        <span class="info-label">מיקום:</span> <span class="info-data">
                            עבודה מהבית / Remote
                        </span>
                    </dd>
                </dl>
            </section>
            <div class="desc">
                <div class="title_c">תיאור המשרה:</div>
                משרה מלאה מהבית עם Python ו-FastAPI.
            </div>
        </div>
    </div>
</div>
"""

SAMPLE_HYBRID_JOB_HTML = """
<div class="item">
    <a href="/jobslobby/algorithm/machine-learning/155033/" class="position p-27094">
        <h2 class="title">ML Engineer</h2>
    </a>
    <div class="item_content">
        <div class="inner">
            <section class="info meta">
                <dl>
                    <dd>
                        <span class="info-label">מיקום:</span> <span class="info-data">
                            הרצליה (מודל היברידי)
                        </span>
                    </dd>
                </dl>
            </section>
            <div class="desc">
                <div class="title_c">תיאור המשרה:</div>
                עבודה במודל היברידי משולב. דרוש ידע ב-PyTorch ו-Kubernetes.
            </div>
        </div>
    </div>
</div>
"""


class TestGotFriendsCategories:
    """Tests for GOTFRIENDS_CATEGORIES dictionary."""

    def test_categories_not_empty(self):
        assert len(GOTFRIENDS_CATEGORIES) >= 10

    def test_key_categories_present(self):
        expected_keys = [
            "ai",
            "ai_engineer",
            "llm_engineer",
            "algorithm",
            "machine_learning",
            "algorithm_engineer",
            "data_scientist",
            "deep_learning",
            "software",
            "backend",
        ]
        for key in expected_keys:
            assert key in GOTFRIENDS_CATEGORIES
            assert GOTFRIENDS_CATEGORIES[key].startswith("/jobslobby/")


class TestParseGotFriendsJobItem:
    """Tests for parse_gotfriends_job_item parser."""

    def test_parses_sample_job(self):
        job = parse_gotfriends_job_item(SAMPLE_JOB_HTML)
        assert job is not None
        assert job.job_id == "gotfriends_155031"
        assert "AI Senior Software Engineer משרות הובלה בסטארט-אפ רפואי & חדשני" == job.title
        assert job.company == "GotFriends"
        assert job.source == "gotfriends"
        assert job.url == f"{GOTFRIENDS_BASE_URL}/jobslobby/ai/ai-engineer/155031/"
        assert job.apply_url == f"{GOTFRIENDS_BASE_URL}/jobslobby/ai/ai-engineer/155031/"
        assert job.location == 'ת"א והמרכז'
        assert "החברה פיתחה פתרון פורץ דרך" in job.description
        assert "<div" not in job.description
        assert any("python" in s.lower() for s in job.tech_stack)
        assert any("pytorch" in s.lower() for s in job.tech_stack)
        assert job.work_mode == WorkMode.ONSITE

    def test_parses_remote_job(self):
        job = parse_gotfriends_job_item(SAMPLE_REMOTE_JOB_HTML)
        assert job is not None
        assert job.job_id == "gotfriends_155032"
        assert job.work_mode == WorkMode.REMOTE

    def test_parses_hybrid_job(self):
        job = parse_gotfriends_job_item(SAMPLE_HYBRID_JOB_HTML)
        assert job is not None
        assert job.job_id == "gotfriends_155033"
        assert job.work_mode == WorkMode.HYBRID

    def test_handles_missing_location(self):
        html_without_loc = """
        <div class="item">
            <a href="/jobslobby/ai/155034/" class="position">
                <h2 class="title">Data Scientist</h2>
            </a>
            <div class="desc">
                Python and SQL needed.
            </div>
        </div>
        """
        job = parse_gotfriends_job_item(html_without_loc)
        assert job is not None
        assert job.location == "Israel"

    def test_returns_none_for_invalid_html(self):
        invalid_html = '<div class="sidebar">No job here</div>'
        assert parse_gotfriends_job_item(invalid_html) is None

    def test_unescapes_html_entities_in_title_and_location(self):
        html_entities = """
        <div class="item">
            <a href="/jobslobby/ai/155035/">
                <h2 class="title">&quot;Tech&quot; &amp; AI &lt;Lead&gt;</h2>
            </a>
            <span class="info-data">ת&quot;א &amp; רמת גן</span>
            <div class="desc">Some desc</div>
        </div>
        """
        job = parse_gotfriends_job_item(html_entities)
        assert job is not None
        assert job.title == '"Tech" & AI <Lead>'
        assert job.location == 'ת"א & רמת גן'


class TestGotFriendsSource:
    """Tests for GotFriendsSource job source."""

    def test_source_metadata(self):
        source = GotFriendsSource()
        assert source.source_id == "gotfriends"
        assert source.display_name == "GotFriends"
        assert source.category == SourceCategory.PUBLIC
        assert source.supports_auto_apply is True
        assert source.supports_bookmarks is False

    def test_request_headers_user_agent(self):
        from job_mcp.sources.public.gotfriends import REQUEST_HEADERS, REPO_URL
        assert "TechJobMCP/1.0" in REQUEST_HEADERS["User-Agent"]
        assert REPO_URL in REQUEST_HEADERS["User-Agent"]
        assert bytes.fromhex("7a7669656c69").decode() not in REQUEST_HEADERS["User-Agent"].lower()

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        source = GotFriendsSource()
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_resp):
            healthy = await source.check_health()
            assert healthy is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        source = GotFriendsSource()
        mock_resp = AsyncMock()
        mock_resp.status_code = 503

        with patch("httpx.AsyncClient.get", return_value=mock_resp):
            healthy = await source.check_health()
            assert healthy is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self):
        source = GotFriendsSource()
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            healthy = await source.check_health()
            assert healthy is False

    @pytest.mark.asyncio
    async def test_fetch_jobs_returns_jobs_and_deduplicates(self):
        source = GotFriendsSource(
            categories={
                "ai": "/jobslobby/ai/",
                "ai_engineer": "/jobslobby/ai/ai-engineer/",
            },
            request_delay=0.0,
        )

        page_html = f"<html><body>{SAMPLE_JOB_HTML}{SAMPLE_REMOTE_JOB_HTML}</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = page_html
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            jobs = await source.fetch_jobs(limit=50)

        # Both categories returned the same 2 jobs; deduplication should keep exactly 2
        assert len(jobs) == 2
        job_ids = [j.job_id for j in jobs]
        assert "gotfriends_155031" in job_ids
        assert "gotfriends_155032" in job_ids

    @pytest.mark.asyncio
    async def test_fetch_jobs_respects_limit(self):
        source = GotFriendsSource(
            categories={"ai": "/jobslobby/ai/"},
            request_delay=0.0,
        )

        page_html = f"<html><body>{SAMPLE_JOB_HTML}{SAMPLE_REMOTE_JOB_HTML}{SAMPLE_HYBRID_JOB_HTML}</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = page_html
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            jobs = await source.fetch_jobs(limit=2)

        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_empty_categories(self):
        source = GotFriendsSource(categories={}, request_delay=0.0)
        jobs = await source.fetch_jobs(limit=10)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_jobs_handles_http_errors_gracefully(self):
        source = GotFriendsSource(
            categories={"ai": "/jobslobby/ai/", "backend": "/jobslobby/backend/"},
            request_delay=0.0,
        )

        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 500
        mock_resp_fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_resp_fail)
        )

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.text = f"<html><body>{SAMPLE_JOB_HTML}</body></html>"
        mock_resp_ok.raise_for_status = MagicMock()

        async def fake_get(url, **kwargs):
            if "ai" in url:
                return mock_resp_ok
            raise httpx.ConnectError("Network issue")

        with patch("httpx.AsyncClient.get", side_effect=fake_get):
            jobs = await source.fetch_jobs(limit=10)

        assert len(jobs) == 1
        assert jobs[0].job_id == "gotfriends_155031"

    @pytest.mark.asyncio
    async def test_fetch_jobs_filters_by_preferences(self):
        source = GotFriendsSource(
            categories={"ai": "/jobslobby/ai/"},
            request_delay=0.0,
        )

        page_html = f"<html><body>{SAMPLE_JOB_HTML}{SAMPLE_REMOTE_JOB_HTML}</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = page_html
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            prefs = JobPreferences(exclude_keywords=["Backend"])
            jobs = await source.fetch_jobs(preferences=prefs, limit=10)

        assert len(jobs) == 1
        assert jobs[0].job_id == "gotfriends_155031"


class TestGotFriendsBackwardCompatAndRegistry:
    """Tests for backward compat shim, public __init__, and SourceRegistry integration."""

    def test_backward_compat_shim_import(self):
        from job_mcp.sources.gotfriends import (
            GOTFRIENDS_CATEGORIES as SHIM_CATS,
            GotFriendsSource as ShimSource,
            parse_gotfriends_job_item as shim_parse,
        )
        assert ShimSource is GotFriendsSource
        assert SHIM_CATS is GOTFRIENDS_CATEGORIES
        assert shim_parse is parse_gotfriends_job_item

    def test_sources_public_init_exports(self):
        import job_mcp.sources.public as pub
        assert hasattr(pub, "GotFriendsSource")
        assert hasattr(pub, "GOTFRIENDS_CATEGORIES")
        assert hasattr(pub, "parse_gotfriends_job_item")

    def test_sources_init_exports(self):
        import job_mcp.sources as src
        assert hasattr(src, "GotFriendsSource")
        assert hasattr(src, "GOTFRIENDS_CATEGORIES")
        assert hasattr(src, "parse_gotfriends_job_item")

    def test_registry_integration(self):
        from job_mcp.sources.registry import (
            create_default_registry,
            get_registered_providers,
        )

        providers = get_registered_providers()
        assert "gotfriends" in providers
        assert providers["gotfriends"].default_enabled is True
        assert providers["gotfriends"].env_var == "ENABLE_GOTFRIENDS"

        reg = create_default_registry()
        assert "gotfriends" in reg
        assert isinstance(reg.get("gotfriends"), GotFriendsSource)

        reg_disabled = create_default_registry(enable_gotfriends=False)
        assert "gotfriends" not in reg_disabled
