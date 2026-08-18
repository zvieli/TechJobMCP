"""Comprehensive tests for direct API client and payload mapping."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from job_mcp.core.api_client import (
    fetch_jobs_via_api,
    fetch_saved_jobs_batch,
    fetch_user_resume_profile,
    parse_api_job_dict,
)
from job_mcp.models.schemas import Job, WorkMode


# ==========================================
# 1. Tests for parse_api_job_dict
# ==========================================


def test_parse_api_job_dict_full_payload():
    """Test parsing a comprehensive API job payload."""
    raw = {
        "id": 141111273,
        "title": "Senior Fullstack Engineer (Python / React)",
        "company_name": "CyberTech Ltd",
        "company": {"name": "CyberTech Ltd", "id": 542},
        "location": {
            "city": "Tel Aviv",
            "full_address": "Tel Aviv, Israel",
            "basic": {
                "city": "Tel Aviv",
                "display_name": "Tel Aviv-Yafo, Israel",
            },
            "work_model": {
                "type": "Hybrid",
                "is_hybrid": True,
                "is_remote": False,
            },
        },
        "skills_required": ["Python", "FastAPI"],
        "skills": ["React", "TypeScript"],
        "tech_stack": ["PostgreSQL", "Docker"],
        "description": "We are seeking an experienced Fullstack Engineer.",
        "requirements": "Must have experience with Kubernetes and AWS.",
        "salary": {
            "formatted": "35,000 - 45,000 ILS",
            "min": 35000,
            "max": 45000,
            "currency": "ILS",
        },
        "posted_date": "2026-08-18T10:00:00Z",
        "job_url": "https://hiremetech.com/jobs/141111273",
        "is_saved": True,
    }

    job = parse_api_job_dict(raw)

    assert isinstance(job, Job)
    assert job.job_id == "141111273"
    assert job.title == "Senior Fullstack Engineer (Python / React)"
    assert job.company == "CyberTech Ltd"
    assert "Tel Aviv" in job.location
    assert job.work_mode == WorkMode.HYBRID
    assert job.is_bookmarked is True
    assert job.salary_range == "35,000 - 45,000 ILS"
    assert job.posted_date == "2026-08-18T10:00:00Z"
    assert job.url == "https://hiremetech.com/jobs/141111273"

    tech_stack_lower = {t.lower() for t in job.tech_stack}
    expected_tech = {"python", "fastapi", "react", "typescript", "postgresql", "docker", "kubernetes", "aws"}
    assert expected_tech.issubset(tech_stack_lower)

    assert "experienced Fullstack Engineer" in job.description
    assert "Kubernetes and AWS" in job.description


def test_parse_api_job_dict_work_mode_variations():
    """Test all variations of work mode mapping."""
    # 1. Location dict with is_remote
    raw_remote = {
        "id": "1",
        "title": "Dev",
        "location": {"work_model": {"is_remote": True}},
    }
    assert parse_api_job_dict(raw_remote).work_mode == WorkMode.REMOTE

    # 2. Location dict with type == "Remote"
    raw_remote_type = {
        "id": "2",
        "title": "Dev",
        "location": {"work_model": {"type": "Remote"}},
    }
    assert parse_api_job_dict(raw_remote_type).work_mode == WorkMode.REMOTE

    # 3. Top-level work_mode string "Hybrid"
    raw_hybrid = {
        "id": "3",
        "title": "Dev",
        "work_mode": "Hybrid",
    }
    assert parse_api_job_dict(raw_hybrid).work_mode == WorkMode.HYBRID

    # 4. Onsite type variations
    for onsite_term in ("Onsite", "On-site", "office"):
        raw_onsite = {
            "id": "4",
            "title": "Dev",
            "location": {"work_model": {"type": onsite_term}},
        }
        assert parse_api_job_dict(raw_onsite).work_mode == WorkMode.ONSITE

    # 5. Infer from location string
    raw_loc_str = {
        "id": "5",
        "title": "Dev",
        "location": "Tel Aviv (Hybrid)",
    }
    assert parse_api_job_dict(raw_loc_str).work_mode == WorkMode.HYBRID


def test_parse_api_job_dict_company_resolution():
    """Test company resolution across company_name, company dict, company string, or missing."""
    assert parse_api_job_dict({"id": "1", "company_name": "Acme Inc"}).company == "Acme Inc"
    assert parse_api_job_dict({"id": "2", "company": {"name": "Beta Corp"}}).company == "Beta Corp"
    assert parse_api_job_dict({"id": "3", "company": "Gamma LLC"}).company == "Gamma LLC"
    assert parse_api_job_dict({"id": "4"}).company == ""


def test_parse_api_job_dict_tech_stack_complex():
    """Test tech stack with object lists, string lists, duplicates, and heuristic text extraction."""
    raw = {
        "id": "tech-1",
        "title": "Go and Rust Backend Architect",
        "skills_required": [{"name": "Go"}, {"skill": "Rust"}],
        "skills": ["PostgreSQL", "Docker", "Go"],  # Go duplicated
        "tech_stack": [{"title": "Kubernetes"}, "Terraform"],
        "description": "Building high throughput streaming systems with Kafka and Redis.",
        "requirements": "Strong background in Linux and AWS.",
    }

    job = parse_api_job_dict(raw)
    tech_lower = [t.lower() for t in job.tech_stack]

    # Check that duplicates are removed
    assert tech_lower.count("go") == 1

    # Check that all sources are included
    for expected in ["go", "rust", "postgresql", "docker", "kubernetes", "terraform", "kafka", "redis", "linux", "aws"]:
        assert expected in tech_lower


def test_parse_api_job_dict_salary_variations():
    """Test salary parsing from formatted, min/max, min only, max only, string, and missing."""
    # Min and Max
    job1 = parse_api_job_dict({"id": "1", "salary": {"min": 25000, "max": 35000, "currency": "ILS"}})
    assert job1.salary_range == "25,000 - 35,000 ILS"

    # Min only
    job2 = parse_api_job_dict({"id": "2", "salary": {"min": 30000, "currency": "USD"}})
    assert job2.salary_range == "30,000+ USD"

    # Max only
    job3 = parse_api_job_dict({"id": "3", "salary": {"max": 40000, "currency": "EUR"}})
    assert job3.salary_range == "Up to 40,000 EUR"

    # String salary
    job4 = parse_api_job_dict({"id": "4", "salary": "Competitive"})
    assert job4.salary_range == "Competitive"

    # String salary_range
    job5 = parse_api_job_dict({"id": "5", "salary_range": "30k-40k ILS"})
    assert job5.salary_range == "30k-40k ILS"

    # Missing salary
    job6 = parse_api_job_dict({"id": "6"})
    assert job6.salary_range is None


def test_parse_api_job_dict_minimal_fallback():
    """Test parsing minimal payload with fallbacks."""
    raw = {
        "id": 12345,
        "title": "Software Developer",
    }

    job = parse_api_job_dict(raw)
    assert job.job_id == "12345"
    assert job.title == "Software Developer"
    assert job.company == ""
    assert job.location == ""
    assert job.work_mode is None
    assert job.is_bookmarked is False
    assert job.salary_range is None
    assert job.description == ""


# ==========================================
# 2. Tests for fetch_jobs_via_api
# ==========================================


@pytest.mark.asyncio
async def test_fetch_jobs_via_api_success():
    """Test successful fetching and parsing of jobs from API."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "jobs": [
            {
                "id": 101,
                "title": "Backend Engineer",
                "company_name": "Tech Corp",
                "skills_required": ["Python", "FastAPI"],
            },
            {
                "id": 102,
                "title": "Frontend Engineer",
                "company_name": "Web Corp",
                "skills_required": ["React", "TypeScript"],
            },
        ],
        "total": 2,
    })

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    jobs = await fetch_jobs_via_api(mock_request_context, page=1, size=20, sort_by="posted_date", sort_order="desc")

    assert len(jobs) == 2
    assert jobs[0].job_id == "101"
    assert jobs[0].title == "Backend Engineer"
    assert jobs[0].company == "Tech Corp"
    assert jobs[1].job_id == "102"

    mock_request_context.get.assert_awaited_once()
    called_url = mock_request_context.get.call_args[0][0]
    assert "/api/jobs/search" in called_url
    assert "page=1" in called_url
    assert "size=20" in called_url
    assert "sort_by=posted_date" in called_url
    assert "sort_order=desc" in called_url
    assert "country=Israel" in called_url


@pytest.mark.asyncio
async def test_fetch_jobs_via_api_data_key_fallback():
    """Test parsing API response where jobs list is under 'data' key."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "data": [
            {"id": "501", "title": "DevOps Engineer", "company": "CloudOps"},
        ]
    })

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    jobs = await fetch_jobs_via_api(mock_request_context)
    assert len(jobs) == 1
    assert jobs[0].job_id == "501"
    assert jobs[0].title == "DevOps Engineer"


@pytest.mark.asyncio
async def test_fetch_jobs_via_api_failure():
    """Test handling of non-200 API response."""
    mock_response = AsyncMock()
    mock_response.status = 500

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="HireMe API returned status 500"):
        await fetch_jobs_via_api(mock_request_context, page=1, size=10)


# ==========================================
# 3. Tests for fetch_saved_jobs_batch
# ==========================================


@pytest.mark.asyncio
async def test_fetch_saved_jobs_batch_success():
    """Test batch checking saved jobs."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "jobs": {
            "101": {"saved": True, "saved_at": "2026-08-18T10:00:00Z"},
            "102": {"saved": False},
        }
    })

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    saved_data = await fetch_saved_jobs_batch(mock_request_context, ["101", "102"])

    assert "101" in saved_data
    assert saved_data["101"]["saved"] is True
    assert saved_data["102"]["saved"] is False

    mock_request_context.get.assert_awaited_once()
    called_url = mock_request_context.get.call_args[0][0]
    assert "/api/saved-jobs/check-batch" in called_url
    assert "job_ids=101,102" in called_url


@pytest.mark.asyncio
async def test_fetch_saved_jobs_batch_empty_input():
    """Test batch checking with empty input returns empty dict without network call."""
    mock_request_context = AsyncMock()

    saved_data = await fetch_saved_jobs_batch(mock_request_context, [])
    assert saved_data == {}
    mock_request_context.get.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_saved_jobs_batch_error():
    """Test non-200 error handling in batch check."""
    mock_response = AsyncMock()
    mock_response.status = 502

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="HireMe saved-jobs batch check returned status 502"):
        await fetch_saved_jobs_batch(mock_request_context, ["101"])


# ==========================================
# 4. Tests for fetch_user_resume_profile
# ==========================================


@pytest.mark.asyncio
async def test_fetch_user_resume_profile_success():
    """Test fetching user resume profile from API."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "technical_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        "github_url": "https://github.com/developer",
        "experience_years": 5,
    })

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    profile = await fetch_user_resume_profile(mock_request_context)

    assert "technical_skills" in profile
    assert "Python" in profile["technical_skills"]
    assert profile["github_url"] == "https://github.com/developer"

    mock_request_context.get.assert_awaited_once()
    called_url = mock_request_context.get.call_args[0][0]
    assert "/api/resume/profile" in called_url


@pytest.mark.asyncio
async def test_fetch_user_resume_profile_error():
    """Test handling error when fetching user resume profile."""
    mock_response = AsyncMock()
    mock_response.status = 401

    mock_request_context = AsyncMock()
    mock_request_context.get = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="HireMe resume profile fetch returned status 401"):
        await fetch_user_resume_profile(mock_request_context)


# ==========================================
# 5. Tests for SessionManager.check_session_health
# ==========================================


@pytest.mark.asyncio
async def test_check_session_health_fast_api_success():
    """Test fast API health check succeeds via /api/auth/me without full page navigation."""
    from job_mcp.core.auth import SessionManager

    manager = SessionManager(user_data_dir="/tmp/test_profile")
    mock_page = AsyncMock()
    mock_page.url = "https://hiremetech.com/he-il/jobs-app"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "user": {
            "id": "usr_987",
            "email": "test@example.com",
            "name": "Test User",
        }
    })
    mock_page.request.get = AsyncMock(return_value=mock_resp)
    manager.get_page = AsyncMock(return_value=mock_page)

    result = await manager.check_session_health()

    assert result is True
    mock_page.request.get.assert_awaited_once()
    called_url = mock_page.request.get.call_args[0][0]
    assert "/api/auth/me" in called_url
    mock_page.goto.assert_not_called()


@pytest.mark.asyncio
async def test_check_session_health_fast_api_unauthenticated_401():
    """Test fast API health check returns False immediately on HTTP 401."""
    from job_mcp.core.auth import SessionManager

    manager = SessionManager(user_data_dir="/tmp/test_profile")
    mock_page = AsyncMock()

    mock_resp = AsyncMock()
    mock_resp.status = 401
    mock_page.request.get = AsyncMock(return_value=mock_resp)
    manager.get_page = AsyncMock(return_value=mock_page)

    result = await manager.check_session_health()

    assert result is False
    mock_page.request.get.assert_awaited_once()
    mock_page.goto.assert_not_called()


@pytest.mark.asyncio
async def test_check_session_health_fast_api_unauthenticated_403():
    """Test fast API health check returns False immediately on HTTP 403."""
    from job_mcp.core.auth import SessionManager

    manager = SessionManager(user_data_dir="/tmp/test_profile")
    mock_page = AsyncMock()

    mock_resp = AsyncMock()
    mock_resp.status = 403
    mock_page.request.get = AsyncMock(return_value=mock_resp)
    manager.get_page = AsyncMock(return_value=mock_page)

    result = await manager.check_session_health()

    assert result is False
    mock_page.request.get.assert_awaited_once()
    mock_page.goto.assert_not_called()


@pytest.mark.asyncio
async def test_check_session_health_api_error_fallback_to_page_goto():
    """Test fallback to page.goto navigation when /api/auth/me request throws network/timeout error."""
    from job_mcp.core.auth import SessionManager

    manager = SessionManager(user_data_dir="/tmp/test_profile")
    mock_page = AsyncMock()
    mock_page.request.get = AsyncMock(side_effect=Exception("Connection timeout"))

    mock_goto_resp = AsyncMock()
    mock_goto_resp.status = 200
    mock_page.goto = AsyncMock(return_value=mock_goto_resp)
    mock_page.url = "https://hiremetech.com/he-il/jobs-app"
    manager.get_page = AsyncMock(return_value=mock_page)

    result = await manager.check_session_health()

    assert result is True
    mock_page.request.get.assert_awaited_once()
    mock_page.goto.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_session_health_api_200_missing_user_fallback():
    """Test fallback to page.goto navigation when /api/auth/me returns 200 without valid user dict."""
    from job_mcp.core.auth import SessionManager

    manager = SessionManager(user_data_dir="/tmp/test_profile")
    mock_page = AsyncMock()

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"user": None})
    mock_page.request.get = AsyncMock(return_value=mock_resp)

    mock_goto_resp = AsyncMock()
    mock_goto_resp.status = 200
    mock_page.goto = AsyncMock(return_value=mock_goto_resp)
    mock_page.url = "https://hiremetech.com/he-il/jobs-app"
    manager.get_page = AsyncMock(return_value=mock_page)

    result = await manager.check_session_health()

    assert result is True
    mock_page.request.get.assert_awaited_once()
    mock_page.goto.assert_awaited_once()


# ==========================================
# 6. Tests for Tool Integration & Seamless Fallback
# ==========================================


@pytest.mark.asyncio
async def test_warm_cache_api_success():
    """Verify _warm_cache attempts pure HTTP API first and populates cache without browser launch or DOM navigation."""
    from job_mcp.core.api_client import JobCache
    from job_mcp.main import _warm_cache

    sample_jobs = [
        Job(job_id="api-1", title="API Engineer", company="API Corp", tech_stack=["Python"])
    ]

    mock_session = AsyncMock()
    mock_page = AsyncMock()
    mock_session.ensure_ready.return_value = mock_page
    mock_session.get_page.return_value = mock_page

    cache = JobCache(ttl_minutes=15)

    with patch("job_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api, \
         patch("job_mcp.sources.hiremetech.browser_extract_jobs", new_callable=AsyncMock) as mock_dom:
        mock_api.return_value = sample_jobs

        await _warm_cache(mock_session, cache)

        assert len(cache.get_all()) == 1
        assert cache.get_all()[0].job_id == "api-1"
        mock_api.assert_awaited_once_with(None, size=50)
        mock_dom.assert_not_called()
        mock_session.ensure_ready.assert_not_called()
        mock_page.goto.assert_not_called()


@pytest.mark.asyncio
async def test_warm_cache_api_failure_fallback_to_dom():
    """Verify _warm_cache falls back to DOM scraping when API fetch fails."""
    from job_mcp.core.api_client import JobCache
    from job_mcp.main import _warm_cache

    sample_jobs = [
        Job(job_id="dom-1", title="DOM Engineer", company="DOM Corp", tech_stack=["React"])
    ]

    mock_session = AsyncMock()
    mock_page = AsyncMock()
    mock_page.url = "https://hiremetech.com/login"
    mock_session.ensure_ready.return_value = mock_page
    mock_session.get_page.return_value = mock_page

    cache = JobCache(ttl_minutes=15)

    with patch("job_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api, \
         patch("job_mcp.sources.hiremetech.browser_extract_jobs", new_callable=AsyncMock) as mock_dom:
        mock_api.side_effect = RuntimeError("API 500 Error")
        mock_dom.return_value = sample_jobs

        await _warm_cache(mock_session, cache)

        assert len(cache.get_all()) == 1
        assert cache.get_all()[0].job_id == "dom-1"
        assert mock_api.await_count == 2
        mock_dom.assert_awaited_once_with(mock_page)
        mock_page.goto.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_job_matches_api_success():
    """Verify get_job_matches fetches via direct pure HTTP API and skips DOM scraping when API succeeds."""
    from fastmcp import Context
    from job_mcp.core.api_client import JobCache
    from job_mcp.main import get_job_matches
    from job_mcp.sources import JobAggregator, SourceRegistry
    from job_mcp.sources.hiremetech import HireMeTechSource

    sample_jobs = [
        Job(job_id="api-201", title="Cloud Architect", company="CloudCo", tech_stack=["AWS", "Terraform"])
    ]

    cache = JobCache(ttl_minutes=10)
    mock_session = AsyncMock()
    mock_page = AsyncMock()
    mock_session.ensure_ready.return_value = mock_page
    mock_session.get_page.return_value = mock_page

    reg = SourceRegistry()
    reg.register(HireMeTechSource(session_manager=mock_session))
    agg = JobAggregator(registry=reg, cache=cache)

    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {
        "session": mock_session,
        "cache": cache,
        "registry": reg,
        "aggregator": agg,
    }

    with patch("job_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api, \
         patch("job_mcp.sources.hiremetech.browser_extract_jobs", new_callable=AsyncMock) as mock_dom:
        mock_api.return_value = sample_jobs

        res = await get_job_matches(force_refresh=True, ctx=ctx)

        assert res["success"] is True
        assert len(res["data"]) == 1
        assert res["data"][0]["job_id"] == "api-201"
        mock_api.assert_awaited_once_with(None, size=50)
        mock_dom.assert_not_called()


@pytest.mark.asyncio
async def test_get_job_matches_api_failure_fallback_to_dom():
    """Verify get_job_matches seamlessly falls back to DOM scraping when API fetch throws an error."""
    from fastmcp import Context
    from job_mcp.core.api_client import JobCache
    from job_mcp.main import get_job_matches
    from job_mcp.sources import JobAggregator, SourceRegistry
    from job_mcp.sources.hiremetech import HireMeTechSource

    sample_jobs = [
        Job(job_id="dom-301", title="Frontend Specialist", company="UI Corp", tech_stack=["Vue"])
    ]

    cache = JobCache(ttl_minutes=10)
    mock_session = AsyncMock()
    mock_page = AsyncMock()
    mock_page.url = "https://hiremetech.com/login"
    mock_session.ensure_ready.return_value = mock_page
    mock_session.get_page.return_value = mock_page

    reg = SourceRegistry()
    reg.register(HireMeTechSource(session_manager=mock_session))
    agg = JobAggregator(registry=reg, cache=cache)

    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {
        "session": mock_session,
        "cache": cache,
        "registry": reg,
        "aggregator": agg,
    }

    with patch("job_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api, \
         patch("job_mcp.sources.hiremetech.browser_extract_jobs", new_callable=AsyncMock) as mock_dom:
        mock_api.side_effect = RuntimeError("API service unavailable")
        mock_dom.return_value = sample_jobs

        res = await get_job_matches(force_refresh=True, ctx=ctx)

        assert res["success"] is True
        assert len(res["data"]) == 1
        assert res["data"][0]["job_id"] == "dom-301"
        assert mock_api.await_count == 2
        mock_dom.assert_awaited_once_with(mock_page)
        mock_page.goto.assert_awaited_once()


@pytest.mark.asyncio
async def test_filter_jobs_by_preferences_supplements_skills_from_resume_profile():
    """Verify filter_jobs_by_preferences queries online resume profile when cv_path & tech_stack are omitted."""
    from fastmcp import Context
    from job_mcp.core.api_client import JobCache
    from job_mcp.main import filter_jobs_by_preferences

    cache = JobCache(ttl_minutes=10)
    cache.update([
        Job(job_id="job-py", title="Python Backend", company="Alpha", tech_stack=["Python", "FastAPI"]),
        Job(job_id="job-php", title="PHP Maintenance", company="Beta", tech_stack=["PHP"]),
    ])

    mock_session = AsyncMock()
    mock_page = AsyncMock()
    mock_session.ensure_ready.return_value = mock_page
    mock_session.get_page.return_value = mock_page

    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {"session": mock_session, "cache": cache}

    with patch("job_mcp.main.fetch_user_resume_profile", new_callable=AsyncMock) as mock_profile:
        mock_profile.return_value = {"technical_skills": ["Python", "FastAPI"]}

        res = await filter_jobs_by_preferences(ctx=ctx)

        assert res["success"] is True
        assert len(res["data"]) == 2
        # Python job should rank higher due to matched score from online resume profile
        assert res["data"][0]["job_id"] == "job-py"
        assert res["data"][0]["match_score"] > res["data"][1]["match_score"]
        mock_profile.assert_awaited_once_with(mock_page.request)


# ==========================================
# 7. Tests for Explainability & Seniority Detection
# ==========================================


def test_parse_api_job_dict_seniority_and_summary_enrichment():
    """Verify parse_api_job_dict automatically enriches seniority_level and description_summary."""
    raw = {
        "id": "enrich-1",
        "title": "Senior Staff Backend Engineer (Python)",
        "company_name": "DeepScale",
        "description": "We are seeking a seasoned engineer. You will lead system architecture and scalability. Competitive package offered.",
        "skills_required": ["Python", "Docker"],
    }
    job = parse_api_job_dict(raw)
    assert job.seniority_level == "Senior"
    assert job.description_summary is not None
    assert "seasoned engineer" in job.description_summary
    assert len(job.description_summary) <= 250


def test_filter_jobs_explainable_match_reasons():
    """Verify filter_jobs produces structured explainability details."""
    from job_mcp.core.api_client import filter_jobs
    from job_mcp.models.schemas import JobPreferences

    jobs = [
        Job(
            job_id="exp-1",
            title="Junior Cloud Developer",
            company="CloudCo",
            location="Tel Aviv",
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
            description="Build modern cloud APIs. Great environment for entry-level developers.",
            salary_range="25,000 - 30,000 ILS",
        ),
    ]

    prefs = JobPreferences(
        tech_stack=["Python", "FastAPI", "Kubernetes"],
        work_mode=WorkMode.REMOTE,
        location="Tel Aviv",
        min_salary=20000,
    )

    filtered = filter_jobs(jobs, prefs)
    assert len(filtered) == 1
    job = filtered[0]

    assert "Python" in job.matched_skills
    assert "FastAPI" in job.matched_skills
    assert "Kubernetes" in job.missing_skills
    assert job.seniority_level == "Junior"
    assert job.description_summary is not None

    # Verify structured match reasons
    assert any("Target stack matched" in r for r in job.match_reasons)
    assert any("Remote work mode aligned" in r for r in job.match_reasons)
    assert any("Junior level position" in r for r in job.match_reasons)
    assert any("Location aligned" in r for r in job.match_reasons)
    assert any("Salary range meets requirement" in r for r in job.match_reasons)


def test_filter_jobs_senior_exclusion_smart_boundary():
    """Verify smart seniority exclusion checks title vs custom exclusion keywords."""
    from job_mcp.core.api_client import filter_jobs
    from job_mcp.models.schemas import JobPreferences

    jobs = [
        # 1. Junior job mentioning senior in description (should be KEPT)
        Job(
            job_id="keep-1",
            title="Junior Backend Engineer",
            company="StartupCo",
            tech_stack=["Python"],
            description="You will work alongside senior software engineers on scalable microservices.",
        ),
        # 2. Senior title (should be EXCLUDED)
        Job(
            job_id="drop-senior",
            title="Senior Python Architect",
            company="BigTech",
            tech_stack=["Python"],
            description="Drive product architecture.",
        ),
        # 3. Lead title (should be EXCLUDED)
        Job(
            job_id="drop-lead",
            title="Team Lead (Python)",
            company="ScaleCo",
            tech_stack=["Python"],
            description="Manage backend development team.",
        ),
        # 4. Job mentioning JavaScript when Java is excluded (word boundary should NOT exclude JavaScript)
        Job(
            job_id="keep-js",
            title="Fullstack Developer",
            company="WebTech",
            tech_stack=["JavaScript", "React"],
            description="Frontend & fullstack web application development.",
        ),
        # 5. Job mentioning Java when Java is excluded (should be EXCLUDED)
        Job(
            job_id="drop-java",
            title="Backend Developer",
            company="LegacyCo",
            tech_stack=["Java", "Spring"],
            description="Enterprise Java and Spring development.",
        ),
    ]

    prefs = JobPreferences(
        exclude_keywords=["Senior", "Lead", "Java"],
    )

    filtered = filter_jobs(jobs, prefs)
    kept_ids = {j.job_id for j in filtered}

    assert "keep-1" in kept_ids
    assert "keep-js" in kept_ids
    assert "drop-senior" not in kept_ids
    assert "drop-lead" not in kept_ids
    assert "drop-java" not in kept_ids


# ==========================================
# 8. Tests for Pure HTTP & Concurrency Locking (Task 2)
# ==========================================


@pytest.mark.asyncio
async def test_fetch_jobs_via_api_httpx_direct_pure_http():
    """Verify fetch_jobs_via_api with request_context=None makes pure HTTP request via httpx."""
    import httpx

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={
        "jobs": [
            {"id": "http-1", "title": "Pure HTTP Engineer", "company": "NoBrowserCo"}
        ]
    })

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp

        jobs = await fetch_jobs_via_api(None, page=1, size=25)

        assert len(jobs) == 1
        assert jobs[0].job_id == "http-1"
        assert jobs[0].title == "Pure HTTP Engineer"
        mock_get.assert_awaited_once()
        called_url = mock_get.call_args[0][0]
        assert "/api/jobs/search" in called_url
        assert "size=25" in called_url


@pytest.mark.asyncio
async def test_fetch_jobs_via_api_httpx_client_instance():
    """Verify fetch_jobs_via_api accepts httpx.AsyncClient instance directly."""
    import httpx
    from job_mcp.core.api_client import DEFAULT_HTTP_HEADERS

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={
        "jobs": [
            {"id": "client-1", "title": "Client Injected Job", "company": "InjectedCo"}
        ]
    })
    mock_client.get = AsyncMock(return_value=mock_resp)

    jobs = await fetch_jobs_via_api(client=mock_client, page=2, size=15)

    assert len(jobs) == 1
    assert jobs[0].job_id == "client-1"
    mock_client.get.assert_awaited_once()
    called_url = mock_client.get.call_args[0][0]
    assert "page=2" in called_url
    assert "size=15" in called_url
    assert mock_client.get.call_args[1]["headers"] == DEFAULT_HTTP_HEADERS


@pytest.mark.asyncio
async def test_hiremetech_source_pure_http_no_browser():
    """Verify HireMeTechSource.fetch_jobs performs pure HTTP request without touching SessionManager."""
    from job_mcp.sources.hiremetech import HireMeTechSource

    mock_session = AsyncMock()
    source = HireMeTechSource(session_manager=mock_session)

    sample_jobs = [
        Job(job_id="hmt-http-1", title="Fast Backend", company="FastCo", tech_stack=["Python"])
    ]

    with patch("job_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = sample_jobs

        jobs = await source.fetch_jobs(limit=10)

        assert len(jobs) == 1
        assert jobs[0].job_id == "hmt-http-1"
        assert jobs[0].source == "hiremetech"
        mock_api.assert_awaited_once_with(None, size=10)
        mock_session.ensure_ready.assert_not_called()
        mock_session.get_page.assert_not_called()


@pytest.mark.asyncio
async def test_session_manager_ensure_ready_concurrency():
    """Verify concurrent ensure_ready() calls are serialized with _lock and don't double initialize."""
    import asyncio
    from job_mcp.core.auth import SessionManager

    mock_pw = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://hiremetech.com/he-il/jobs-app"
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"user": {"id": "u1"}})
    mock_page.request.get = AsyncMock(return_value=mock_resp)
    mock_context.pages = [mock_page]
    mock_pw.chromium.launch_persistent_context.return_value = mock_context

    mock_cm = MagicMock()
    mock_cm.start = AsyncMock(return_value=mock_pw)

    with patch("job_mcp.core.auth.async_playwright", return_value=mock_cm):
        manager = SessionManager(user_data_dir="/tmp/test_profile_concurrency_ready")
        assert not manager.is_running

        results = await asyncio.gather(
            manager.ensure_ready(max_retries=1),
            manager.ensure_ready(max_retries=1),
            manager.ensure_ready(max_retries=1),
        )

        assert len(results) == 3
        for p in results:
            assert p == mock_page
        assert manager.is_running
        mock_cm.start.assert_awaited_once()
        mock_pw.chromium.launch_persistent_context.assert_awaited_once()

        await manager.shutdown()
        assert not manager.is_running



