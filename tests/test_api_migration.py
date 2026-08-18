"""Comprehensive tests for direct API client and payload mapping."""

import pytest
from unittest.mock import AsyncMock

from hireme_mcp.core.api_client import (
    fetch_jobs_via_api,
    fetch_saved_jobs_batch,
    fetch_user_resume_profile,
    parse_api_job_dict,
)
from hireme_mcp.models.schemas import Job, WorkMode


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
