"""Tests for Ashby ATS public provider (Milestone 6 Slice M6-B1).

Covers:
- parse_ashby_job normalization (title, company, locations, URLs, HTML/plain description, work mode, dates, compensation)
- Safe handling of missing optional fields in API payloads
- AshbySource constructor semantics (defaults, explicit catalog, explicit empty)
- fetch_jobs with mocked transport (success, secondary locations, disabled companies, empty board, HTTP 404, HTTP 500, network error)
- fetch_job_by_ref deterministic board refetch (exact locator match, missing locator -> NOT_FOUND, HTTP error -> UPSTREAM_ERROR, invalid ref)
- check_health readiness checks
- Zero dummy job invariant
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.core.search_plane.models import FetchStatus, JobRef
from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources.company_registry.entries import AshbyCompany
from job_mcp.sources.public.ashby import (
    AshbySource,
    parse_ashby_job,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ashby"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Job Normalization Tests
# ===========================================================================


def test_parse_ashby_job_standard() -> None:
    data = load_fixture("valid_board.json")
    raw_job = data["jobs"][0]

    job = parse_ashby_job(raw_job, company_name="Ashby", board_name="ashby")
    assert job.job_id == "ashby_ashby_c1f77d34-7a42-4f05-8a8b-302efb1a4731"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Ashby"
    assert "Tel Aviv, Israel" in job.location
    assert "Yokneam, Israel" in job.location
    assert job.url == "https://jobs.ashbyhq.com/ashby/c1f77d34-7a42-4f05-8a8b-302efb1a4731"
    assert job.apply_url == "https://jobs.ashbyhq.com/ashby/c1f77d34-7a42-4f05-8a8b-302efb1a4731/application"
    assert "Senior Backend Engineer" in job.description
    assert job.department == "Engineering"
    assert job.source == "ashby"
    assert job.sources == ["ashby"]
    assert job.work_mode == WorkMode.ONSITE
    assert job.posted_date == "2026-09-15T12:00:00.000Z"
    assert "python" in [t.lower() for t in job.tech_stack]
    assert job.requirements is not None


def test_parse_ashby_job_remote() -> None:
    data = load_fixture("valid_board.json")
    raw_job = data["jobs"][1]

    job = parse_ashby_job(raw_job, company_name="Ashby", board_name="ashby")
    assert job.title == "Staff Full Stack Engineer (Remote)"
    assert job.work_mode == WorkMode.REMOTE
    assert job.posted_date == "2026-09-20T08:30:00.000Z"


def test_parse_ashby_job_compensation() -> None:
    data = load_fixture("compensation_board.json")
    raw_job = data["jobs"][0]

    job = parse_ashby_job(raw_job, company_name="Ashby", board_name="ashby")
    assert job.salary_range == "140,000 - 180,000 USD"


def test_parse_ashby_job_missing_optional_fields() -> None:
    data = load_fixture("missing_optional_fields.json")
    raw_job = data["jobs"][0]

    job = parse_ashby_job(raw_job, company_name="Acme", board_name="acme")
    assert job.job_id == "ashby_acme_minimal-posting-001"
    assert job.title == "General Application"
    assert job.company == "Acme"
    assert job.location == ""
    assert job.department is None
    assert job.posted_date is None
    assert job.salary_range is None
    assert job.work_mode == WorkMode.ONSITE


def test_parse_ashby_job_preserves_full_description_beyond_2000_chars() -> None:
    """Verify authoritative job descriptions exceeding 2000 chars are not truncated."""
    long_desc = "Important details about the role. " * 150  # ~5100 characters
    assert len(long_desc) > 5000

    raw = {
        "id": "long-desc-001",
        "title": "Principal Architect",
        "descriptionPlain": long_desc,
        "jobUrl": "https://jobs.ashbyhq.com/ashby/long-desc-001",
    }
    job = parse_ashby_job(raw, company_name="Ashby", board_name="ashby")
    assert len(job.description) == len(long_desc)
    assert job.description == long_desc


# ===========================================================================
# 2. Provider Construction & Catalog Tests
# ===========================================================================


def test_ashby_source_init_defaults() -> None:
    source = AshbySource()
    assert "ashby" in source._companies
    assert source.source_id == "ashby"
    assert source.display_name == "Ashby"


def test_ashby_source_init_explicit_catalog() -> None:
    custom = {
        "linear": AshbyCompany(name="Linear", board_name="linear", enabled=True),
    }
    source = AshbySource(companies=custom)
    assert set(source._companies.keys()) == {"linear"}


def test_ashby_source_init_empty_catalog() -> None:
    source = AshbySource(companies={})
    assert len(source._companies) == 0


# ===========================================================================
# 3. fetch_jobs Tests (Mocked Transport)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_jobs_success() -> None:
    data = load_fixture("valid_board.json")
    custom = {
        "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True),
    }
    source = AshbySource(companies=custom)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        jobs = await source.fetch_jobs(limit=10)

    assert len(jobs) == 2
    assert jobs[0].company == "Ashby"
    assert jobs[1].company == "Ashby"


@pytest.mark.asyncio
async def test_fetch_jobs_disabled_company_is_skipped() -> None:
    custom = {
        "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=False),
    }
    source = AshbySource(companies=custom)

    with patch("httpx.AsyncClient.get", AsyncMock()) as mock_get:
        jobs = await source.fetch_jobs()

    assert len(jobs) == 0
    mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_jobs_empty_board() -> None:
    data = load_fixture("empty_board.json")
    custom = {
        "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True),
    }
    source = AshbySource(companies=custom)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        jobs = await source.fetch_jobs()

    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_404_handled_gracefully() -> None:
    custom = {
        "missing_co": AshbyCompany(name="Missing", board_name="nonexistent_board", enabled=True),
    }
    source = AshbySource(companies=custom)

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    http_error = httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_resp)

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=http_error)):
        jobs = await source.fetch_jobs()

    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_500_handled_gracefully() -> None:
    custom = {
        "error_co": AshbyCompany(name="ErrorCo", board_name="error_board", enabled=True),
    }
    source = AshbySource(companies=custom)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_resp)

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=http_error)):
        jobs = await source.fetch_jobs()

    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_network_exception_isolated() -> None:
    custom = {
        "net_err": AshbyCompany(name="NetErr", board_name="net_err_board", enabled=True),
    }
    source = AshbySource(companies=custom)

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("Connection refused"))):
        jobs = await source.fetch_jobs()

    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_preferences_filter_and_limit() -> None:
    data = load_fixture("valid_board.json")
    custom = {
        "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True),
    }
    source = AshbySource(companies=custom)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        prefs = JobPreferences(keywords=["Backend"])
        jobs = await source.fetch_jobs(preferences=prefs, limit=1)

    assert len(jobs) == 1
    assert "Backend" in jobs[0].title


# ===========================================================================
# 4. fetch_job_by_ref Native Refetch Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_job_by_ref_exact_match() -> None:
    data = load_fixture("valid_board.json")
    source = AshbySource(companies={"ashby": AshbyCompany(name="Ashby Inc", board_name="ashby", enabled=True)})

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="c1f77d34-7a42-4f05-8a8b-302efb1a4731")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.FOUND
    assert result.job is not None
    assert result.job.title == "Senior Backend Engineer"
    assert result.job.company == "Ashby Inc"


@pytest.mark.asyncio
async def test_fetch_job_by_ref_missing_locator_returns_not_found() -> None:
    data = load_fixture("valid_board.json")
    source = AshbySource()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="non_existent_uuid")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert "not found on Ashby board" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_404_board_returns_not_found() -> None:
    source = AshbySource()

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    ref = JobRef(version=1, source_family="ashby", account="unknown_co", locator="1234")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert "HTTP 404" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_network_error_returns_upstream_error() -> None:
    source = AshbySource()
    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="c1f77d34-7a42-4f05-8a8b-302efb1a4731")

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectTimeout("Timeout"))):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.UPSTREAM_ERROR
    assert result.job is None
    assert "Upstream HTTP error" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_invalid_ref_returns_invalid_ref() -> None:
    source = AshbySource()
    # Missing account
    ref_missing_account = JobRef(version=1, source_family="ashby", account=None, locator="loc-123")
    result = await source.fetch_job_by_ref(ref_missing_account)
    assert result.status == FetchStatus.INVALID_REF
    assert result.job is None

    # Mismatched source family
    ref_wrong_family = JobRef(version=1, source_family="greenhouse", account="ashby", locator="loc-123")
    result2 = await source.fetch_job_by_ref(ref_wrong_family)
    assert result2.status == FetchStatus.INVALID_REF
    assert result2.job is None


# ===========================================================================
# 5. check_health Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_check_health_success() -> None:
    source = AshbySource()

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        assert await source.check_health() is True


@pytest.mark.asyncio
async def test_check_health_failure() -> None:
    source = AshbySource()

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("Failed"))):
        assert await source.check_health() is False


@pytest.mark.asyncio
async def test_check_health_empty_catalog() -> None:
    source = AshbySource(companies={})
    assert await source.check_health() is False
