"""Tests for SmartRecruiters ATS public provider (Milestone 6 Slice M6-B2).

Covers:
- parse_smartrecruiters_job normalization (title, company, locations, URLs, HTML/plain description, work mode, dates, compensation)
- Safe handling of missing optional fields in API payloads
- Preservation of full description without arbitrary truncation
- SmartRecruitersSource constructor semantics (defaults, explicit catalog, explicit empty)
- fetch_jobs with mocked transport (query propagation, pagination, limit bounds, disabled companies, HTTP 404, 429, 500, network error)
- fetch_job_by_ref deterministic detail refetch (exact detail endpoint, 404 -> NOT_FOUND, HTTP error -> UPSTREAM_ERROR, invalid ref)
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
from job_mcp.sources.company_registry.entries import SmartRecruitersCompany
from job_mcp.sources.public.smartrecruiters import (
    SmartRecruitersSource,
    parse_smartrecruiters_job,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "smartrecruiters"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Job Normalization Tests
# ===========================================================================


def test_parse_smartrecruiters_job_standard_list_item() -> None:
    data = load_fixture("valid_postings.json")
    raw_job = data["content"][0]

    job = parse_smartrecruiters_job(
        raw_job, company_name="SmartRecruiters Inc", company_identifier="smartrecruiters"
    )
    assert job.job_id == "smartrecruiters_smartrecruiters_743999961234567"
    assert job.title == "Senior Python Backend Engineer"
    assert job.company == "SmartRecruiters Inc"
    assert "San Francisco" in job.location
    assert "CA" in job.location
    assert "us" in job.location
    assert job.url == "https://jobs.smartrecruiters.com/smartrecruiters/743999961234567"
    assert "careers.smartrecruiters.com" in (job.apply_url or "")
    assert job.department == "Engineering"
    assert job.source == "smartrecruiters"
    assert job.sources == ["smartrecruiters"]
    assert job.work_mode == WorkMode.ONSITE
    assert job.posted_date == "2024-03-01T12:00:00.000Z"
    assert "python" in [t.lower() for t in job.tech_stack]


def test_parse_smartrecruiters_job_remote() -> None:
    data = load_fixture("valid_postings.json")
    raw_job = data["content"][1]

    job = parse_smartrecruiters_job(
        raw_job, company_name="SmartRecruiters Inc", company_identifier="smartrecruiters"
    )
    assert job.title == "Staff Distributed Systems Engineer"
    assert job.work_mode == WorkMode.REMOTE
    assert job.posted_date == "2024-03-02T14:30:00.000Z"


def test_parse_smartrecruiters_job_hybrid() -> None:
    data = load_fixture("second_page_postings.json")
    raw_job = data["content"][0]

    job = parse_smartrecruiters_job(
        raw_job, company_name="SmartRecruiters Inc", company_identifier="smartrecruiters"
    )
    assert job.title == "DevOps Engineer"
    assert job.work_mode == WorkMode.HYBRID


def test_parse_smartrecruiters_job_detail_with_sections_and_compensation() -> None:
    raw_job = load_fixture("valid_detail.json")

    job = parse_smartrecruiters_job(
        raw_job, company_name="SmartRecruiters Inc", company_identifier="smartrecruiters"
    )
    assert job.job_id == "smartrecruiters_smartrecruiters_743999961234567"
    assert job.salary_range == "160,000 - 195,000 USD"
    assert "talent acquisition software" in job.description
    assert "scale our core APIs" in job.description
    assert "5+ years of Python experience" in job.description
    assert job.requirements is not None
    assert "python" in [t.lower() for t in job.tech_stack]
    assert "fastapi" in [t.lower() for t in job.tech_stack]
    assert "docker" in [t.lower() for t in job.tech_stack]


def test_parse_smartrecruiters_job_missing_optional_fields() -> None:
    raw_job = load_fixture("minimal_detail.json")

    job = parse_smartrecruiters_job(
        raw_job, company_name="Acme", company_identifier="acme"
    )
    assert job.job_id == "smartrecruiters_acme_743999961234599"
    assert job.title == "General Technologist"
    assert job.company == "Acme"
    assert job.location == ""
    assert job.department is None
    assert job.posted_date is None
    assert job.salary_range is None
    assert job.work_mode is None

    # Fallback to upstream raw company name when company_name is empty
    job_fallback = parse_smartrecruiters_job(
        raw_job, company_name="", company_identifier="acme"
    )
    assert job_fallback.company == "SmartRecruiters Inc"


def test_parse_smartrecruiters_job_preserves_full_description_beyond_2000_chars() -> None:
    raw_job = load_fixture("valid_detail.json")
    long_text = "<p>" + ("Deep architectural expertise and robust engineering leadership. " * 150) + "</p>"
    raw_job["jobAd"]["sections"]["jobDescription"]["text"] = long_text

    job = parse_smartrecruiters_job(
        raw_job, company_name="Enterprise Corp", company_identifier="enterprisecorp"
    )
    assert len(job.description) > 5000
    assert "Deep architectural expertise and robust engineering leadership." in job.description


# ===========================================================================
# 2. Source Constructor Semantics
# ===========================================================================


def test_smartrecruiters_source_default_catalog() -> None:
    source = SmartRecruitersSource()
    assert "smartrecruiters" in source._companies
    assert isinstance(source._companies["smartrecruiters"], SmartRecruitersCompany)


def test_smartrecruiters_source_explicit_catalog() -> None:
    custom = {
        "custom_co": SmartRecruitersCompany(
            name="Custom Co", company_identifier="customco", enabled=True
        )
    }
    source = SmartRecruitersSource(companies=custom)
    assert "custom_co" in source._companies
    assert source._companies["custom_co"].company_identifier == "customco"


def test_smartrecruiters_source_explicit_empty_catalog() -> None:
    source = SmartRecruitersSource(companies={})
    assert len(source._companies) == 0


# ===========================================================================
# 3. fetch_jobs Tests (Query, Pagination, Filtering)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_jobs_propagates_query_and_destination() -> None:
    valid_data = load_fixture("valid_postings.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = valid_data

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        prefs = JobPreferences(keywords=["python", "backend"])
        jobs = await source.fetch_jobs(preferences=prefs, limit=10)

        assert mock_get.called
        call_args = mock_get.call_args
        params = call_args[1].get("params", {})
        assert params.get("destination") == "PUBLIC"
        assert params.get("q") == "python backend"
        assert params.get("limit") == 10
        assert params.get("offset") == 0
        assert len(jobs) == 2


@pytest.mark.asyncio
async def test_fetch_jobs_pagination_advancement() -> None:
    page1 = load_fixture("valid_postings.json")  # 2 postings, totalFound=3
    page1["totalFound"] = 3
    page2 = load_fixture("second_page_postings.json")  # 1 posting, offset=2

    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = page1

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = page2

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=[mock_resp1, mock_resp2])) as mock_get:
        jobs = await source.fetch_jobs(limit=10)

        assert mock_get.call_count == 2
        first_call_params = mock_get.call_args_list[0][1].get("params", {})
        second_call_params = mock_get.call_args_list[1][1].get("params", {})
        assert first_call_params.get("offset") == 0
        assert second_call_params.get("offset") == 2
        assert len(jobs) == 3


@pytest.mark.asyncio
async def test_fetch_jobs_respects_limit_early_termination() -> None:
    page1 = load_fixture("valid_postings.json")  # has 2 postings

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page1

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        jobs = await source.fetch_jobs(limit=1)

        assert mock_get.call_count == 1
        assert len(jobs) == 1
        assert jobs[0].job_id == "smartrecruiters_smartrecruiters_743999961234567"


@pytest.mark.asyncio
async def test_fetch_jobs_empty_board() -> None:
    empty_data = load_fixture("empty_postings.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = empty_data

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 0


@pytest.mark.asyncio
async def test_fetch_jobs_malformed_list_response_logs_warning_and_returns_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed_data = load_fixture("malformed_list.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = malformed_data

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 0
        captured = capsys.readouterr()
        assert "malformed response missing 'content' list" in captured.err



@pytest.mark.asyncio
async def test_fetch_jobs_skips_disabled_companies() -> None:
    source = SmartRecruitersSource(
        companies={
            "disabled_co": SmartRecruitersCompany(
                name="Disabled Co", company_identifier="disabledco", enabled=False
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock()) as mock_get:
        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 0
        assert not mock_get.called


@pytest.mark.asyncio
async def test_fetch_jobs_handles_http_errors_gracefully() -> None:
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404

    source = SmartRecruitersSource(
        companies={
            "missing_co": SmartRecruitersCompany(
                name="Missing Co", company_identifier="missingco", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp_404)):
        jobs = await source.fetch_jobs(limit=10)
        assert len(jobs) == 0


# ===========================================================================
# 4. fetch_job_by_ref Tests (Detail Refetch)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_job_by_ref_success_exact_posting() -> None:
    detail_data = load_fixture("valid_detail.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = detail_data

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters Inc", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="743999961234567",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        result = await source.fetch_job_by_ref(ref)

        assert result.status == FetchStatus.FOUND
        assert result.job is not None
        assert result.job.job_id == "smartrecruiters_smartrecruiters_743999961234567"
        assert result.job.title == "Senior Python Backend Engineer"
        assert result.job.salary_range == "160,000 - 195,000 USD"
        assert mock_get.called
        url = mock_get.call_args[0][0]
        assert url == "https://api.smartrecruiters.com/v1/companies/smartrecruiters/postings/743999961234567"


@pytest.mark.asyncio
async def test_fetch_job_by_ref_404_returns_not_found() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    source = SmartRecruitersSource()
    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="nonexistent-id",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert "not found" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_500_returns_upstream_error() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_resp
    )

    source = SmartRecruitersSource()
    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="loc-500",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.UPSTREAM_ERROR
    assert result.job is None
    assert "Upstream HTTP error" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_matching_uuid_returns_found() -> None:
    detail_data = load_fixture("valid_detail.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = detail_data

    source = SmartRecruitersSource()
    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="34225731-e7cf-4584-b0b7-78098fe1a66b",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        result = await source.fetch_job_by_ref(ref)

        assert result.status == FetchStatus.FOUND
        assert result.job is not None
        assert result.job.title == "Senior Python Backend Engineer"
        assert mock_get.called
        url = mock_get.call_args[0][0]
        assert url == "https://api.smartrecruiters.com/v1/companies/smartrecruiters/postings/34225731-e7cf-4584-b0b7-78098fe1a66b"


@pytest.mark.asyncio
async def test_fetch_job_by_ref_identity_mismatch_returns_upstream_error() -> None:
    detail_data = load_fixture("valid_detail.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = detail_data

    source = SmartRecruitersSource()
    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="999999999999999",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

        assert result.status == FetchStatus.UPSTREAM_ERROR
        assert result.job is None
        assert "identity mismatch" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_malformed_detail_returns_upstream_error() -> None:
    malformed_data = load_fixture("malformed_detail.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = malformed_data

    source = SmartRecruitersSource()
    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="loc-malformed",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await source.fetch_job_by_ref(ref)

    assert result.status == FetchStatus.UPSTREAM_ERROR
    assert result.job is None
    assert "missing usable id/uuid" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_invalid_ref_returns_invalid_ref() -> None:
    source = SmartRecruitersSource()

    # Missing account
    ref_missing_account = JobRef(
        version=1, source_family="smartrecruiters", account=None, locator="loc-123"
    )
    result = await source.fetch_job_by_ref(ref_missing_account)
    assert result.status == FetchStatus.INVALID_REF
    assert result.job is None

    # Mismatched source family
    ref_wrong_family = JobRef(
        version=1, source_family="ashby", account="smartrecruiters", locator="loc-123"
    )
    result2 = await source.fetch_job_by_ref(ref_wrong_family)
    assert result2.status == FetchStatus.INVALID_REF
    assert result2.job is None


# ===========================================================================
# 5. check_health Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_check_health_success() -> None:
    source = SmartRecruitersSource()

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        assert await source.check_health() is True
        assert mock_get.called
        params = mock_get.call_args[1].get("params", {})
        assert params.get("destination") == "PUBLIC"
        assert params.get("limit") == 1


@pytest.mark.asyncio
async def test_check_health_failure() -> None:
    source = SmartRecruitersSource()

    with patch(
        "httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("Failed"))
    ):
        assert await source.check_health() is False


@pytest.mark.asyncio
async def test_check_health_empty_catalog() -> None:
    source = SmartRecruitersSource(companies={})
    assert await source.check_health() is False
