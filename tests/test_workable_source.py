"""Tests for Workable ATS public provider (Milestone 6 Slice M6-B3).

Covers:
- parse_workable_job normalization (title, company, locations, URLs, HTML/plain description, work mode, dates, seniority)
- Safe handling of missing optional fields in API payloads
- Preservation of full description without arbitrary truncation (>2000 chars)
- WorkableSource constructor semantics (defaults, explicit catalog, explicit empty)
- fetch_jobs with mocked transport (limit bounds, filtering, error isolation, disabled companies)
- fetch_job_by_ref deterministic account re-query + exact shortcode match (FOUND, NOT_FOUND, UPSTREAM_ERROR, INVALID_REF)
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
from job_mcp.sources.company_registry.entries import WorkableCompany
from job_mcp.sources.public.workable import (
    WorkableSource,
    _detect_work_mode,
    parse_workable_job,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "workable"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Job Normalization Tests
# ===========================================================================


def test_parse_workable_job_standard() -> None:
    data = load_fixture("valid_account.json")
    raw_job = data["jobs"][0]

    job = parse_workable_job(
        raw_job, company_name="Hugging Face", account_subdomain="huggingface"
    )
    assert job.job_id == "workable_huggingface_F4C096B22E"
    assert job.title == "Senior Machine Learning Engineer"
    assert job.company == "Hugging Face"
    assert "New York" in job.location
    assert "NY" in job.location
    assert "United States" in job.location
    assert job.url == "https://apply.workable.com/huggingface/j/F4C096B22E/"
    assert job.apply_url == "https://apply.workable.com/huggingface/j/F4C096B22E/apply/"
    assert job.department == "Engineering"
    assert job.seniority_level == "Senior level"
    assert job.source == "workable"
    assert job.sources == ["workable"]
    assert job.work_mode == WorkMode.REMOTE
    assert job.posted_date == "2024-03-01"
    assert "transformer" in job.description.lower()
    assert any("python" in t.lower() for t in job.tech_stack)
    assert job.requirements is not None
    assert job.responsibilities is not None


def test_parse_workable_job_work_modes() -> None:
    data = load_fixture("multiple_jobs.json")
    jobs = data["jobs"]

    # Job 0: telecommuting=True -> REMOTE
    job0 = parse_workable_job(jobs[0], company_name="Hugging Face", account_subdomain="huggingface")
    assert job0.work_mode == WorkMode.REMOTE

    # Job 1: telecommuting=False, Paris location, no remote text -> ONSITE
    job1 = parse_workable_job(jobs[1], company_name="Hugging Face", account_subdomain="huggingface")
    assert job1.work_mode == WorkMode.ONSITE
    assert "Paris" in job1.location
    assert job1.seniority_level == "Mid level"

    # Job 2: telecommuting=False, London location, text has "hybrid" -> HYBRID
    job2 = parse_workable_job(jobs[2], company_name="Hugging Face", account_subdomain="huggingface")
    assert job2.work_mode == WorkMode.HYBRID
    assert "London" in job2.location


def test_detect_work_mode_truthful_fallback() -> None:
    # telecommuting flag takes precedence
    assert _detect_work_mode(True, "New York", "onsite role") == WorkMode.REMOTE

    # Textual indicators
    assert _detect_work_mode(False, "Paris", "This is a remote position") == WorkMode.REMOTE
    assert _detect_work_mode(False, "Berlin", "Flexible hybrid policy") == WorkMode.HYBRID
    assert _detect_work_mode(False, "London", "On-site collaboration expected") == WorkMode.ONSITE

    # Location present without remote/hybrid keywords -> ONSITE
    assert _detect_work_mode(False, "Tokyo, Japan", "Develop software") == WorkMode.ONSITE

    # Zero evidence -> None (truthful, do not manufacture ONSITE)
    assert _detect_work_mode(False, "", "Develop software in an agile team") is None


def test_parse_workable_job_full_description_unclamped() -> None:
    data = load_fixture("full_description.json")
    raw_job = data["jobs"][0]

    job = parse_workable_job(
        raw_job, company_name="Hugging Face", account_subdomain="huggingface"
    )
    assert len(job.description) > 2000
    assert "Section 1: Distributed training across thousands of GPUs" in job.description
    assert "Continuous learning and competitive compensation" in job.description


def test_parse_workable_job_minimal_fields() -> None:
    data = load_fixture("minimal_job.json")
    raw_job = data["jobs"][0]

    job = parse_workable_job(
        raw_job, company_name="Minimal Co", account_subdomain="minimal"
    )
    assert job.job_id == "workable_minimal_MIN1234567"
    assert job.title == "Junior Developer"
    assert job.company == "Minimal Co"
    assert job.location == ""
    assert job.work_mode is None
    assert job.department is None
    assert job.seniority_level is None
    assert job.posted_date is None
    assert job.salary_range is None


# ===========================================================================
# 2. Source Constructor Tests
# ===========================================================================


def test_workable_source_defaults() -> None:
    src = WorkableSource()
    assert src.source_id == "workable"
    assert "huggingface" in src._companies


def test_workable_source_custom_catalog() -> None:
    custom = {
        "acme": WorkableCompany(name="Acme Inc", account_subdomain="acme", enabled=True),
        "disabled": WorkableCompany(name="Disabled Co", account_subdomain="disabled", enabled=False),
    }
    src = WorkableSource(companies=custom)
    assert len(src._companies) == 2
    assert src._companies["acme"].account_subdomain == "acme"


def test_workable_source_empty_catalog() -> None:
    src = WorkableSource(companies={})
    assert len(src._companies) == 0


# ===========================================================================
# 3. fetch_jobs Tests with Mocked Transport
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_jobs_success() -> None:
    data = load_fixture("multiple_jobs.json")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    src = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        jobs = await src.fetch_jobs(limit=10)

    assert len(jobs) == 3
    assert jobs[0].job_id == "workable_huggingface_F4C096B22E"
    assert jobs[1].job_id == "workable_huggingface_A1B2C3D4E5"
    assert jobs[2].job_id == "workable_huggingface_9Z8Y7X6W5V"


@pytest.mark.asyncio
async def test_fetch_jobs_respects_limit() -> None:
    data = load_fixture("multiple_jobs.json")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    src = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            )
        }
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        jobs = await src.fetch_jobs(limit=2)

    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_fetch_jobs_preferences_filtering() -> None:
    data = load_fixture("multiple_jobs.json")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()

    src = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            )
        }
    )

    prefs = JobPreferences(work_mode=WorkMode.REMOTE)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        jobs = await src.fetch_jobs(preferences=prefs, limit=10)

    assert len(jobs) == 1
    assert jobs[0].work_mode == WorkMode.REMOTE
    assert jobs[0].title == "Senior Machine Learning Engineer"


@pytest.mark.asyncio
async def test_fetch_jobs_no_enabled_companies() -> None:
    src = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=False
            )
        }
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        jobs = await src.fetch_jobs(limit=10)
        assert mock_get.call_count == 0
    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_error_isolation() -> None:
    success_data = load_fixture("valid_account.json")
    resp_success = MagicMock(status_code=200, json=lambda: success_data, raise_for_status=MagicMock())

    resp_404 = MagicMock(status_code=404)
    resp_404.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=resp_404
    )

    src = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            ),
            "deadco": WorkableCompany(
                name="Dead Co", account_subdomain="deadco", enabled=True
            ),
        }
    )

    async def side_effect(url: str, **kwargs: Any) -> MagicMock:
        if "huggingface" in url:
            return resp_success
        raise httpx.HTTPStatusError("Not Found", request=MagicMock(), response=resp_404)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        jobs = await src.fetch_jobs(limit=10)

    # Success company returned its jobs despite deadco failing
    assert len(jobs) == 1
    assert jobs[0].company == "Hugging Face"


@pytest.mark.asyncio
async def test_fetch_jobs_malformed_response_handled_gracefully() -> None:
    malformed_data = load_fixture("malformed_account.json")
    resp_malformed = MagicMock(status_code=200, json=lambda: malformed_data, raise_for_status=MagicMock())

    src = WorkableSource(
        companies={
            "bad": WorkableCompany(name="Bad", account_subdomain="bad", enabled=True)
        }
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = resp_malformed
        jobs = await src.fetch_jobs(limit=10)

    assert jobs == []


# ===========================================================================
# 4. fetch_job_by_ref Tests (Native Refetch Seam)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_job_by_ref_found() -> None:
    data = load_fixture("multiple_jobs.json")
    mock_resp = MagicMock(status_code=200, json=lambda: data, raise_for_status=MagicMock())

    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="A1B2C3D4E5")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.FOUND
    assert res.job is not None
    assert res.job.job_id == "workable_huggingface_A1B2C3D4E5"
    assert res.job.title == "Frontend Engineer"


@pytest.mark.asyncio
async def test_fetch_job_by_ref_not_found_in_account() -> None:
    data = load_fixture("multiple_jobs.json")
    mock_resp = MagicMock(status_code=200, json=lambda: data, raise_for_status=MagicMock())

    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="NONEXISTENT")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.NOT_FOUND
    assert res.job is None
    assert "NONEXISTENT" in (res.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_account_404() -> None:
    resp_404 = MagicMock(status_code=404)
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="unknownaccount", locator="ABC")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = resp_404
        res = await src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.NOT_FOUND
    assert res.job is None
    assert "unknownaccount" in (res.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_job_by_ref_http_500() -> None:
    resp_500 = MagicMock(status_code=500)
    resp_500.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=resp_500
    )
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="F4C096B22E")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = resp_500
        res = await src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.UPSTREAM_ERROR
    assert res.job is None


@pytest.mark.asyncio
async def test_fetch_job_by_ref_malformed_response() -> None:
    mock_resp = MagicMock(status_code=200, json=lambda: "not-a-dict", raise_for_status=MagicMock())
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="F4C096B22E")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.UPSTREAM_ERROR
    assert res.job is None
    assert "malformed" in (res.diagnostic or "").lower()


@pytest.mark.asyncio
async def test_fetch_job_by_ref_invalid_ref() -> None:
    src = WorkableSource()

    # Wrong source family
    ref_wrong = JobRef(version=1, source_family="ashby", account="huggingface", locator="F4C096B22E")
    res1 = await src.fetch_job_by_ref(ref_wrong)
    assert res1.status == FetchStatus.INVALID_REF

    # Missing account
    ref_no_acc = JobRef(version=1, source_family="workable", account=None, locator="F4C096B22E")
    res2 = await src.fetch_job_by_ref(ref_no_acc)
    assert res2.status == FetchStatus.INVALID_REF

    # Missing locator
    ref_no_loc = JobRef.model_construct(version=1, source_family="workable", account="huggingface", locator="")
    res3 = await src.fetch_job_by_ref(ref_no_loc)
    assert res3.status == FetchStatus.INVALID_REF


# ===========================================================================
# 5. check_health Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_check_health_success() -> None:
    mock_resp = MagicMock(status_code=200)
    src = WorkableSource()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        assert await src.check_health() is True


@pytest.mark.asyncio
async def test_check_health_failure() -> None:
    mock_resp = MagicMock(status_code=500)
    src = WorkableSource()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        assert await src.check_health() is False


@pytest.mark.asyncio
async def test_check_health_no_enabled_companies() -> None:
    src = WorkableSource(
        companies={
            "hf": WorkableCompany(name="HF", account_subdomain="hf", enabled=False)
        }
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        assert await src.check_health() is False
        assert mock_get.call_count == 0
