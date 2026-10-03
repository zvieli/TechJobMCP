"""Tests for Workable ATS integration with the Unified Search Plane (Milestone 6-B3).

Verifies:
- Workable source capability declarations (supports_search=True, supports_native_fetch=True, supports_query=False, supports_company_filter=True, supports_work_mode=False, supports_pagination=False)
- JobRef generation and deterministic encode/decode round-trip for Workable postings
- Safe extraction of account_subdomain and shortcode via rsplit('_', 1)
- SearchPlaneAdapter.search() retrieval, normalization, limit, and company filtering
- SearchPlaneAdapter.fetch() cache hits and native refetch dispatch via generic seam
- Negative cases: 404 -> NOT_FOUND (never synthetic dummy Job), HTTP 500, invalid ref
- Configuration-only onboarding of new Workable companies without Python code edits
- 15 mutation proofs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

import httpx
import pytest

from job_mcp.core.api_client import JobCache
from job_mcp.core.search_plane.adapter import SOURCE_CAPABILITY_MAP, SearchPlaneAdapter
from job_mcp.core.search_plane.models import (
    FetchStatus,
    JobRef,
    JobSearchRequest,
    SourceCapabilities,
)
from job_mcp.models.schemas import Job, WorkMode
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.company_registry.entries import WorkableCompany
from job_mcp.sources.public.workable import (
    WorkableSource,
    _detect_work_mode,
    parse_workable_job,
)
from job_mcp.sources.registry import SourceRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "workable"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Capability Map Tests
# ===========================================================================


def test_workable_capability_map_declaration() -> None:
    """Verify Workable capabilities are accurately declared in Search Plane."""
    assert "workable" in SOURCE_CAPABILITY_MAP
    caps = SOURCE_CAPABILITY_MAP["workable"]
    assert isinstance(caps, SourceCapabilities)
    assert caps.supports_search is True
    assert caps.supports_native_fetch is True
    assert caps.supports_url_fetch is False
    assert caps.supports_query is False
    assert caps.supports_company_filter is True
    assert caps.supports_work_mode is False
    assert caps.supports_pagination is False


# ===========================================================================
# 2. JobRef Generation and Round-Trip Tests
# ===========================================================================


def test_workable_job_ref_creation_and_roundtrip() -> None:
    """Verify create_job_ref parses Workable job_id into opaque, stateless JobRef."""
    adapter = SearchPlaneAdapter()
    data = load_fixture("valid_account.json")
    job = parse_workable_job(
        data["jobs"][0],
        company_name="Hugging Face",
        account_subdomain="huggingface",
    )
    assert job.job_id == "workable_huggingface_F4C096B22E"

    ref = adapter.create_job_ref(job)
    assert ref.version == 1
    assert ref.source_family == "workable"
    assert ref.account == "huggingface"
    assert ref.locator == "F4C096B22E"

    encoded = ref.encode()
    assert encoded.startswith("v1_")
    decoded = JobRef.decode(encoded)
    assert decoded == ref


@pytest.mark.parametrize(
    ("account_subdomain", "shortcode"),
    [
        ("huggingface", "F4C096B22E"),
        ("acme-corp", "A1B2C3D4E5"),
        ("ai-research-lab", "9Z8Y7X6W5V"),
        ("company123", "SHORTCODE1"),
    ],
)
def test_workable_job_ref_account_slug_variants(
    account_subdomain: str, shortcode: str
) -> None:
    """Verify hyphenated or alphanumeric account subdomains produce stable JobRef."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id=f"workable_{account_subdomain}_{shortcode}",
        title="Software Engineer",
        company="Display Name",
        source="workable",
    )

    ref = adapter.create_job_ref(job)
    assert ref.source_family == "workable"
    assert ref.account == account_subdomain
    assert ref.locator == shortcode

    decoded = JobRef.decode(ref.encode())
    assert decoded.account == account_subdomain
    assert decoded.locator == shortcode


def test_workable_job_ref_no_prefix_fails_explicitly() -> None:
    """Non-decorated job_id raises ValueError."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="custom_raw_id",
        title="Engineer",
        company="Acme Corp",
        source="workable",
    )

    with pytest.raises(
        ValueError, match="Cannot deterministically derive Workable routing coordinates"
    ):
        adapter.create_job_ref(job)


def test_workable_job_ref_missing_delimiter_fails() -> None:
    """Job ID with workable_ prefix but missing locator delimiter raises ValueError."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="workable_nodelimiter",
        title="Engineer",
        company="Display Corp",
        source="workable",
    )

    with pytest.raises(
        ValueError, match="Malformed Workable job_id missing locator delimiter"
    ):
        adapter.create_job_ref(job)


def test_workable_job_ref_empty_account_or_locator_fails() -> None:
    """Empty account or locator raises ValueError."""
    adapter = SearchPlaneAdapter()
    job1 = Job(
        job_id="workable__locator",
        title="Engineer",
        company="Display Corp",
        source="workable",
    )
    with pytest.raises(
        ValueError, match="Malformed Workable job_id has empty account or locator"
    ):
        adapter.create_job_ref(job1)

    job2 = Job(
        job_id="workable_account_",
        title="Engineer",
        company="Display Corp",
        source="workable",
    )
    with pytest.raises(
        ValueError, match="Malformed Workable job_id has empty account or locator"
    ):
        adapter.create_job_ref(job2)


# ===========================================================================
# 3. SearchPlaneAdapter.search() Integration Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_search_with_workable() -> None:
    """Verify search() queries Workable, returns SearchPlane items with encoded refs."""
    multiple_jobs = load_fixture("multiple_jobs.json")
    mock_resp = httpx.Response(
        200, json=multiple_jobs, request=httpx.Request("GET", "https://example.com")
    )

    source = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            )
        }
    )

    registry = SourceRegistry()
    registry.register(source)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator, registry=registry)

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        request = JobSearchRequest(
            sources=["workable"],
            limit=5,
        )
        result_set = await adapter.search(request)

        assert len(result_set.items) == 3
        first = result_set.items[0]
        assert first.source_family == "workable"
        assert first.company == "Hugging Face"
        assert first.title == "Senior Machine Learning Engineer"
        ref = JobRef.decode(first.ref)
        assert ref.account == "huggingface"
        assert ref.locator == "F4C096B22E"


@pytest.mark.asyncio
async def test_search_plane_adapter_search_company_filter() -> None:
    """Company filter post-filtering works on Workable results."""
    multiple_jobs = load_fixture("multiple_jobs.json")
    mock_resp = httpx.Response(
        200, json=multiple_jobs, request=httpx.Request("GET", "https://example.com")
    )

    source = WorkableSource(
        companies={
            "huggingface": WorkableCompany(
                name="Hugging Face", account_subdomain="huggingface", enabled=True
            )
        }
    )

    registry = SourceRegistry()
    registry.register(source)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator, registry=registry)

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        request = JobSearchRequest(
            company="Hugging",
            sources=["workable"],
            limit=5,
        )
        result_set = await adapter.search(request)
        assert len(result_set.items) == 3

        request_nomatch = JobSearchRequest(
            company="Nonexistent",
            sources=["workable"],
            limit=5,
        )
        result_set_empty = await adapter.search(request_nomatch)
        assert len(result_set_empty.items) == 0


# ===========================================================================
# 4. SearchPlaneAdapter.fetch() Integration Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_cache_hit() -> None:
    """Cached Workable job is returned immediately without network call."""
    cache = JobCache()
    job = Job(
        job_id="workable_huggingface_F4C096B22E",
        title="Senior ML Engineer",
        company="Hugging Face",
        source="workable",
    )
    cache.update([job])

    adapter = SearchPlaneAdapter(cache=cache)
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="F4C096B22E")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        res = await adapter.fetch(ref)
        assert mock_get.call_count == 0

    assert res.status == FetchStatus.FOUND
    assert res.job is not None
    assert res.job.job_id == job.job_id


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_native_refetch_seam_success() -> None:
    """Cache miss delegates to source.fetch_job_by_ref and caches on FOUND."""
    cache = JobCache()
    data = load_fixture("multiple_jobs.json")
    mock_resp = httpx.Response(
        200, json=data, request=httpx.Request("GET", "https://example.com")
    )

    source = WorkableSource()
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(cache=cache, registry=registry)

    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="A1B2C3D4E5")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        res = await adapter.fetch(ref.encode())

    assert res.status == FetchStatus.FOUND
    assert res.job is not None
    assert res.job.title == "Frontend Engineer"

    # Verify populated in cache
    cached = cache.get_by_id("workable_huggingface_A1B2C3D4E5")
    assert cached is not None
    assert cached.title == "Frontend Engineer"


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_not_found_never_dummy_job() -> None:
    """Missing posting returns NOT_FOUND with job=None (zero synthetic dummy jobs)."""
    data = load_fixture("multiple_jobs.json")
    mock_resp = httpx.Response(
        200, json=data, request=httpx.Request("GET", "https://example.com")
    )

    source = WorkableSource()
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="MISSING")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        res = await adapter.fetch(ref)

    assert res.status == FetchStatus.NOT_FOUND
    assert res.job is None
    assert "MISSING" in (res.diagnostic or "")


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_upstream_error() -> None:
    """HTTP 500 maps to UPSTREAM_ERROR."""
    resp_500 = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))

    source = WorkableSource()
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="F4C096B22E")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=resp_500)):
        res = await adapter.fetch(ref)

    assert res.status == FetchStatus.UPSTREAM_ERROR
    assert res.job is None


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_invalid_string_ref() -> None:
    """Garbage string ref returns INVALID_REF."""
    adapter = SearchPlaneAdapter()
    res = await adapter.fetch("invalid_garbage_ref")
    assert res.status == FetchStatus.INVALID_REF
    assert res.job is None


# ===========================================================================
# 5. Configuration-Only Onboarding Proof
# ===========================================================================


@pytest.mark.asyncio
async def test_config_only_onboarding_proof() -> None:
    """Adding a new company to CompanyRegistry dynamically enables fetching and refetching without editing workable.py."""
    dynamic_company = WorkableCompany(
        name="Dynamic AI",
        account_subdomain="dynamic-ai",
        enabled=True,
    )
    src = WorkableSource(companies={"dynamic_ai": dynamic_company})

    payload = {
        "name": "Dynamic AI",
        "description": "Next gen AI",
        "jobs": [
            {
                "title": "Staff AI Researcher",
                "shortcode": "DYNAI12345",
                "telecommuting": True,
                "city": "Tel Aviv",
                "country": "Israel",
                "description": "<p>Build foundation models.</p>",
            }
        ],
    }
    mock_resp = httpx.Response(200, json=payload, request=httpx.Request("GET", "https://example.com"))

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        jobs = await src.fetch_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].company == "Dynamic AI"
        assert jobs[0].job_id == "workable_dynamic-ai_DYNAI12345"

        ref = JobRef(version=1, source_family="workable", account="dynamic-ai", locator="DYNAI12345")
        res = await src.fetch_job_by_ref(ref)
        assert res.status == FetchStatus.FOUND
        assert res.job is not None
        assert res.job.title == "Staff AI Researcher"


# ===========================================================================
# 6. Reconciled Original 15 Mutation Proofs
# ===========================================================================


@pytest.mark.asyncio
async def test_mutation_proof_01_wrong_public_base_url_fails() -> None:
    """Original Mutation 1: wrong Workable public base URL.

    Asserts that request URL targets host 'www.workable.com' and path '/api/accounts/{account}'.
    Fails if base URL is mutated to a different domain, path, or API version.
    """
    src = WorkableSource(
        companies={"huggingface": WorkableCompany(name="HF", account_subdomain="huggingface", enabled=True)}
    )
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))) as mock_get:
        await src.fetch_jobs(limit=10)
        assert mock_get.called
        called_url = mock_get.call_args[0][0]
        parsed = urlsplit(called_url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "www.workable.com"
        assert parsed.path == "/api/accounts/huggingface"


@pytest.mark.asyncio
async def test_mutation_proof_02_missing_details_parameter_fails() -> None:
    """Original Mutation 2: missing details=true query parameter.

    Workable account endpoint requires details=true to return full job models.
    Fails if details=true is omitted from outgoing URL query.
    """
    src = WorkableSource(
        companies={"huggingface": WorkableCompany(name="HF", account_subdomain="huggingface", enabled=True)}
    )
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))) as mock_get:
        await src.fetch_jobs(limit=10)
        assert mock_get.called
        called_url = mock_get.call_args[0][0]
        parsed = urlsplit(called_url)
        assert parsed.query == "details=true", f"Expected 'details=true', got {parsed.query!r}"

    # Also during fetch_job_by_ref
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))) as mock_get:
        ref = JobRef(version=1, source_family="workable", account="huggingface", locator="TOK")
        await src.fetch_job_by_ref(ref)
        assert mock_get.called
        called_url = mock_get.call_args[0][0]
        parsed = urlsplit(called_url)
        assert parsed.query == "details=true", f"Expected 'details=true', got {parsed.query!r}"


@pytest.mark.asyncio
async def test_mutation_proof_03_wrong_account_subdomain_fails() -> None:
    """Original Mutation 3: wrong account subdomain used as routing coordinate.

    Proves that configured account_subdomain (not company name, catalog id, or hardcoded slug)
    is the exact routing coordinate used in the request URL path.
    """
    src = WorkableSource(
        companies={
            "custom_id": WorkableCompany(name="Custom Display Name", account_subdomain="exact-subdomain", enabled=True)
        }
    )
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))) as mock_get:
        await src.fetch_jobs(limit=10)
        assert mock_get.called
        called_url = mock_get.call_args[0][0]
        parsed = urlsplit(called_url)
        assert parsed.path == "/api/accounts/exact-subdomain"
        assert "custom_id" not in parsed.path
        assert "Custom Display Name" not in parsed.path


@pytest.mark.asyncio
async def test_mutation_proof_04_disabled_account_still_queried_fails() -> None:
    """Original Mutation 4: disabled account still queried.

    A disabled company in the catalog must never trigger an outgoing network request.
    """
    src = WorkableSource(
        companies={
            "enabled_co": WorkableCompany(name="Enabled", account_subdomain="enabled-co", enabled=True),
            "disabled_co": WorkableCompany(name="Disabled", account_subdomain="disabled-co", enabled=False),
        }
    )
    queried_urls: list[str] = []

    async def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        queried_urls.append(url)
        return httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", url))

    with patch("httpx.AsyncClient.get", side_effect=fake_get):
        await src.fetch_jobs(limit=10)

    assert len(queried_urls) == 1
    assert "enabled-co" in queried_urls[0]
    assert not any("disabled-co" in u for u in queried_urls)


@pytest.mark.asyncio
async def test_mutation_proof_05_config_added_account_ignored_fails() -> None:
    """Original Mutation 5: config-added account ignored.

    A newly configured account in CompanyRegistry/catalog must be discovered and queried.
    """
    dynamic = WorkableCompany(name="New Startup", account_subdomain="new-startup", enabled=True)
    src = WorkableSource(companies={"new_startup": dynamic})

    payload = {"jobs": [{"title": "Founder", "shortcode": "SHORT12345"}]}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json=payload, request=httpx.Request("GET", "https://example.com")))):
        jobs = await src.fetch_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].company == "New Startup"
        assert jobs[0].job_id == "workable_new-startup_SHORT12345"


@pytest.mark.asyncio
async def test_mutation_proof_06_malformed_response_treated_as_valid_empty_account_fails() -> None:
    """Original Mutation 6: malformed response treated as valid empty account.

    A malformed upstream payload (non-dict or non-list jobs) must produce UPSTREAM_ERROR on refetch,
    and must not be conflated with a valid empty account containing jobs=[].
    """
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="SHORT12345")

    # 1. Non-dict payload -> UPSTREAM_ERROR
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json=["invalid", "list"], request=httpx.Request("GET", "https://example.com")))):
        res_malformed = await src.fetch_job_by_ref(ref)
        assert res_malformed.status == FetchStatus.UPSTREAM_ERROR
        assert res_malformed.job is None

    # 2. Non-list jobs field -> UPSTREAM_ERROR
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": "not-a-list"}, request=httpx.Request("GET", "https://example.com")))):
        res_nonlist = await src.fetch_job_by_ref(ref)
        assert res_nonlist.status == FetchStatus.UPSTREAM_ERROR
        assert res_nonlist.job is None

    # 3. Valid account with jobs: [] -> NOT_FOUND (different status!)
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))):
        res_valid_empty = await src.fetch_job_by_ref(ref)
        assert res_valid_empty.status == FetchStatus.NOT_FOUND
        assert res_valid_empty.job is None


@pytest.mark.asyncio
async def test_mutation_proof_07_refetch_returns_first_posting_instead_of_exact_locator_fails() -> None:
    """Original Mutation 7: refetch returns first posting instead of exact locator.

    When multiple postings exist on an account, fetch_job_by_ref must match exact locator.
    Fails if implementation naively returns jobs[0].
    """
    src = WorkableSource()
    data = load_fixture("multiple_jobs.json")  # jobs[0] is F4C096B22E, jobs[1] is A1B2C3D4E5
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="A1B2C3D4E5")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json=data, request=httpx.Request("GET", "https://example.com")))):
        res = await src.fetch_job_by_ref(ref)
        assert res.status == FetchStatus.FOUND
        assert res.job is not None
        assert res.job.job_id == "workable_huggingface_A1B2C3D4E5"
        assert res.job.job_id != "workable_huggingface_F4C096B22E"
        assert res.job.title == "Frontend Engineer"


@pytest.mark.asyncio
async def test_mutation_proof_08_locator_missing_but_returns_found_fails() -> None:
    """Original Mutation 8: locator missing but returns FOUND.

    When the requested locator does not exist in the account's jobs array,
    fetch_job_by_ref must return NOT_FOUND with job=None, never FOUND.
    """
    src = WorkableSource()
    data = load_fixture("multiple_jobs.json")
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="NONEXISTENT")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json=data, request=httpx.Request("GET", "https://example.com")))):
        res = await src.fetch_job_by_ref(ref)
        assert res.status == FetchStatus.NOT_FOUND
        assert res.status != FetchStatus.FOUND
        assert res.job is None


def test_mutation_proof_09_malformed_ref_uses_display_company_as_routing_coordinate_fails() -> None:
    """Original Mutation 9: malformed ref uses display company as routing coordinate.

    Unlike legacy providers, Workable must fail explicitly with ValueError on malformed
    job_ids and must NEVER fall back to job.company as routing account.
    """
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="malformed_id_without_prefix",
        title="Engineer",
        company="Legitimate Company Name",
        source="workable",
    )
    with pytest.raises(ValueError, match="Cannot deterministically derive Workable routing coordinates"):
        adapter.create_job_ref(job)


def test_mutation_proof_10_arbitrary_description_truncation_fails() -> None:
    """Original Mutation 10: arbitrary description truncation.

    Full HTML/plain text description (>2000 chars) must be preserved in Job.description without truncation.
    """
    data = load_fixture("full_description.json")
    job = parse_workable_job(data["jobs"][0], "HF", "huggingface")
    assert len(job.description) > 2000
    assert not job.description.endswith("...")


def test_mutation_proof_11_zero_work_mode_evidence_becomes_onsite_fails() -> None:
    """Original Mutation 11: zero work-mode evidence becomes ONSITE.

    When telecommuting is False, location is empty, and description has no work mode keywords,
    work_mode must be None, NOT WorkMode.ONSITE.
    """
    work_mode = _detect_work_mode(False, "", "General software engineering role")
    assert work_mode is None
    assert work_mode != WorkMode.ONSITE


@pytest.mark.asyncio
async def test_mutation_proof_12_http_429_and_5xx_incorrectly_become_not_found_fails() -> None:
    """Original Mutation 12: 429/5xx incorrectly becomes NOT_FOUND.

    HTTP 429 rate limit or HTTP 500 server error during refetch must return UPSTREAM_ERROR,
    never NOT_FOUND.
    """
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="F4C096B22E")

    # 429 Rate Limit
    resp_429 = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=resp_429)):
        res429 = await src.fetch_job_by_ref(ref)
        assert res429.status == FetchStatus.UPSTREAM_ERROR
        assert res429.status != FetchStatus.NOT_FOUND

    # 500 Server Error
    resp_500 = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=resp_500)):
        res500 = await src.fetch_job_by_ref(ref)
        assert res500.status == FetchStatus.UPSTREAM_ERROR
        assert res500.status != FetchStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_mutation_proof_13_lifecycle_status_such_as_expired_leaks_into_b3_fails() -> None:
    """Original Mutation 13: lifecycle status such as EXPIRED leaks into B3.

    Fetch outcomes must be strictly factual (FOUND, NOT_FOUND, UPSTREAM_ERROR, INVALID_REF).
    A missing posting must never be labeled EXPIRED, STALE, or DEAD.
    """
    src = WorkableSource()
    ref = JobRef(version=1, source_family="workable", account="huggingface", locator="MISSING")
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json={"jobs": []}, request=httpx.Request("GET", "https://example.com")))):
        res = await src.fetch_job_by_ref(ref)
        assert res.status == FetchStatus.NOT_FOUND
        assert res.status not in ("expired", "stale", "dead", "inactive")


def test_mutation_proof_14_capability_incorrectly_claims_native_query_fails() -> None:
    """Original Mutation 14: capability incorrectly claims native query.

    Workable public widget endpoint does not support server-side query filtering;
    supports_query must be False.
    """
    assert SOURCE_CAPABILITY_MAP["workable"].supports_query is False


def test_mutation_proof_15_capability_incorrectly_claims_pagination_fails() -> None:
    """Original Mutation 15: capability incorrectly claims pagination.

    Workable public widget endpoint returns all jobs in one payload without pagination;
    supports_pagination must be False.
    """
    assert SOURCE_CAPABILITY_MAP["workable"].supports_pagination is False


# ===========================================================================
# 7. Additional Negative & Invariant Proofs
# ===========================================================================


def test_auxiliary_delimiter_parsing_hyphenated_account() -> None:
    """Parsing with job_id.split('_')[1] instead of rsplit('_', 1) breaks hyphenated accounts."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="workable_hugging-face-inc_TOKEN123",
        title="Engineer",
        company="HF",
        source="workable",
    )
    ref = adapter.create_job_ref(job)
    assert ref.account == "hugging-face-inc"
    assert ref.locator == "TOKEN123"


def test_auxiliary_subdomain_rejects_underscores() -> None:
    """Subdomain validator permitting underscores must fail."""
    from job_mcp.sources.company_registry.core import (
        CompanyRegistryError,
        validate_document,
    )
    with pytest.raises(CompanyRegistryError):
        validate_document(
            {
                "providers": {
                    "workable": {
                        "companies": [
                            {"id": "test", "name": "Test", "account_subdomain": "has_underscore"}
                        ]
                    }
                }
            }
        )


def test_auxiliary_job_ref_version_invariance() -> None:
    """JobRef version must be 1."""
    adapter = SearchPlaneAdapter()
    job = Job(job_id="workable_acc_tok", title="T", company="C", source="workable")
    ref = adapter.create_job_ref(job)
    assert ref.version == 1


@pytest.mark.asyncio
async def test_auxiliary_cache_populated_on_found() -> None:
    """Cache must be populated on FOUND refetch."""
    cache = JobCache()
    adapter = SearchPlaneAdapter(cache=cache)
    payload = {"jobs": [{"title": "T", "shortcode": "TOK", "description": "D"}]}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=httpx.Response(200, json=payload, request=httpx.Request("GET", "https://example.com")))):
        ref = JobRef(version=1, source_family="workable", account="acc", locator="TOK")
        res = await adapter.fetch(ref)
    assert res.status == FetchStatus.FOUND
    assert cache.get_by_id("workable_acc_TOK") is not None
