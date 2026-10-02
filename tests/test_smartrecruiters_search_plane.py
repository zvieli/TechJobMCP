"""Tests for SmartRecruiters ATS integration with the Unified Search Plane (Milestone 6-B2).

Verifies:
- SmartRecruiters source capability declarations (supports_query=True, supports_pagination=True)
- JobRef generation and deterministic encode/decode round-trip for SmartRecruiters postings
- Separation of companyIdentifier from postingId across underscore/hyphen/dot slug variants
- SearchPlaneAdapter.search() retrieval, query propagation, and normalization
- SearchPlaneAdapter.fetch() cache hits and native refetch dispatch via generic seam
- Negative cases: 404 -> NOT_FOUND (never synthetic dummy Job), HTTP 500, invalid ref
- Configuration-only onboarding of new SmartRecruiters companies without Python code edits
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

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
from job_mcp.models.schemas import Job
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.company_registry.entries import SmartRecruitersCompany
from job_mcp.sources.public.smartrecruiters import (
    SmartRecruitersSource,
    parse_smartrecruiters_job,
)
from job_mcp.sources.registry import SourceRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "smartrecruiters"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Capability Map Tests
# ===========================================================================


def test_smartrecruiters_capability_map_declaration() -> None:
    """Verify SmartRecruiters capabilities are accurately declared in Search Plane."""
    assert "smartrecruiters" in SOURCE_CAPABILITY_MAP
    caps = SOURCE_CAPABILITY_MAP["smartrecruiters"]
    assert isinstance(caps, SourceCapabilities)
    assert caps.supports_search is True
    assert caps.supports_native_fetch is True
    assert caps.supports_url_fetch is False
    assert caps.supports_query is True
    assert caps.supports_company_filter is True
    assert caps.supports_work_mode is False
    assert caps.supports_pagination is True


# ===========================================================================
# 2. JobRef Generation and Round-Trip Tests
# ===========================================================================


def test_smartrecruiters_job_ref_creation_and_roundtrip() -> None:
    """Verify create_job_ref parses SmartRecruiters job_id into opaque, stateless JobRef."""
    adapter = SearchPlaneAdapter()
    valid_postings = load_fixture("valid_postings.json")
    job = parse_smartrecruiters_job(
        valid_postings["content"][0],
        company_name="SmartRecruiters Inc",
        company_identifier="smartrecruiters",
    )
    assert job.job_id == "smartrecruiters_smartrecruiters_743999961234567"

    ref = adapter.create_job_ref(job)
    assert ref.version == 1
    assert ref.source_family == "smartrecruiters"
    assert ref.account == "smartrecruiters"
    assert ref.locator == "743999961234567"

    encoded = ref.encode()
    assert encoded.startswith("v1_")
    decoded = JobRef.decode(encoded)
    assert decoded == ref


@pytest.mark.parametrize(
    ("company_identifier", "posting_id"),
    [
        ("my_company", "743999961234567"),
        ("foo-bar", "743999961234568"),
        ("foo.bar", "743999961234569"),
        ("complex_sub_domain_slug", "uuid-posting-token-1234"),
    ],
)
def test_smartrecruiters_job_ref_identifier_slug_variants(
    company_identifier: str, posting_id: str
) -> None:
    """Verify company identifiers with underscores, hyphens, or dots are parsed without ambiguity."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id=f"smartrecruiters_{company_identifier}_{posting_id}",
        title="Software Engineer",
        company="Display Name",
        source="smartrecruiters",
    )

    ref = adapter.create_job_ref(job)
    assert ref.source_family == "smartrecruiters"
    assert ref.account == company_identifier
    assert ref.locator == posting_id

    # Must round-trip perfectly
    decoded = JobRef.decode(ref.encode())
    assert decoded.account == company_identifier
    assert decoded.locator == posting_id


def test_smartrecruiters_job_ref_url_fallback() -> None:
    """Verify create_job_ref extracts coordinates from canonical SmartRecruiters URL."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="custom_raw_id",
        title="Engineer",
        company="Acme Corp",
        source="smartrecruiters",
        url="https://jobs.smartrecruiters.com/acme-corp/743999961234567",
    )

    ref = adapter.create_job_ref(job)
    assert ref.source_family == "smartrecruiters"
    assert ref.account == "acme-corp"
    assert ref.locator == "743999961234567"


def test_smartrecruiters_job_ref_malformed_fails_without_company_fallback() -> None:
    """Malformed job IDs raise ValueError rather than using display company name."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="malformed_job_id",
        title="Engineer",
        company="Display Corp",
        source="smartrecruiters",
    )

    with pytest.raises(
        ValueError, match="Cannot deterministically derive SmartRecruiters routing coordinates"
    ):
        adapter.create_job_ref(job)


# ===========================================================================
# 3. SearchPlaneAdapter.search() Integration Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_search_with_smartrecruiters() -> None:
    """Verify search() queries SmartRecruiters, applies query keywords, and returns SearchPlane items."""
    valid_postings = load_fixture("valid_postings.json")

    mock_resp = httpx.Response(
        200, json=valid_postings, request=httpx.Request("GET", "https://example.com")
    )

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters Inc", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    registry = SourceRegistry()
    registry.register(source)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator, registry=registry)

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        request = JobSearchRequest(
            query="python backend",
            sources=["smartrecruiters"],
            limit=5,
        )
        result_set = await adapter.search(request)

        assert mock_get.called
        params = mock_get.call_args[1].get("params", {})
        assert params.get("destination") == "PUBLIC"
        assert params.get("q") == "python backend"
        assert len(result_set.items) == 2

        first_item = result_set.items[0]
        assert first_item.source_family == "smartrecruiters"
        assert first_item.title == "Senior Python Backend Engineer"
        assert first_item.company == "SmartRecruiters Inc"
        assert first_item.retrieval_method == "structured"

        # Decode ref
        decoded_ref = JobRef.decode(first_item.ref)
        assert decoded_ref.source_family == "smartrecruiters"
        assert decoded_ref.account == "smartrecruiters"
        assert decoded_ref.locator == "743999961234567"


# ===========================================================================
# 4. SearchPlaneAdapter.fetch() Generic Seam & Native Refetch Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_cache_hit() -> None:
    """Cache hits return immediately without invoking network."""
    cache = JobCache(ttl_minutes=60)
    adapter = SearchPlaneAdapter(cache=cache)
    job = Job(
        job_id="smartrecruiters_smartrecruiters_743999961234567",
        title="Senior Python Backend Engineer",
        company="SmartRecruiters Inc",
        source="smartrecruiters",
        description="Cached description",
    )
    cache.update([job])

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="743999961234567",
    )
    result = await adapter.fetch(ref)

    assert result.status == FetchStatus.FOUND
    assert result.job is not None
    assert result.job.title == "Senior Python Backend Engineer"
    assert result.job.description == "Cached description"


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_native_refetch_success() -> None:
    """Cache miss delegates to SmartRecruitersSource.fetch_job_by_ref via generic seam."""
    detail_data = load_fixture("valid_detail.json")

    mock_resp = httpx.Response(
        200, json=detail_data, request=httpx.Request("GET", "https://example.com")
    )

    source = SmartRecruitersSource(
        companies={
            "smartrecruiters": SmartRecruitersCompany(
                name="SmartRecruiters Inc", company_identifier="smartrecruiters", enabled=True
            )
        }
    )

    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="743999961234567",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        result = await adapter.fetch(ref)

        assert result.status == FetchStatus.FOUND
        assert result.job is not None
        assert result.job.title == "Senior Python Backend Engineer"
        assert result.job.salary_range == "160,000 - 195,000 USD"
        assert mock_get.called
        assert (
            mock_get.call_args[0][0]
            == "https://api.smartrecruiters.com/v1/companies/smartrecruiters/postings/743999961234567"
        )



@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_404_returns_not_found() -> None:
    """Missing posting returns NOT_FOUND with job=None; never fabricates dummy Job."""
    mock_resp = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))

    source = SmartRecruitersSource()
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="nonexistent-743",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await adapter.fetch(ref)

    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert "not found" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_500_returns_upstream_error() -> None:
    """Upstream 500 error returns UPSTREAM_ERROR with job=None."""
    mock_resp = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))

    source = SmartRecruitersSource()
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="smartrecruiters",
        locator="error-743",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        result = await adapter.fetch(ref)

    assert result.status == FetchStatus.UPSTREAM_ERROR
    assert result.job is None


# ===========================================================================
# 5. Configuration-Only Onboarding Proof
# ===========================================================================


@pytest.mark.asyncio
async def test_smartrecruiters_config_only_addition_without_source_edits() -> None:
    """Prove a company added solely via catalog config is queried and refetched without editing smartrecruiters.py."""
    detail_data = load_fixture("valid_detail.json")
    detail_data["id"] = "999888777"
    detail_data["name"] = "Chief AI Architect"

    mock_resp = httpx.Response(
        200, json=detail_data, request=httpx.Request("GET", "https://example.com")
    )

    # Custom company injected strictly via config catalog
    config_companies = {
        "acme_corp": SmartRecruitersCompany(
            name="Acme Corporation",
            company_identifier="acmecorp",
            enabled=True,
        )
    }

    source = SmartRecruitersSource(companies=config_companies)
    registry = SourceRegistry()
    registry.register(source)
    adapter = SearchPlaneAdapter(registry=registry)

    ref = JobRef(
        version=1,
        source_family="smartrecruiters",
        account="acmecorp",
        locator="999888777",
    )

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)) as mock_get:
        res = await adapter.fetch(ref)
        assert res.status == FetchStatus.FOUND
        assert res.job is not None
        assert res.job.title == "Chief AI Architect"
        assert res.job.company == "Acme Corporation"
        assert (
            mock_get.call_args[0][0]
            == "https://api.smartrecruiters.com/v1/companies/acmecorp/postings/999888777"
        )
