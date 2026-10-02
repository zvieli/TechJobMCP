"""Tests for Ashby ATS integration with the Unified Search Plane (Milestone 6-B1).

Verifies:
- Ashby source capability declarations
- JobRef generation and deterministic encode/decode round-trip for Ashby postings
- SearchPlaneAdapter.search() retrieval and normalization for Ashby postings
- SearchPlaneAdapter.fetch() cache hits and native refetch dispatch via AshbySource.fetch_job_by_ref
- Negative cases: missing posting returns NOT_FOUND (never synthetic dummy Job),
  HTTP 404, HTTP 500, invalid ref
- Configuration-only onboarding of new Ashby companies without Python code edits
- Mutation proofs verifying exact locator matching and error classification
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from job_mcp.core.search_plane.adapter import SOURCE_CAPABILITY_MAP, SearchPlaneAdapter
from job_mcp.core.search_plane.models import (
    FetchStatus,
    JobRef,
    JobSearchRequest,
    SourceCapabilities,
)
from job_mcp.models.schemas import Job
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.company_registry.entries import AshbyCompany
from job_mcp.sources.public.ashby import AshbySource, parse_ashby_job
from job_mcp.sources.registry import SourceRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ashby"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Capability Map Tests
# ===========================================================================


def test_ashby_capability_map_declaration() -> None:
    """Verify Ashby capabilities are accurately declared in Search Plane."""
    assert "ashby" in SOURCE_CAPABILITY_MAP
    caps = SOURCE_CAPABILITY_MAP["ashby"]
    assert isinstance(caps, SourceCapabilities)
    assert caps.supports_search is True
    assert caps.supports_native_fetch is True
    assert caps.supports_company_filter is True
    assert caps.supports_work_mode is True
    assert caps.supports_query is False
    assert caps.supports_pagination is False


# ===========================================================================
# 2. JobRef Generation and Round-Trip Tests
# ===========================================================================


def test_ashby_job_ref_creation_and_roundtrip() -> None:
    """Verify create_job_ref parses Ashby job_id into opaque, stateless JobRef."""
    adapter = SearchPlaneAdapter()
    valid_board = load_fixture("valid_board.json")
    job = parse_ashby_job(valid_board["jobs"][0], company_name="Ashby", board_name="ashby")
    assert job.job_id == "ashby_ashby_c1f77d34-7a42-4f05-8a8b-302efb1a4731"

    ref = adapter.create_job_ref(job)
    assert ref.version == 1
    assert ref.source_family == "ashby"
    assert ref.account == "ashby"
    assert ref.locator == "c1f77d34-7a42-4f05-8a8b-302efb1a4731"

    encoded = ref.encode()
    assert encoded.startswith("v1_")
    decoded = JobRef.decode(encoded)
    assert decoded == ref


def test_ashby_job_ref_custom_board_slug() -> None:
    """Verify create_job_ref handles multi-part board slugs and locators."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id="ashby_openai-internal_98765432-abcd-ef01-2345-6789abcdef01",
        title="Software Engineer",
        company="OpenAI Internal",
        source="ashby",
        url="https://jobs.ashbyhq.com/openai-internal/98765432-abcd-ef01-2345-6789abcdef01",
    )
    ref = adapter.create_job_ref(job)
    assert ref.source_family == "ashby"
    assert ref.account == "openai-internal"
    assert ref.locator == "98765432-abcd-ef01-2345-6789abcdef01"

    decoded = JobRef.decode(ref.encode())
    assert decoded == ref


@pytest.mark.parametrize(
    ("board_name", "locator"),
    [
        ("my_company", "c1f77d34-7a42-4f05-8a8b-302efb1a4731"),
        ("foo-bar", "98765432-abcd-ef01-2345-6789abcdef01"),
        ("foo.bar", "12345678-abcd-ef01-2345-6789abcdef01"),
        ("complex_sub_domain_slug", "a1b2c3d4-e5f6-7890-1234-56789abcdef0"),
    ],
)
def test_ashby_job_ref_board_slug_variants_underscore_hyphen_dot(
    board_name: str, locator: str
) -> None:
    """Verify board names containing underscores, hyphens, and dots round-trip without ambiguity."""
    adapter = SearchPlaneAdapter()
    job = Job(
        job_id=f"ashby_{board_name}_{locator}",
        title="Senior Platform Engineer",
        company="Display Name Inc",  # Intentionally distinct from board_name slug
        source="ashby",
    )

    ref = adapter.create_job_ref(job)
    assert ref.source_family == "ashby"
    assert ref.account == board_name
    assert ref.locator == locator

    encoded = ref.encode()
    decoded = JobRef.decode(encoded)
    assert decoded.account == board_name
    assert decoded.locator == locator
    assert decoded == ref


@pytest.mark.asyncio
async def test_ashby_job_ref_underscore_routes_fetch_correctly() -> None:
    """Verify that an underscore-containing board name routes fetch() to the correct board URL."""
    board_name = "my_company"
    uuid = "c1f77d34-7a42-4f05-8a8b-302efb1a4731"
    job = Job(
        job_id=f"ashby_{board_name}_{uuid}",
        title="Senior Backend Engineer",
        company="My Company Inc.",
        source="ashby",
    )

    adapter = SearchPlaneAdapter()
    ref = adapter.create_job_ref(job)
    assert ref.account == "my_company"
    assert ref.locator == uuid

    # Mock response for my_company board
    board_payload = {
        "apiVersion": "1",
        "jobs": [
            {
                "id": uuid,
                "title": "Senior Backend Engineer",
                "location": "Tel Aviv",
                "isRemote": False,
                "jobUrl": f"https://jobs.ashbyhq.com/{board_name}/{uuid}",
            }
        ],
    }

    mock_resp = httpx.Response(200, json=board_payload, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp) as mock_get:
        fetch_res = await adapter.fetch(ref.encode())
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert f"/job-board/{board_name}" in called_url
        assert "includeCompensation=true" in called_url

    assert fetch_res.status == FetchStatus.FOUND
    assert fetch_res.job is not None
    assert fetch_res.job.title == "Senior Backend Engineer"
    assert fetch_res.job.job_id == f"ashby_{board_name}_{uuid}"


def test_ashby_job_ref_malformed_fails_without_company_fallback() -> None:
    """Verify malformed Ashby job IDs fail explicitly and NEVER fall back to job.company display name."""
    adapter = SearchPlaneAdapter()

    # No locator delimiter
    job_no_delim = Job(
        job_id="ashby_singletoken",
        title="Engineer",
        company="Insecure Display Company",
        source="ashby",
    )
    with pytest.raises(ValueError, match="missing locator delimiter"):
        adapter.create_job_ref(job_no_delim)

    # Empty payload or wrong prefix
    job_wrong_prefix = Job(
        job_id="custom_id_12345",
        title="Engineer",
        company="Insecure Display Company",
        source="ashby",
    )
    with pytest.raises(ValueError, match="Cannot deterministically derive"):
        adapter.create_job_ref(job_wrong_prefix)


# ===========================================================================
# 3. SearchPlaneAdapter.search() with Ashby
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_search_with_ashby() -> None:
    """Verify SearchPlaneAdapter.search() returns JobSearchResultItems with valid JobRefs."""
    valid_board = load_fixture("valid_board.json")
    ashby_src = AshbySource(
        companies={"ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True)}
    )

    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    mock_resp = httpx.Response(200, json=valid_board, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        result_set = await adapter.search(JobSearchRequest(sources=["ashby"], limit=10))

    assert len(result_set.items) == 2
    item1 = result_set.items[0]
    assert item1.source_family == "ashby"
    assert item1.title == "Senior Backend Engineer"
    assert item1.company == "Ashby"
    assert item1.ref.startswith("v1_")

    ref1 = JobRef.decode(item1.ref)
    assert ref1.source_family == "ashby"
    assert ref1.account == "ashby"
    assert ref1.locator == "c1f77d34-7a42-4f05-8a8b-302efb1a4731"


@pytest.mark.asyncio
async def test_search_plane_adapter_search_company_filter() -> None:
    """Verify company filter restricts search results to the requested company."""
    valid_board = load_fixture("valid_board.json")
    ashby_src = AshbySource(
        companies={
            "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True),
            "other": AshbyCompany(name="OtherCo", board_name="otherco", enabled=True),
        }
    )

    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    mock_resp = httpx.Response(200, json=valid_board, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        # Filtering for Ashby should include Ashby roles
        res = await adapter.search(JobSearchRequest(company="Ashby", sources=["ashby"]))
        assert len(res.items) == 2

        # Filtering for NonExistent should yield 0 results
        res_empty = await adapter.search(JobSearchRequest(company="NonExistent", sources=["ashby"]))
        assert len(res_empty.items) == 0


# ===========================================================================
# 4. SearchPlaneAdapter.fetch() with Ashby
# ===========================================================================


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_cache_hit() -> None:
    """Verify cached Ashby job returns FOUND immediately without HTTP call."""
    from job_mcp.core.api_client import JobCache

    cache = JobCache(ttl_minutes=60)
    cached_job = Job(
        job_id="ashby_ashby_cached-loc-123",
        title="Cached Engineer",
        company="Ashby",
        source="ashby",
        url="https://jobs.ashbyhq.com/ashby/cached-loc-123",
    )
    cache.update([cached_job])

    registry = SourceRegistry()
    registry.register(AshbySource())
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator, cache=cache)

    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="cached-loc-123")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        fetch_res = await adapter.fetch(ref.encode())
        mock_get.assert_not_called()

    assert fetch_res.status == FetchStatus.FOUND
    assert fetch_res.job is not None
    assert fetch_res.job.title == "Cached Engineer"


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_native_board_refetch_success() -> None:
    """Verify cache-miss fetches Ashby board, matches locator, and returns FOUND."""
    valid_board = load_fixture("valid_board.json")
    ashby_src = AshbySource(
        companies={"ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True)}
    )
    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    ref = JobRef(
        version=1,
        source_family="ashby",
        account="ashby",
        locator="c1f77d34-7a42-4f05-8a8b-302efb1a4731",
    )
    mock_resp = httpx.Response(200, json=valid_board, request=httpx.Request("GET", "https://example.com"))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        fetch_res = await adapter.fetch(ref.encode())

    assert fetch_res.status == FetchStatus.FOUND
    assert fetch_res.job is not None
    assert fetch_res.job.title == "Senior Backend Engineer"
    assert fetch_res.job.job_id == "ashby_ashby_c1f77d34-7a42-4f05-8a8b-302efb1a4731"


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_not_found_on_missing_posting() -> None:
    """Verify missing posting locator returns NOT_FOUND and NEVER fabricates dummy Job."""
    valid_board = load_fixture("valid_board.json")
    ashby_src = AshbySource()
    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    ref = JobRef(
        version=1,
        source_family="ashby",
        account="ashby",
        locator="non-existent-locator-9999",
    )
    mock_resp = httpx.Response(200, json=valid_board, request=httpx.Request("GET", "https://example.com"))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        fetch_res = await adapter.fetch(ref.encode())

    assert fetch_res.status == FetchStatus.NOT_FOUND
    assert fetch_res.job is None
    assert "not found" in (fetch_res.diagnostic or "").lower()


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_board_404_returns_not_found() -> None:
    """Verify board HTTP 404 returns NOT_FOUND and no dummy job."""
    ashby_src = AshbySource()
    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    ref = JobRef(version=1, source_family="ashby", account="ghost_board", locator="loc-123")
    mock_resp = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        fetch_res = await adapter.fetch(ref.encode())

    assert fetch_res.status == FetchStatus.NOT_FOUND
    assert fetch_res.job is None
    assert "404" in (fetch_res.diagnostic or "")


@pytest.mark.asyncio
async def test_search_plane_adapter_fetch_upstream_error() -> None:
    """Verify HTTP 500 returns UPSTREAM_ERROR."""
    ashby_src = AshbySource()
    registry = SourceRegistry()
    registry.register(ashby_src)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="loc-123")
    mock_resp = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        fetch_res = await adapter.fetch(ref.encode())

    assert fetch_res.status == FetchStatus.UPSTREAM_ERROR
    assert fetch_res.job is None


# ===========================================================================
# 5. Configuration-Only Company Addition
# ===========================================================================


@pytest.mark.asyncio
async def test_configuration_only_ashby_addition() -> None:
    """Verify adding a new company via config dictionary enables search without Python edits."""
    # Simulated config entry: company added dynamically to Ashby catalog
    new_company = AshbyCompany(name="Ramp", board_name="ramp", enabled=True)
    custom_catalog = {"ramp": new_company}

    source = AshbySource(companies=custom_catalog)
    registry = SourceRegistry()
    registry.register(source)
    aggregator = JobAggregator(registry=registry)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    ramp_jobs = {
        "apiVersion": "1",
        "jobs": [
            {
                "id": "ramp-uuid-001",
                "title": "Backend Staff Engineer",
                "jobUrl": "https://jobs.ashbyhq.com/ramp/ramp-uuid-001",
                "location": "New York, NY",
                "isRemote": False,
            }
        ],
    }

    mock_resp = httpx.Response(200, json=ramp_jobs, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp) as mock_get:
        res = await adapter.search(JobSearchRequest(company="Ramp", sources=["ashby"]))
        # Verify requested URL was for the configured board
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "ramp" in called_url

    assert len(res.items) == 1
    assert res.items[0].title == "Backend Staff Engineer"
    assert res.items[0].company == "Ramp"
    ref = JobRef.decode(res.items[0].ref)
    assert ref.account == "ramp"
    assert ref.locator == "ramp-uuid-001"


# ===========================================================================
# 6. Mutation Proofs
# ===========================================================================


@pytest.mark.asyncio
async def test_mutation_proof_locator_matching() -> None:
    """Mutation proof 1 & 3: locator matching must be exact; never return first posting."""
    valid_board = load_fixture("valid_board.json")
    ashby_src = AshbySource()

    # Second posting on valid_board
    second_uuid = "e8a93b42-1234-4567-89ab-cdef01234567"
    ref_second = JobRef(version=1, source_family="ashby", account="ashby", locator=second_uuid)

    mock_resp = httpx.Response(200, json=valid_board, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await ashby_src.fetch_job_by_ref(ref_second)

    # Must match the second posting, NOT the first posting (c1f77d34-...)
    assert res.status == FetchStatus.FOUND
    assert res.job is not None
    assert res.job.title == "Staff Full Stack Engineer (Remote)"
    assert res.job.job_id == f"ashby_ashby_{second_uuid}"

    # Partial locator prefix must NOT match
    partial_ref = JobRef(
        version=1,
        source_family="ashby",
        account="ashby",
        locator="e8a93b42",  # Prefix only
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res_partial = await ashby_src.fetch_job_by_ref(partial_ref)

    assert res_partial.status == FetchStatus.NOT_FOUND
    assert res_partial.job is None


@pytest.mark.asyncio
async def test_mutation_malformed_board_response_returns_upstream_error() -> None:
    """Mutation proof 8: malformed JSON response must return UPSTREAM_ERROR, not treated as empty board."""
    ashby_src = AshbySource()
    ref = JobRef(version=1, source_family="ashby", account="ashby", locator="some-id")

    # Malformed: 'jobs' is an integer, string, or null
    mock_resp = httpx.Response(200, json={"apiVersion": "1", "jobs": "invalid_not_a_list"}, request=httpx.Request("GET", "https://example.com"))
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await ashby_src.fetch_job_by_ref(ref)

    assert res.status == FetchStatus.UPSTREAM_ERROR
    assert res.job is None
    assert "non-list" in (res.diagnostic or "").lower()


@pytest.mark.asyncio
async def test_mutation_wrong_base_url_fails() -> None:
    """Mutation proof 1: altered base URL must not succeed against standard endpoint."""
    with patch("job_mcp.sources.public.ashby.BOARDS_API_BASE", "https://invalid.example.com/wrong-path"):
        ashby_src = AshbySource()
        ref = JobRef(version=1, source_family="ashby", account="ashby", locator="some-id")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = httpx.Response(404, request=httpx.Request("GET", "https://invalid.example.com/wrong-path"))
            res = await ashby_src.fetch_job_by_ref(ref)
            called_url = mock_get.call_args[0][0]
            assert "wrong-path" in called_url
            assert res.status == FetchStatus.NOT_FOUND
