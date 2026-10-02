"""Tests for Milestone 6 Unified Search Plane Aggregator Adapter (M6-A2).

Covers:
- Capability matrix mapping for existing 11 source families
- Search request mapping onto aggregator and preference filters
- Result conversion from canonical Job to JobSearchResultItem with opaque JobRef
- JobRef stability and deterministic generation
- Fetch resolution order:
  1. Invalid ref detection
  2. Cache fast path
  3. Native refetch (LinkedIn) with success, 404, and error handling
  4. Unsupported refetch for sources without native single-job endpoints
- Invariant: Zero synthetic dummy Job fabrication
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_mcp.core.api_client import JobCache
from job_mcp.core.search_plane.adapter import (
    SOURCE_CAPABILITY_MAP,
    SearchPlaneAdapter,
    get_source_capabilities,
)
from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
    JobSearchRequest,
    JobSearchResultItem,
    JobSearchResultSet,
)
from job_mcp.models.schemas import Job, WorkMode
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.authenticated.linkedin import LinkedInSource
from job_mcp.sources.registry import SourceRegistry

# ===========================================================================
# 1. Source Capabilities Mapping
# ===========================================================================


def test_capability_matrix_completeness() -> None:
    """All active source families are cataloged in the capability map."""
    expected_sources = {
        "linkedin",
        "greenhouse",
        "lever",
        "workday",
        "eightfold",
        "direct_tech",
        "comeet",
        "alljobs",
        "gotfriends",
        "jobify",
        "hiremetech",
        "ashby",
        "smartrecruiters",
    }
    assert set(SOURCE_CAPABILITY_MAP.keys()) == expected_sources


def test_linkedin_capabilities() -> None:
    caps = get_source_capabilities("linkedin")
    assert caps.supports_search is True
    assert caps.supports_native_fetch is True
    assert caps.supports_url_fetch is True
    assert caps.supports_query is True
    assert caps.supports_work_mode is True
    assert caps.supports_pagination is True


@pytest.mark.parametrize("src", ["greenhouse", "lever", "workday", "eightfold", "direct_tech", "comeet"])
def test_company_driven_sources_capabilities(src: str) -> None:
    caps = get_source_capabilities(src)
    assert caps.supports_search is True
    assert caps.supports_native_fetch is False
    assert caps.supports_company_filter is True


def test_unknown_source_capabilities_fallback() -> None:
    """Unknown source families fail closed with all capabilities false."""
    caps = get_source_capabilities("non_existent_family")
    assert caps.supports_search is False
    assert caps.supports_native_fetch is False
    assert caps.supports_url_fetch is False
    assert caps.supports_query is False
    assert caps.supports_company_filter is False
    assert caps.supports_work_mode is False
    assert caps.supports_pagination is False


# ===========================================================================
# 2. Search Request Mapping & Execution
# ===========================================================================


@pytest.mark.asyncio
async def test_search_request_maps_to_aggregator() -> None:
    """JobSearchRequest maps query, location, work_mode, and limit to aggregator."""
    mock_aggregator = MagicMock(spec=JobAggregator)
    sample_jobs = [
        Job(
            job_id="greenhouse_101",
            title="Senior Backend Engineer",
            company="JFrog",
            location="Tel Aviv",
            source="greenhouse",
            work_mode=WorkMode.HYBRID,
            url="https://boards.greenhouse.io/jfrog/jobs/101",
        ),
        Job(
            job_id="lever_202",
            title="Full Stack Engineer",
            company="Redis",
            location="Tel Aviv",
            source="lever",
            work_mode=WorkMode.REMOTE,
            url="https://jobs.lever.co/redis/202",
        ),
    ]
    mock_aggregator.fetch_all_jobs = AsyncMock(return_value=sample_jobs)

    adapter = SearchPlaneAdapter(aggregator=mock_aggregator)
    request = JobSearchRequest(
        query="Engineer",
        location="Tel Aviv",
        work_mode=WorkMode.HYBRID,
        limit=10,
        sources=["greenhouse", "lever"],
    )

    result_set = await adapter.search(request)

    # Verify aggregator was called with mapped preferences
    mock_aggregator.fetch_all_jobs.assert_awaited_once()
    call_kwargs = mock_aggregator.fetch_all_jobs.call_args.kwargs
    assert call_kwargs["sources"] == ["greenhouse", "lever"]
    prefs = call_kwargs["preferences"]
    assert prefs.keywords == ["Engineer"]
    assert prefs.location == "Tel Aviv"
    assert prefs.work_mode == WorkMode.HYBRID

    # Verify result set structure
    assert isinstance(result_set, JobSearchResultSet)
    assert len(result_set.items) == 2

    first = result_set.items[0]
    assert isinstance(first, JobSearchResultItem)
    assert first.title == "Senior Backend Engineer"
    assert first.company == "JFrog"
    assert first.source_family == "greenhouse"
    assert first.canonical_url == "https://boards.greenhouse.io/jfrog/jobs/101"

    # Verify opaque ref round-trips
    decoded_ref = JobRef.decode(first.ref)
    assert decoded_ref.source_family == "greenhouse"
    assert decoded_ref.account == "JFrog"
    assert decoded_ref.locator == "101"


@pytest.mark.asyncio
async def test_search_company_filter() -> None:
    """When request specifies company, results are filtered accordingly."""
    mock_aggregator = MagicMock(spec=JobAggregator)
    jobs = [
        Job(job_id="gh_1", title="Dev", company="JFrog", source="greenhouse"),
        Job(job_id="gh_2", title="Dev", company="Acme Corp", source="greenhouse"),
    ]
    mock_aggregator.fetch_all_jobs = AsyncMock(return_value=jobs)

    adapter = SearchPlaneAdapter(aggregator=mock_aggregator)
    res = await adapter.search(JobSearchRequest(company="jfrog"))

    assert len(res.items) == 1
    assert res.items[0].company == "JFrog"


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    mock_aggregator = MagicMock(spec=JobAggregator)
    jobs = [
        Job(job_id=f"j_{i}", title=f"Dev {i}", company="Acme", source="greenhouse")
        for i in range(10)
    ]
    mock_aggregator.fetch_all_jobs = AsyncMock(return_value=jobs)

    adapter = SearchPlaneAdapter(aggregator=mock_aggregator)
    res = await adapter.search(JobSearchRequest(limit=3))
    assert len(res.items) == 3


# ===========================================================================
# 3. JobRef Derivation & Stability
# ===========================================================================


def test_job_ref_stability_across_adapter_instances() -> None:
    """Identical jobs must produce the identical JobRef across separate adapter instances."""
    job = Job(
        job_id="workday_nvidia_JR19842",
        title="Hardware Engineer",
        company="NVIDIA",
        source="workday",
    )
    adapter1 = SearchPlaneAdapter(aggregator=MagicMock())
    adapter2 = SearchPlaneAdapter(aggregator=MagicMock())

    ref1 = adapter1.create_job_ref(job)
    ref2 = adapter2.create_job_ref(job)

    assert ref1 == ref2
    assert ref1.encode() == ref2.encode()
    assert ref1.source_family == "workday"
    assert ref1.account == "nvidia"
    assert ref1.locator == "JR19842"


def test_job_ref_derivation_linkedin() -> None:
    job = Job(
        job_id="linkedin_4152839402",
        title="ML Engineer",
        company="OpenAI",
        source="linkedin",
    )
    adapter = SearchPlaneAdapter(aggregator=MagicMock())
    ref = adapter.create_job_ref(job)
    assert ref.source_family == "linkedin"
    assert ref.account is None
    assert ref.locator == "4152839402"


# ===========================================================================
# 4. Fetch Adapter & Resolution Order
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_invalid_ref_string() -> None:
    adapter = SearchPlaneAdapter()
    result = await adapter.fetch("not_a_valid_ref")
    assert isinstance(result, FetchResult)
    assert result.status == FetchStatus.INVALID_REF
    assert result.job is None
    assert "Failed to decode" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_cache_fast_path() -> None:
    """When a job exists in JobCache, it is returned immediately as FOUND."""
    cache = JobCache(ttl_minutes=60)
    cached_job = Job(
        job_id="greenhouse_12345",
        title="Staff SRE",
        company="JFrog",
        location="Tel Aviv",
        source="greenhouse",
    )
    cache.update([cached_job])

    adapter = SearchPlaneAdapter(cache=cache)
    ref = JobRef(version=1, source_family="greenhouse", account="JFrog", locator="12345")

    result = await adapter.fetch(ref)
    assert result.status == FetchStatus.FOUND
    assert result.job is not None
    assert result.job.title == "Staff SRE"
    assert result.job.company == "JFrog"


@pytest.mark.asyncio
async def test_fetch_linkedin_native_success() -> None:
    """LinkedInSource.fetch_job_details is used when LinkedIn job is not in cache."""
    cache = JobCache(ttl_minutes=60)
    registry = SourceRegistry()
    mock_linkedin = MagicMock(spec=LinkedInSource)
    mock_linkedin.source_id = "linkedin"
    mock_linkedin.fetch_job_details = AsyncMock(
        return_value={
            "title": "Principal Architect",
            "company": "Microsoft",
            "location": "Herzliya",
            "description": "Lead cloud architecture",
            "work_mode": WorkMode.HYBRID,
            "posted_date": "2026-10-01",
            "apply_url": "https://careers.microsoft.com/123",
            "tech_stack": ["Azure", "Kubernetes"],
            "seniority_level": "Principal",
            "department": "Engineering",
            "requirements": "10+ years experience",
            "responsibilities": "Design systems",
            "company_overview": "Tech company",
        }
    )
    registry.register(mock_linkedin)

    adapter = SearchPlaneAdapter(registry=registry, cache=cache)
    ref = JobRef(version=1, source_family="linkedin", locator="99887766")

    result = await adapter.fetch(ref)
    assert result.status == FetchStatus.FOUND
    assert result.job is not None
    assert result.job.title == "Principal Architect"
    assert result.job.company == "Microsoft"
    assert result.job.work_mode == WorkMode.HYBRID
    assert result.job.tech_stack == ["Azure", "Kubernetes"]

    # Verify it was added to cache
    assert cache.get_by_id("linkedin_99887766") is not None


@pytest.mark.asyncio
async def test_fetch_linkedin_not_found() -> None:
    """When LinkedIn detail endpoint returns None, fetch reports NOT_FOUND."""
    registry = SourceRegistry()
    mock_linkedin = MagicMock(spec=LinkedInSource)
    mock_linkedin.source_id = "linkedin"
    mock_linkedin.fetch_job_details = AsyncMock(return_value=None)
    registry.register(mock_linkedin)

    adapter = SearchPlaneAdapter(registry=registry)
    ref = JobRef(version=1, source_family="linkedin", locator="00000000")

    result = await adapter.fetch(ref)
    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert "not found" in (result.diagnostic or "").lower()


@pytest.mark.asyncio
async def test_fetch_linkedin_upstream_error() -> None:
    """When LinkedIn detail endpoint raises network/HTTP error, fetch reports UPSTREAM_ERROR."""
    registry = SourceRegistry()
    mock_linkedin = MagicMock(spec=LinkedInSource)
    mock_linkedin.source_id = "linkedin"
    mock_linkedin.fetch_job_details = AsyncMock(side_effect=RuntimeError("Connection reset by peer"))
    registry.register(mock_linkedin)

    adapter = SearchPlaneAdapter(registry=registry)
    ref = JobRef(version=1, source_family="linkedin", locator="11223344")

    result = await adapter.fetch(ref)
    assert result.status == FetchStatus.UPSTREAM_ERROR
    assert result.job is None
    assert "Connection reset" in (result.diagnostic or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("src", ["greenhouse", "lever", "workday", "eightfold", "direct_tech", "gotfriends"])
async def test_fetch_unsupported_refetch_for_sources_without_native_endpoint(src: str) -> None:
    """Sources lacking native single-job fetch endpoints return UNSUPPORTED_REFETCH on cache miss."""
    adapter = SearchPlaneAdapter()
    ref = JobRef(version=1, source_family=src, account="somecorp", locator="pos_123")

    result = await adapter.fetch(ref)
    assert result.status == FetchStatus.UNSUPPORTED_REFETCH
    assert result.job is None
    assert "does not support native single-posting refetch" in (result.diagnostic or "")


@pytest.mark.asyncio
async def test_fetch_never_fabricates_dummy_placeholder_jobs() -> None:
    """The legacy fallback synthesizing Job(title='Job <id>', company='Unknown Company') must NEVER occur."""
    adapter = SearchPlaneAdapter()
    for src in ["greenhouse", "lever", "workday", "eightfold", "direct_tech", "linkedin", "alljobs"]:
        ref = JobRef(version=1, source_family=src, locator="fake_id")
        result = await adapter.fetch(ref)
        if result.status != FetchStatus.FOUND:
            assert result.job is None, f"Dummy job fabricated for {src} under status {result.status}"


@pytest.mark.asyncio
async def test_search_plane_adapter_does_not_double_deduplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """SearchPlaneAdapter relies on JobAggregator for dedup and does not invoke deduplicate_jobs."""
    import job_mcp.sources.aggregator as agg_module

    dedup_called = 0
    original_dedup = agg_module.deduplicate_jobs

    def spy_dedup(jobs: list[Job], *args: object, **kwargs: object) -> list[Job]:
        nonlocal dedup_called
        dedup_called += 1
        return original_dedup(jobs, *args, **kwargs)

    monkeypatch.setattr(agg_module, "deduplicate_jobs", spy_dedup)

    mock_aggregator = MagicMock(spec=JobAggregator)
    sample_jobs = [
        Job(job_id="gh_1", title="Dev", company="CompA", url="https://example.com/1"),
        Job(job_id="lv_1", title="Dev", company="CompA", url="https://example.com/2"),
    ]
    # Aggregator returns already-deduplicated canonical jobs
    mock_aggregator.fetch_all_jobs = AsyncMock(return_value=sample_jobs)

    adapter = SearchPlaneAdapter(aggregator=mock_aggregator)
    request = JobSearchRequest(query="Dev")
    result = await adapter.search(request)

    # SearchPlaneAdapter did NOT call deduplicate_jobs itself
    assert dedup_called == 0
    assert len(result.items) == 2


@pytest.mark.asyncio
async def test_aggregator_owns_single_dedup_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves JobAggregator owns the single authoritative dedup pass during search."""
    import job_mcp.sources.aggregator as agg_module

    dedup_calls = 0
    original_dedup = agg_module.deduplicate_jobs

    def spy_dedup(jobs: list[Job], *args: object, **kwargs: object) -> list[Job]:
        nonlocal dedup_calls
        dedup_calls += 1
        return original_dedup(jobs, *args, **kwargs)

    monkeypatch.setattr(agg_module, "deduplicate_jobs", spy_dedup)

    mock_source = MagicMock()
    mock_source.source_id = "mock_src"
    mock_source.fetch_jobs = AsyncMock(
        return_value=[
            Job(job_id="j1", title="Engineer", company="Alpha", url="https://a.com/1"),
            Job(job_id="j2", title="Engineer", company="Alpha", url="https://a.com/1"),  # duplicate URL
        ]
    )
    mock_registry = MagicMock(spec=SourceRegistry)
    mock_registry.get_active.return_value = [mock_source]

    aggregator = JobAggregator(registry=mock_registry, cache=None)
    adapter = SearchPlaneAdapter(aggregator=aggregator)

    result = await adapter.search(JobSearchRequest(query="Engineer"))
    # Exactly one dedup pass happened (owned by JobAggregator)
    assert dedup_calls == 1
    # Deduplication reduced 2 identical URLs to 1 canonical job
    assert len(result.items) == 1


@pytest.mark.asyncio
async def test_fetch_unknown_source_fails_closed() -> None:
    """Fetch on a ref from an unknown source family returns UNSUPPORTED_REFETCH on cache miss."""
    adapter = SearchPlaneAdapter()
    unknown_ref = JobRef(version=1, source_family="completely_unknown", account=None, locator="12345").encode()
    res = await adapter.fetch(unknown_ref)
    assert res.status == FetchStatus.UNSUPPORTED_REFETCH
    assert "does not support native single-posting refetch" in (res.diagnostic or "")
