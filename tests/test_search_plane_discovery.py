"""End-to-end integration and behavior tests for deterministic ATS discovery (Milestone 6-C)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from job_mcp.core.search_plane.discovery import (
    MAX_BODY_BYTES,
    DiscoveryStatus,
    DiscoveryTarget,
    classify_direct_url,
    discover_companies,
    discover_company,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "discovery"


# ===========================================================================
# 1. Direct URL Classification
# ===========================================================================


def test_classify_direct_url_ashby() -> None:
    """Ashby direct URLs extract clean board_name."""
    fam, acc, unsup, err = classify_direct_url("https://jobs.ashbyhq.com/example")
    assert fam == "ashby"
    assert acc == "example"
    assert unsup is None
    assert err is None

    # Trailing slash and query parameters
    fam, acc, _unsup, _err = classify_direct_url("https://jobs.ashbyhq.com/example/?source=web")
    assert fam == "ashby"
    assert acc == "example"

    # Multi-segment path fails
    fam, _acc, _unsup, err = classify_direct_url("https://jobs.ashbyhq.com/example/job/123")
    assert fam is None
    assert err is not None
    assert "multiple path segments" in err


def test_classify_direct_url_smartrecruiters() -> None:
    """SmartRecruiters direct URLs extract clean company_identifier."""
    fam, acc, unsup, err = classify_direct_url("https://careers.smartrecruiters.com/example")
    assert fam == "smartrecruiters"
    assert acc == "example"
    assert unsup is None
    assert err is None

    # jobs.smartrecruiters.com single segment
    fam, acc, _unsup, _err = classify_direct_url("https://jobs.smartrecruiters.com/example/")
    assert fam == "smartrecruiters"
    assert acc == "example"


def test_classify_direct_url_workable() -> None:
    """Workable apply URLs extract clean account_subdomain."""
    fam, acc, unsup, err = classify_direct_url("https://apply.workable.com/example/")
    assert fam == "workable"
    assert acc == "example"
    assert unsup is None
    assert err is None

    # Public API endpoint directly passed
    fam, acc, _unsup, _err = classify_direct_url("https://www.workable.com/api/accounts/example")
    assert fam == "workable"
    assert acc == "example"


def test_classify_direct_url_unsupported() -> None:
    """Greenhouse, Lever, and Workday direct URLs classify positively as UNSUPPORTED."""
    fam, _acc, unsup, _err = classify_direct_url("https://boards.greenhouse.io/acme")
    assert fam is None
    assert unsup == "greenhouse"

    _fam, _acc, unsup, _err = classify_direct_url("https://jobs.lever.co/acme")
    assert unsup == "lever"

    _fam, _acc, unsup, _err = classify_direct_url("https://acme.myworkdayjobs.com/careers")
    assert unsup == "workday"


# ===========================================================================
# 2. Redirect Inspection
# ===========================================================================


@pytest.mark.asyncio
async def test_redirect_to_ashby_confirmed() -> None:
    """A generic career URL redirecting to Ashby yields CONFIRMED."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(302, headers={"Location": "https://jobs.ashbyhq.com/acme"})
        if url == "https://api.ashbyhq.com/posting-api/job-board/acme":
            return httpx.Response(200, json={"jobs": [{"title": "Eng"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.CONFIRMED
    assert len(res.candidates) == 1
    assert res.candidates[0].source_family == "ashby"
    assert res.candidates[0].account == "acme"
    kinds = [e.kind for e in res.evidence]
    assert "REDIRECT" in kinds
    assert "REDIRECT_PROVIDER_URL" in kinds


@pytest.mark.asyncio
async def test_redirect_limit_max_3() -> None:
    """A chain of 4 redirects terminates and stops further redirect following."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/hop0":
            return httpx.Response(302, headers={"Location": "https://company.com/hop1"})
        if url == "https://company.com/hop1":
            return httpx.Response(302, headers={"Location": "https://company.com/hop2"})
        if url == "https://company.com/hop2":
            return httpx.Response(302, headers={"Location": "https://company.com/hop3"})
        if url == "https://company.com/hop3":
            # 4th redirect
            return httpx.Response(302, headers={"Location": "https://jobs.ashbyhq.com/acme"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/hop0")
    res = await discover_company(target, client=client)

    # 4th redirect is not followed; yields NOT_FOUND with REDIRECT_LIMIT
    assert res.status == DiscoveryStatus.NOT_FOUND
    kinds = [e.kind for e in res.evidence]
    assert "REDIRECT_LIMIT" in kinds


@pytest.mark.asyncio
async def test_redirect_cycle_detected() -> None:
    """A redirect cycle is detected and terminated safely without looping."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/a":
            return httpx.Response(302, headers={"Location": "https://company.com/b"})
        if url == "https://company.com/b":
            return httpx.Response(302, headers={"Location": "https://company.com/a"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/a")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.NOT_FOUND
    kinds = [e.kind for e in res.evidence]
    assert "REDIRECT_CYCLE" in kinds


@pytest.mark.asyncio
async def test_redirect_to_blocked_ip_prevented() -> None:
    """A redirect to a private IP (e.g. 10.0.0.1) is blocked before request execution."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(302, headers={"Location": "http://10.0.0.1/admin"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.NOT_FOUND
    kinds = [e.kind for e in res.evidence]
    assert "BLOCKED_REDIRECT" in kinds
    assert "blocked" in (res.diagnostic or "").lower()


# ===========================================================================
# 3. Body Fingerprint & HTML Inspection
# ===========================================================================


@pytest.mark.asyncio
async def test_body_fingerprint_ashby() -> None:
    """A generic career page embedding an Ashby link is discovered and verified."""
    html_content = (FIXTURES_DIR / "generic_to_ashby.html").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(200, text=html_content)
        if url == "https://api.ashbyhq.com/posting-api/job-board/acme":
            return httpx.Response(200, json={"jobs": [{"title": "Software Engineer"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.CONFIRMED
    assert len(res.candidates) == 1
    assert res.candidates[0].source_family == "ashby"
    assert res.candidates[0].account == "acme"


@pytest.mark.asyncio
async def test_body_fingerprint_unsupported_greenhouse() -> None:
    """A career page embedding a Greenhouse link reports UNSUPPORTED."""
    html_content = (FIXTURES_DIR / "unsupported_ats.html").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_content)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.UNSUPPORTED
    assert "greenhouse" in (res.diagnostic or "").lower()


@pytest.mark.asyncio
async def test_body_fingerprint_ambiguous() -> None:
    """A career page containing both Ashby and Workable links returns AMBIGUOUS."""
    html_content = (FIXTURES_DIR / "ambiguous_provider_links.html").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(200, text=html_content)
        if "ashbyhq.com" in url:
            return httpx.Response(200, json={"jobs": [{"title": "Eng"}]})
        if "workable.com" in url:
            return httpx.Response(200, json={"jobs": [{"title": "Eng"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.AMBIGUOUS
    assert len(res.candidates) == 2
    fams = [c.source_family for c in res.candidates]
    assert "ashby" in fams
    assert "workable" in fams


@pytest.mark.asyncio
async def test_oversized_body_truncated_at_256kib() -> None:
    """Inspection stops at 256 KiB; fingerprints placed beyond byte 262144 are ignored."""
    oversized_bytes = (FIXTURES_DIR / "oversized_body.html").read_bytes()
    assert len(oversized_bytes) > MAX_BODY_BYTES

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_bytes)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/large")
    res = await discover_company(target, client=client)

    # Hidden fingerprint beyond 256KiB was not read; returns NOT_FOUND
    assert res.status == DiscoveryStatus.NOT_FOUND
    kinds = [e.kind for e in res.evidence]
    assert "BODY_TRUNCATED" in kinds


# ===========================================================================
# 4. Company Name Discovery
# ===========================================================================


@pytest.mark.asyncio
async def test_company_name_existing_registry_match() -> None:
    """A company name matching an existing M5 registry entry returns CONFIRMED immediately."""
    target = DiscoveryTarget(company_name="Hugging Face")
    res = await discover_company(target)

    assert res.status == DiscoveryStatus.CONFIRMED
    assert len(res.candidates) == 1
    assert res.candidates[0].source_family == "workable"
    assert res.candidates[0].account == "huggingface"
    kinds = [e.kind for e in res.evidence]
    assert "REGISTRY_MATCH" in kinds


@pytest.mark.asyncio
async def test_company_name_candidate_slug_verification() -> None:
    """An unknown company generates bounded candidate slugs and confirms when one verifies."""
    request_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        request_urls.append(url)
        # Probe succeeds only for workable: example-ai
        if "apply.workable.com/api/v1/widget/accounts/example-ai" in url:
            return httpx.Response(200, json={"jobs": [{"title": "AI Engineer"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(company_name="Example AI, Inc.")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.CONFIRMED
    assert len(res.candidates) == 1
    assert res.candidates[0].source_family == "workable"
    assert res.candidates[0].account == "example-ai"

    # Request count must be bounded: at most 2 slugs * 3 providers = 6 requests
    assert len(request_urls) <= 6


@pytest.mark.asyncio
async def test_company_name_not_found() -> None:
    """An unknown company where all candidate probes fail returns NOT_FOUND."""
    request_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_urls.append(str(request.url))
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(company_name="Nonexistent Phantom Tech")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.NOT_FOUND
    assert len(res.candidates) == 0
    assert len(request_urls) <= 6


# ===========================================================================
# 5. Global Monotonic Budget
# ===========================================================================


@pytest.mark.asyncio
async def test_budget_exhaustion_terminates() -> None:
    """When monotonic deadline is exceeded, discovery stops immediately and returns NOT_FOUND."""
    current_time = 1000.0

    def mock_clock() -> float:
        nonlocal current_time
        return current_time

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_time
        # Advance time by 6.0 seconds on the first network call
        current_time += 6.0
        return httpx.Response(302, headers={"Location": "https://company.com/next"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/slow")
    res = await discover_company(target, client=client, budget_seconds=5.0, clock=mock_clock)

    assert res.status == DiscoveryStatus.NOT_FOUND
    assert "budget exhausted" in (res.diagnostic or "").lower()


# ===========================================================================
# 6. Batch Discovery
# ===========================================================================


@pytest.mark.asyncio
async def test_discover_companies_batch() -> None:
    """discover_companies preserves input ordering and isolates per-target errors."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "jobs.ashbyhq.com/alpha" in url or "ashbyhq.com/posting-api/job-board/alpha" in url:
            return httpx.Response(200, json={"jobs": [{"title": "Alpha"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    targets = [
        DiscoveryTarget(career_url="https://jobs.ashbyhq.com/alpha"),
        DiscoveryTarget(career_url="file:///etc/passwd"),  # Malformed
        DiscoveryTarget(company_name="Hugging Face"),      # Registry match
    ]

    results = await discover_companies(targets, client=client, max_concurrency=2)

    assert len(results) == 3
    # Preserves input order
    assert results[0].status == DiscoveryStatus.CONFIRMED
    assert results[0].candidates[0].account == "alpha"

    assert results[1].status == DiscoveryStatus.MALFORMED

    assert results[2].status == DiscoveryStatus.CONFIRMED
    assert results[2].candidates[0].account == "huggingface"
