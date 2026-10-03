"""Mutation proof tests for Milestone 6-C deterministic discovery invariants.

Proves that mutations to budget limits, redirect policy, body caps, SSRF guards,
and classification rules fail explicitly.
"""

from __future__ import annotations

import httpx
import pytest

from job_mcp.core.search_plane.discovery import (
    MAX_SLUG_VARIANTS,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
    classify_direct_url,
    discover_company,
    to_registry_config,
)

# ===========================================================================
# Phase 26: 7 Budget Mutation Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_mutation_01_redirects_increased_above_3_fails() -> None:
    """Invariant: max redirects is strictly 3; a 4th redirect must not be followed."""
    hops_visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        hops_visited.append(url)
        if url == "https://company.com/0":
            return httpx.Response(302, headers={"Location": "https://company.com/1"})
        if url == "https://company.com/1":
            return httpx.Response(302, headers={"Location": "https://company.com/2"})
        if url == "https://company.com/2":
            return httpx.Response(302, headers={"Location": "https://company.com/3"})
        if url == "https://company.com/3":
            return httpx.Response(302, headers={"Location": "https://jobs.ashbyhq.com/hidden"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/0")
    res = await discover_company(target, client=client)

    # If MAX_REDIRECTS was mutated to 4, it would have visited hop 3 redirect target and confirmed ashby
    assert res.status == DiscoveryStatus.NOT_FOUND
    assert "https://jobs.ashbyhq.com/hidden" not in hops_visited


@pytest.mark.asyncio
async def test_mutation_02_256kib_limit_removed_fails() -> None:
    """Invariant: body inspection must be capped at 256 KiB; tokens beyond 256 KiB must be ignored."""
    padding = " " * (256 * 1024 + 100)
    oversized_body = f"<html><body>{padding}<a href='https://jobs.ashbyhq.com/oversized'>Link</a></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_body.encode("utf-8"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    # Must be NOT_FOUND because the only link is beyond 256 KiB
    assert res.status == DiscoveryStatus.NOT_FOUND
    assert any(e.kind == "BODY_TRUNCATED" for e in res.evidence)


@pytest.mark.asyncio
async def test_mutation_03_per_request_timeout_instead_of_global_budget_fails() -> None:
    """Invariant: budget is a global monotonic deadline, not per-request."""
    current_time = 0.0

    def mock_clock() -> float:
        return current_time

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_time, request_count
        request_count += 1
        current_time += 3.0  # Consumes 3s on each hop
        return httpx.Response(302, headers={"Location": f"https://company.com/hop{request_count}"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/hop0")
    # Total budget 5.0s. First request takes 3.0s, second request hits 6.0s > 5.0s deadline
    res = await discover_company(target, client=client, budget_seconds=5.0, clock=mock_clock)

    assert res.status == DiscoveryStatus.NOT_FOUND
    assert "budget exhausted" in (res.diagnostic or "").lower()
    # At most 2 requests could have fired before deadline was reached
    assert request_count <= 2


def test_mutation_04_candidate_loop_unbounded_fails() -> None:
    """Invariant: candidate slug generation is strictly bounded to MAX_SLUG_VARIANTS (<=2)."""
    assert MAX_SLUG_VARIANTS <= 2
    from job_mcp.core.search_plane.discovery.service import _generate_candidate_slugs

    slugs = _generate_candidate_slugs("Very Long Enterprise Multiword Technology Solutions Group, Inc.")
    assert len(slugs) <= 2


@pytest.mark.asyncio
async def test_mutation_05_redirect_cycle_protection_removed_fails() -> None:
    """Invariant: redirect cycles must be detected and aborted without looping."""
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        visited.append(url)
        if url == "https://company.com/cycle_a":
            return httpx.Response(302, headers={"Location": "https://company.com/cycle_b"})
        if url == "https://company.com/cycle_b":
            return httpx.Response(302, headers={"Location": "https://company.com/cycle_a"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/cycle_a")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.NOT_FOUND
    assert any(e.kind == "REDIRECT_CYCLE" for e in res.evidence)
    # Must terminate cleanly on cycle detection
    assert len(visited) == 2


@pytest.mark.asyncio
async def test_mutation_06_private_ip_redirect_allowed_fails() -> None:
    """Invariant: redirect to private IP addresses must be blocked before request execution."""
    private_target_requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal private_target_requested
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:8080/secret"})
        if "127.0.0.1" in url:
            private_target_requested = True
            return httpx.Response(200, text="Secret")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert private_target_requested is False
    assert res.status == DiscoveryStatus.NOT_FOUND
    assert any(e.kind == "BLOCKED_REDIRECT" for e in res.evidence)


@pytest.mark.asyncio
async def test_mutation_07_arbitrary_recursive_link_following_added_fails() -> None:
    """Invariant: discovery must NOT spider or recursively follow internal links."""
    spidered_links: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        spidered_links.append(url)
        if url == "https://company.com/careers":
            return httpx.Response(
                200,
                text="<html><body><a href='https://company.com/careers/page2'>Page 2</a></body></html>",
            )
        if url == "https://company.com/careers/page2":
            return httpx.Response(
                200,
                text="<html><body><a href='https://jobs.ashbyhq.com/hidden'>Ashby</a></body></html>",
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    # Page 2 must not be requested; discovery is bounded to target URL inspection
    assert "https://company.com/careers/page2" not in spidered_links
    assert res.status == DiscoveryStatus.NOT_FOUND


# ===========================================================================
# Phase 27: 8 Classification Mutation Tests
# ===========================================================================


def test_mutation_08_ashby_url_classified_as_workable_fails() -> None:
    """Invariant: Ashby URL must never be classified as Workable."""
    fam, _acc, _unsup, _err = classify_direct_url("https://jobs.ashbyhq.com/acme")
    assert fam == "ashby"
    assert fam != "workable"


def test_mutation_09_smartrecruiters_identifier_parsed_from_wrong_path_segment_fails() -> None:
    """Invariant: SmartRecruiters multi-segment posting URLs must not extract posting ID as company."""
    # A job posting URL like /acme/12345 must not extract 12345 as company identifier
    fam, _acc, _unsup, err = classify_direct_url("https://careers.smartrecruiters.com/acme/12345")
    assert fam is None
    assert err is not None


def test_mutation_10_workable_subdomain_derived_from_display_company_text_fails() -> None:
    """Invariant: Workable subdomain must be extracted from authoritative URL, not display text."""
    # In body inspection, 'Powered by Workable' alone must not synthesize an account subdomain
    html = "<html><body><h1>Careers</h1><p>Powered by Workable</p></body></html>"
    from job_mcp.core.search_plane.discovery.fingerprints import inspect_html_body

    cands, _unsup = inspect_html_body(html)
    assert len(cands) == 0


@pytest.mark.asyncio
async def test_mutation_11_first_candidate_wins_instead_of_ambiguous_fails() -> None:
    """Invariant: multiple distinct verified candidates must return AMBIGUOUS, not pick the first."""
    html = """
    <html><body>
    <a href="https://jobs.ashbyhq.com/acme">Ashby</a>
    <a href="https://apply.workable.com/acme/">Workable</a>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(200, text=html)
        if "ashbyhq.com" in url or "workable.com" in url:
            return httpx.Response(200, json={"jobs": [{"title": "Eng"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.AMBIGUOUS
    assert res.status != DiscoveryStatus.CONFIRMED
    assert len(res.candidates) == 2


@pytest.mark.asyncio
async def test_mutation_12_unsupported_page_reported_confirmed_fails() -> None:
    """Invariant: a page with unsupported ATS evidence must return UNSUPPORTED, never CONFIRMED."""
    html = "<html><body><a href='https://boards.greenhouse.io/acme'>Greenhouse</a></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    assert res.status == DiscoveryStatus.UNSUPPORTED
    assert res.status != DiscoveryStatus.CONFIRMED


@pytest.mark.asyncio
async def test_mutation_13_malformed_url_reported_not_found_fails() -> None:
    """Invariant: invalid input syntax or blocked schemes must return MALFORMED, not NOT_FOUND."""
    target = DiscoveryTarget(career_url="file:///etc/shadow")
    res = await discover_company(target)
    assert res.status == DiscoveryStatus.MALFORMED
    assert res.status != DiscoveryStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_mutation_14_verification_5xx_reported_confirmed_fails() -> None:
    """Invariant: an upstream HTTP 500 / 5xx on a probe must NOT confirm the candidate."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://company.com/careers":
            return httpx.Response(200, text="<a href='https://jobs.ashbyhq.com/servererror'>Jobs</a>")
        if "api.ashbyhq.com" in url:
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = DiscoveryTarget(career_url="https://company.com/careers")
    res = await discover_company(target, client=client)

    # 5xx must not confirm
    assert res.status != DiscoveryStatus.CONFIRMED
    assert res.status == DiscoveryStatus.NOT_FOUND


def test_mutation_15_failed_verification_silently_persists_config_fails() -> None:
    """Invariant: unverified or failed discovery results must refuse to persist configuration."""
    target = DiscoveryTarget(company_name="Failed Co")
    res = DiscoveryResult(status=DiscoveryStatus.NOT_FOUND, input=target)

    with pytest.raises(ValueError, match="Cannot project configuration for non-CONFIRMED"):
        to_registry_config(res)
