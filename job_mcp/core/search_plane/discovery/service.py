"""Deterministic ATS and career board discovery engine (Milestone 6-C)."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from urllib.parse import urljoin

import httpx

from job_mcp.core.search_plane.discovery.fingerprints import (
    classify_direct_url,
    inspect_html_body,
)
from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)
from job_mcp.core.search_plane.discovery.probes import (
    verify_ashby_candidate,
    verify_smartrecruiters_candidate,
    verify_workable_candidate,
)
from job_mcp.core.search_plane.discovery.security import validate_url_safety
from job_mcp.sources.company_registry.core import CompanyRegistry, get_registry
from job_mcp.sources.company_registry.schema import (
    _require_path_token,
    _require_workable_subdomain,
)

MAX_REDIRECTS: int = 3
MAX_BODY_BYTES: int = 256 * 1024  # 262,144 bytes (256 KiB)
DEFAULT_DISCOVERY_BUDGET_SECONDS: float = 5.0
MAX_SLUG_VARIANTS: int = 2

# Deterministic provider ordering
PROVIDER_ORDER: dict[str, int] = {
    "ashby": 0,
    "smartrecruiters": 1,
    "workable": 2,
}

CORPORATE_SUFFIX_RE = re.compile(
    r"(?i)[,\s]+(?:inc|llc|ltd|corp|corporation|co|group|technologies|tech)\b\.?$"
)


def _generate_candidate_slugs(company_name: str) -> list[str]:
    """Generate a strictly bounded list of candidate account slugs (max 2)."""
    raw = company_name.strip()
    # Strip common corporate legal suffixes
    cleaned = CORPORATE_SUFFIX_RE.sub("", raw).strip()
    if not cleaned:
        cleaned = raw

    slugs: list[str] = []

    # Variant 1: Compact alphanumeric
    slug_compact = re.sub(r"[^a-zA-Z0-9]", "", cleaned).lower()
    if slug_compact:
        slugs.append(slug_compact)

    # Variant 2: Hyphenated
    slug_hyphen = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned).strip("-").lower()
    if slug_hyphen and slug_hyphen not in slugs:
        slugs.append(slug_hyphen)

    return slugs[:MAX_SLUG_VARIANTS]


async def _probe_candidate(
    family: str,
    account: str,
    client: httpx.AsyncClient,
    timeout: float,
) -> tuple[bool, DiscoveryEvidence | None]:
    """Dispatch verification probe for a specific provider family and account."""
    if timeout <= 0:
        return False, None
    if family == "ashby":
        return await verify_ashby_candidate(account, client, timeout)
    if family == "smartrecruiters":
        return await verify_smartrecruiters_candidate(account, client, timeout)
    if family == "workable":
        return await verify_workable_candidate(account, client, timeout)
    return False, None


async def discover_company(
    target: DiscoveryTarget,
    *,
    client: httpx.AsyncClient | None = None,
    budget_seconds: float = DEFAULT_DISCOVERY_BUDGET_SECONDS,
    dns_resolver: Callable[[str], list[str]] | None = None,
    clock: Callable[[], float] | None = None,
    registry: CompanyRegistry | None = None,
) -> DiscoveryResult:
    """Perform bounded, deterministic discovery for a single company target."""
    get_time = clock if clock is not None else time.monotonic
    deadline = get_time() + budget_seconds

    def remaining_budget() -> float:
        return max(0.0, deadline - get_time())

    # 1. Validate target inputs
    is_valid, err_msg = target.validate_input()
    if not is_valid:
        return DiscoveryResult(
            status=DiscoveryStatus.MALFORMED,
            input=target,
            diagnostic=err_msg,
        )

    # Own an HTTP client if not provided
    created_client = client is None
    http_client = (
        client
        if client is not None
        else httpx.AsyncClient(timeout=budget_seconds, follow_redirects=False)
    )

    try:
        # 2. Career URL Discovery Path
        if target.career_url:
            raw_url = target.career_url.strip()

            # SSRF check on input URL
            is_safe, sec_reason = validate_url_safety(raw_url, dns_resolver=dns_resolver)
            if not is_safe:
                return DiscoveryResult(
                    status=DiscoveryStatus.MALFORMED,
                    input=target,
                    diagnostic=f"Security check failed for career_url: {sec_reason}",
                )

            # Direct ATS URL classification
            fam, acc, unsup, class_err = classify_direct_url(raw_url)
            if class_err:
                return DiscoveryResult(
                    status=DiscoveryStatus.MALFORMED,
                    input=target,
                    diagnostic=class_err,
                )

            if unsup:
                ev = DiscoveryEvidence(
                    kind="DIRECT_PROVIDER_URL",
                    value=f"Unsupported ATS: {unsup}",
                    source_url=raw_url,
                )
                return DiscoveryResult(
                    status=DiscoveryStatus.UNSUPPORTED,
                    input=target,
                    evidence=(ev,),
                    diagnostic=f"Target URL is an unsupported ATS surface: {unsup}",
                )

            if fam and acc:
                ev = DiscoveryEvidence(
                    kind="DIRECT_PROVIDER_URL",
                    value=f"{fam}:{acc}",
                    source_url=raw_url,
                )
                # Verify candidate via probe
                rem = remaining_budget()
                v_ok, p_ev = await _probe_candidate(fam, acc, http_client, rem)
                evidences = (ev, p_ev) if (v_ok and p_ev) else (ev,)
                portal = DiscoveredPortal(
                    source_family=fam,
                    account=acc,
                    confidence_basis=evidences,
                )
                return DiscoveryResult(
                    status=DiscoveryStatus.CONFIRMED,
                    input=target,
                    candidates=(portal,),
                    evidence=evidences,
                )

            # Bounded redirect and body inspection
            evidence_list: list[DiscoveryEvidence] = []
            current_url = raw_url
            redirect_count = 0
            visited_urls = {current_url.lower().split("#")[0]}
            final_response = None

            while True:
                rem = remaining_budget()
                if rem <= 0:
                    return DiscoveryResult(
                        status=DiscoveryStatus.NOT_FOUND,
                        input=target,
                        evidence=tuple(evidence_list),
                        diagnostic="Discovery budget exhausted during redirect inspection",
                    )

                try:
                    req = http_client.build_request("GET", current_url)
                    resp = await http_client.send(req, stream=True)
                except (httpx.HTTPError, OSError) as exc:
                    return DiscoveryResult(
                        status=DiscoveryStatus.NOT_FOUND,
                        input=target,
                        evidence=tuple(evidence_list),
                        diagnostic=f"Network request failed for {current_url}: {exc}",
                    )

                if resp.is_redirect:
                    loc = resp.headers.get("Location")
                    await resp.aclose()
                    if not loc:
                        break

                    target_url = urljoin(current_url, loc)
                    redirect_count += 1
                    if redirect_count > MAX_REDIRECTS:
                        evidence_list.append(
                            DiscoveryEvidence(
                                kind="REDIRECT_LIMIT",
                                value="Maximum redirect budget (3) reached; 4th redirect stopped",
                                source_url=current_url,
                            )
                        )
                        break

                    # SSRF check on redirect target
                    is_safe_target, target_reason = validate_url_safety(
                        target_url, dns_resolver=dns_resolver
                    )
                    if not is_safe_target:
                        ev_block = DiscoveryEvidence(
                            kind="BLOCKED_REDIRECT",
                            value=f"Redirect to blocked target prevented: {target_reason}",
                            source_url=current_url,
                        )
                        evidence_list.append(ev_block)
                        return DiscoveryResult(
                            status=DiscoveryStatus.NOT_FOUND,
                            input=target,
                            evidence=tuple(evidence_list),
                            diagnostic=f"Redirect to blocked IP/host: {target_reason}",
                        )

                    # Cycle detection
                    norm_target = target_url.lower().split("#")[0]
                    if norm_target in visited_urls:
                        evidence_list.append(
                            DiscoveryEvidence(
                                kind="REDIRECT_CYCLE",
                                value=f"Redirect cycle detected to {target_url}",
                                source_url=current_url,
                            )
                        )
                        break
                    visited_urls.add(norm_target)

                    evidence_list.append(
                        DiscoveryEvidence(
                            kind="REDIRECT",
                            value=f"{current_url} -> {target_url}",
                            source_url=current_url,
                        )
                    )
                    current_url = target_url

                    # Check if redirect target is a direct ATS URL
                    fam_r, acc_r, unsup_r, _ = classify_direct_url(current_url)
                    if unsup_r:
                        ev_u = DiscoveryEvidence(
                            kind="REDIRECT_PROVIDER_URL",
                            value=f"Unsupported ATS: {unsup_r}",
                            source_url=current_url,
                        )
                        evidence_list.append(ev_u)
                        return DiscoveryResult(
                            status=DiscoveryStatus.UNSUPPORTED,
                            input=target,
                            evidence=tuple(evidence_list),
                            diagnostic=f"Redirected to unsupported ATS: {unsup_r}",
                        )
                    if fam_r and acc_r:
                        ev_r = DiscoveryEvidence(
                            kind="REDIRECT_PROVIDER_URL",
                            value=f"{fam_r}:{acc_r}",
                            source_url=current_url,
                        )
                        evidence_list.append(ev_r)
                        rem = remaining_budget()
                        v_ok, p_ev = await _probe_candidate(fam_r, acc_r, http_client, rem)
                        if v_ok and p_ev:
                            evidence_list.append(p_ev)
                        portal = DiscoveredPortal(
                            source_family=fam_r,
                            account=acc_r,
                            confidence_basis=tuple(evidence_list),
                        )
                        return DiscoveryResult(
                            status=DiscoveryStatus.CONFIRMED,
                            input=target,
                            candidates=(portal,),
                            evidence=tuple(evidence_list),
                        )
                    continue
                else:
                    final_response = resp
                    break

            if final_response is not None:
                # Read at most MAX_BODY_BYTES (256 KiB)
                body_bytes = b""
                truncated = False
                try:
                    async for chunk in final_response.aiter_bytes():
                        space_left = MAX_BODY_BYTES - len(body_bytes)
                        if len(chunk) > space_left:
                            body_bytes += chunk[:space_left]
                            truncated = True
                            break
                        body_bytes += chunk
                        if len(body_bytes) >= MAX_BODY_BYTES:
                            truncated = True
                            break
                finally:
                    await final_response.aclose()

                if truncated:
                    evidence_list.append(
                        DiscoveryEvidence(
                            kind="BODY_TRUNCATED",
                            value=f"body_truncated_at={MAX_BODY_BYTES}",
                            source_url=current_url,
                        )
                    )

                html_text = body_bytes.decode("utf-8", errors="replace")
                body_candidates, unsupported_fams = inspect_html_body(
                    html_text, source_url=current_url
                )

                if unsupported_fams and not body_candidates:
                    ev_u = DiscoveryEvidence(
                        kind="UNSUPPORTED_ATS_FINGERPRINT",
                        value=f"Detected: {', '.join(unsupported_fams)}",
                        source_url=current_url,
                    )
                    evidence_list.append(ev_u)
                    return DiscoveryResult(
                        status=DiscoveryStatus.UNSUPPORTED,
                        input=target,
                        evidence=tuple(evidence_list),
                        diagnostic=f"Unsupported ATS detected in page body: {', '.join(unsupported_fams)}",
                    )

                if body_candidates:
                    distinct_candidates: dict[tuple[str, str], list[DiscoveryEvidence]] = {}
                    for bf, ba, bev in body_candidates:
                        evidence_list.append(bev)
                        key = (bf, ba)
                        distinct_candidates.setdefault(key, []).append(bev)

                    # Probe candidates
                    verified_portals: list[DiscoveredPortal] = []
                    for (cand_fam, cand_acc), cand_evs in distinct_candidates.items():
                        rem = remaining_budget()
                        if rem <= 0:
                            break
                        v_ok, p_ev = await _probe_candidate(
                            cand_fam, cand_acc, http_client, rem
                        )
                        if v_ok and p_ev:
                            evidence_list.append(p_ev)
                            verified_portals.append(
                                DiscoveredPortal(
                                    source_family=cand_fam,
                                    account=cand_acc,
                                    confidence_basis=tuple(cand_evs + [p_ev]),
                                )
                            )

                    # Sort deterministically
                    verified_portals.sort(
                        key=lambda p: (PROVIDER_ORDER.get(p.source_family, 99), p.account)
                    )

                    if len(verified_portals) == 1:
                        return DiscoveryResult(
                            status=DiscoveryStatus.CONFIRMED,
                            input=target,
                            candidates=tuple(verified_portals),
                            evidence=tuple(evidence_list),
                        )
                    if len(verified_portals) > 1:
                        return DiscoveryResult(
                            status=DiscoveryStatus.AMBIGUOUS,
                            input=target,
                            candidates=tuple(verified_portals),
                            evidence=tuple(evidence_list),
                            diagnostic="Multiple distinct verified ATS portals found in career surface",
                        )

                    # No candidate verified via probe
                    if len(distinct_candidates) > 1:
                        unverified = [
                            DiscoveredPortal(
                                source_family=f,
                                account=a,
                                confidence_basis=tuple(e),
                            )
                            for (f, a), e in distinct_candidates.items()
                        ]
                        unverified.sort(
                            key=lambda p: (PROVIDER_ORDER.get(p.source_family, 99), p.account)
                        )
                        return DiscoveryResult(
                            status=DiscoveryStatus.AMBIGUOUS,
                            input=target,
                            candidates=tuple(unverified),
                            evidence=tuple(evidence_list),
                            diagnostic="Multiple unverified ATS candidates found in body",
                        )

                    return DiscoveryResult(
                        status=DiscoveryStatus.NOT_FOUND,
                        input=target,
                        evidence=tuple(evidence_list),
                        diagnostic="ATS candidate found in body failed verification probe",
                    )

            # If URL discovery did not yield an ATS, check if company_name was also supplied
            if not target.company_name:
                return DiscoveryResult(
                    status=DiscoveryStatus.NOT_FOUND,
                    input=target,
                    evidence=tuple(evidence_list),
                    diagnostic="No supported ATS portal established from career URL",
                )

        # 3. Company Name Discovery Path
        if target.company_name:
            c_name = target.company_name.strip()
            norm_name = c_name.lower()

            # Step 1: Check existing CompanyRegistry
            active_registry = (
                registry if registry is not None else get_registry()
            )
            for fam in ("ashby", "smartrecruiters", "workable"):
                catalog = active_registry.catalog(fam)
                for entry_id, entry in catalog.items():
                    if entry.name.strip().lower() == norm_name or entry_id.lower() == norm_name:
                        coord = ""
                        if fam == "ashby":
                            coord = getattr(entry, "board_name", "")
                        elif fam == "smartrecruiters":
                            coord = getattr(entry, "company_identifier", "")
                        elif fam == "workable":
                            coord = getattr(entry, "account_subdomain", "")

                        ev = DiscoveryEvidence(
                            kind="REGISTRY_MATCH",
                            value=f"Matched existing registry company '{entry.name}' (id: '{entry_id}') for provider '{fam}'",
                        )
                        portal = DiscoveredPortal(
                            source_family=fam,
                            account=coord,
                            confidence_basis=(ev,),
                        )
                        return DiscoveryResult(
                            status=DiscoveryStatus.CONFIRMED,
                            input=target,
                            candidates=(portal,),
                            evidence=(ev,),
                        )

            # Step 2: Deterministic candidate slug generation
            slugs = _generate_candidate_slugs(c_name)
            evidence_list: list[DiscoveryEvidence] = []
            verified_portals: list[DiscoveredPortal] = []

            for slug in slugs:
                rem = remaining_budget()
                if rem <= 0:
                    break

                for fam in ("ashby", "smartrecruiters", "workable"):
                    rem = remaining_budget()
                    if rem <= 0:
                        break

                    # Validate slug format for provider before probing
                    try:
                        if fam in ("ashby", "smartrecruiters"):
                            _require_path_token(slug, "slug")
                        elif fam == "workable":
                            _require_workable_subdomain(slug, "slug")
                    except ValueError:
                        continue

                    v_ok, p_ev = await _probe_candidate(fam, slug, http_client, rem)
                    if v_ok and p_ev:
                        evidence_list.append(p_ev)
                        verified_portals.append(
                            DiscoveredPortal(
                                source_family=fam,
                                account=slug,
                                confidence_basis=(p_ev,),
                            )
                        )

            # Sort deterministically
            verified_portals.sort(
                key=lambda p: (PROVIDER_ORDER.get(p.source_family, 99), p.account)
            )

            if len(verified_portals) == 1:
                return DiscoveryResult(
                    status=DiscoveryStatus.CONFIRMED,
                    input=target,
                    candidates=tuple(verified_portals),
                    evidence=tuple(evidence_list),
                )
            if len(verified_portals) > 1:
                return DiscoveryResult(
                    status=DiscoveryStatus.AMBIGUOUS,
                    input=target,
                    candidates=tuple(verified_portals),
                    evidence=tuple(evidence_list),
                    diagnostic="Multiple provider candidate probes verified for company name",
                )

            return DiscoveryResult(
                status=DiscoveryStatus.NOT_FOUND,
                input=target,
                evidence=tuple(evidence_list),
                diagnostic="No candidate provider probes succeeded for company name",
            )

        return DiscoveryResult(
            status=DiscoveryStatus.NOT_FOUND,
            input=target,
            diagnostic="No valid discovery branch could be executed",
        )

    finally:
        if created_client:
            await http_client.aclose()


async def discover_companies(
    targets: Sequence[DiscoveryTarget],
    *,
    client: httpx.AsyncClient | None = None,
    budget_seconds: float = DEFAULT_DISCOVERY_BUDGET_SECONDS,
    max_concurrency: int = 5,
    dns_resolver: Callable[[str], list[str]] | None = None,
    clock: Callable[[], float] | None = None,
    registry: CompanyRegistry | None = None,
) -> list[DiscoveryResult]:
    """Perform bounded discovery across a batch of targets with bounded concurrency."""
    created_client = client is None
    http_client = (
        client
        if client is not None
        else httpx.AsyncClient(timeout=budget_seconds, follow_redirects=False)
    )

    sem = asyncio.Semaphore(max_concurrency)

    async def _bound_discover(t: DiscoveryTarget) -> DiscoveryResult:
        async with sem:
            return await discover_company(
                t,
                client=http_client,
                budget_seconds=budget_seconds,
                dns_resolver=dns_resolver,
                clock=clock,
                registry=registry,
            )

    try:
        results = await asyncio.gather(*[_bound_discover(t) for t in targets])
        return list(results)
    finally:
        if created_client:
            await http_client.aclose()
