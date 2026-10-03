"""Zero-credential provider verification probes for Milestone 6-C discovery."""

from __future__ import annotations

import httpx

from job_mcp.core.search_plane.discovery.models import DiscoveryEvidence


async def verify_ashby_candidate(
    board_name: str,
    client: httpx.AsyncClient,
    timeout: float,
) -> tuple[bool, DiscoveryEvidence | None]:
    """Verify an Ashby candidate board slug using the public posting API."""
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_name}"
    try:
        res = await client.get(url, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and ("jobs" in data or "jobBoard" in data):
                ev = DiscoveryEvidence(
                    kind="VERIFICATION_PROBE",
                    value=f"ashby: {board_name} confirmed (HTTP 200)",
                    source_url=url,
                )
                return True, ev
    except (httpx.HTTPError, ValueError):
        return False, None
    return False, None


async def verify_smartrecruiters_candidate(
    company_identifier: str,
    client: httpx.AsyncClient,
    timeout: float,
) -> tuple[bool, DiscoveryEvidence | None]:
    """Verify a SmartRecruiters candidate company identifier using the public postings API."""
    url = f"https://api.smartrecruiters.com/v1/companies/{company_identifier}/postings?destination=PUBLIC&limit=1"
    try:
        res = await client.get(url, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "content" in data:
                ev = DiscoveryEvidence(
                    kind="VERIFICATION_PROBE",
                    value=f"smartrecruiters: {company_identifier} confirmed (HTTP 200)",
                    source_url=url,
                )
                return True, ev
    except (httpx.HTTPError, ValueError):
        return False, None
    return False, None


async def verify_workable_candidate(
    account_subdomain: str,
    client: httpx.AsyncClient,
    timeout: float,
) -> tuple[bool, DiscoveryEvidence | None]:
    """Verify a Workable candidate account subdomain using the public widget/account API."""
    # Workable redirects from workable.com/api/accounts/{acc}?details=true to apply.workable.com/api/v1/widget/accounts/{acc}?details=true
    url = f"https://apply.workable.com/api/v1/widget/accounts/{account_subdomain}?details=true"
    try:
        res = await client.get(url, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "jobs" in data:
                ev = DiscoveryEvidence(
                    kind="VERIFICATION_PROBE",
                    value=f"workable: {account_subdomain} confirmed (HTTP 200)",
                    source_url=url,
                )
                return True, ev
    except (httpx.HTTPError, ValueError):
        return False, None
    return False, None
