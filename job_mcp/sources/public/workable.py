"""Workable ATS direct job source — public keyless Accounts API."""

from __future__ import annotations

import asyncio
import html
import os
import re
import time
from typing import Any

import httpx

from job_mcp.core.api_client import filter_jobs
from job_mcp.core.search_plane.models import FetchResult, FetchStatus, JobRef
from job_mcp.core.section_parser import extract_clean_job_tech_stack, parse_job_sections
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BasePublicSource
from job_mcp.sources.company_registry import catalog as registry_catalog
from job_mcp.sources.company_registry.defaults import (
    WORKABLE_COMPANIES as _BUILTIN_COMPANIES,
)
from job_mcp.sources.company_registry.entries import (
    WorkableCompany as RegistryWorkableCompany,
)
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = os.getenv("REPO_URL", "https://github.com/TechJobMCP/TechJobMCP")
ACCOUNTS_API_BASE = "https://www.workable.com/api/accounts"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": f"TechJobMCP/1.0 (Job Aggregator; +{REPO_URL})",
}
MAX_CONCURRENT_REQUESTS = 4

# Company descriptors and curated default catalog are owned by the
# configuration-driven company registry; re-exported for backward compatibility.
WorkableCompany = RegistryWorkableCompany
WORKABLE_COMPANIES: dict[str, WorkableCompany] = _BUILTIN_COMPANIES


def _strip_html(raw_html: str) -> str:
    """Remove HTML tags, unescape entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(
    telecommuting: bool,
    location_str: str,
    description: str,
) -> WorkMode | None:
    """Detect work mode from telecommuting flag and textual fallback.

    Returns None when neither telecommuting nor textual evidence is present.
    """
    if telecommuting:
        return WorkMode.REMOTE

    combined = f"{location_str} {description}".lower()
    if "remote" in combined:
        return WorkMode.REMOTE
    if "hybrid" in combined:
        return WorkMode.HYBRID
    if "on-site" in combined or "onsite" in combined:
        return WorkMode.ONSITE
    if location_str:
        return WorkMode.ONSITE
    return None


def parse_workable_job(
    raw: dict[str, Any], company_name: str, account_subdomain: str
) -> Job:
    """Parse raw Workable API job dictionary into standardized Job model."""
    raw_id = str(raw.get("shortcode") or raw.get("id") or "unknown").strip()
    job_id = f"workable_{account_subdomain}_{raw_id}"
    title = str(raw.get("title") or "Untitled").strip()

    # Authoritative URLs
    url = str(
        raw.get("url")
        or raw.get("shortlink")
        or f"https://apply.workable.com/{account_subdomain}/j/{raw_id}/"
    )
    apply_url = str(raw.get("application_url") or url)

    # Location parsing
    loc_parts = [
        str(raw.get("city") or "").strip(),
        str(raw.get("state") or "").strip(),
        str(raw.get("country") or "").strip(),
    ]
    location_str = ", ".join(p for p in loc_parts if p)
    if not location_str:
        locations = raw.get("locations")
        if isinstance(locations, list) and locations:
            loc_items = []
            for loc in locations:
                if isinstance(loc, dict):
                    p = [
                        str(loc.get("city") or "").strip(),
                        str(loc.get("region") or loc.get("state") or "").strip(),
                        str(loc.get("country") or "").strip(),
                    ]
                    loc_repr = ", ".join(x for x in p if x)
                    if loc_repr:
                        loc_items.append(loc_repr)
                elif isinstance(loc, str) and loc.strip():
                    loc_items.append(loc.strip())
            location_str = " | ".join(loc_items)

    # Description parsing (unclamped, preserves full HTML extraction)
    description_html = str(raw.get("description") or "")
    sections = parse_job_sections(description_html)
    description = _strip_html(description_html)

    # Department / Function
    department = str(raw.get("department") or raw.get("function") or "").strip() or None

    # Seniority level
    experience = str(raw.get("experience") or "").strip() or None

    # Tech stack extraction
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        department=department,
        fallback_text=description,
    )

    # Work mode
    telecommuting = bool(raw.get("telecommuting", False))
    work_mode = _detect_work_mode(telecommuting, location_str, description)

    # Posted date
    posted_date = str(raw.get("published_on") or raw.get("created_at") or "").strip() or None

    # Salary / Compensation
    salary_range: str | None = None
    if raw.get("salary"):
        salary_range = str(raw.get("salary")).strip() or None
    elif raw.get("salary_range"):
        salary_range = str(raw.get("salary_range")).strip() or None

    return Job(
        job_id=job_id,
        title=title,
        company=company_name,
        location=location_str,
        url=url,
        apply_url=apply_url,
        description=description,
        tech_stack=tech_stack,
        source="workable",
        sources=["workable"],
        work_mode=work_mode,
        department=department,
        seniority_level=experience,
        posted_date=posted_date,
        salary_range=salary_range,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
    )


class WorkableSource(BasePublicSource):
    """Job source aggregating roles from companies using Workable ATS.

    Uses the public unauthenticated Workable account jobs API to fetch open positions.
    """

    source_id = "workable"
    display_name = "Workable"
    description = "Workable ATS aggregator for tech companies (public unauthenticated API)"
    timeout: float = 15.0

    def __init__(self, companies: dict[str, WorkableCompany] | None = None) -> None:
        """Initialize WorkableSource.

        Args:
            companies: Optional company catalog. Defaults to the effective
                company registry catalog for 'workable'.
        """
        self._companies = (
            companies if companies is not None else registry_catalog("workable")
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_company_jobs(
        self,
        client: httpx.AsyncClient,
        company: WorkableCompany,
    ) -> list[Job]:
        """Fetch all jobs for a single Workable company account."""
        async with self._semaphore:
            url = f"{ACCOUNTS_API_BASE}/{company.account_subdomain}?details=true"
            try:
                response = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    logger.warning("Workable %s returned non-dict response", company.name)
                    return []
                raw_jobs = data.get("jobs")
                if not isinstance(raw_jobs, list):
                    logger.warning("Workable %s returned non-list 'jobs'", company.name)
                    return []
                return [
                    parse_workable_job(j, company.name, company.account_subdomain)
                    for j in raw_jobs
                    if isinstance(j, dict)
                ]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug(
                        "Workable account '%s' HTTP 404 (account not found or inactive)",
                        company.account_subdomain,
                    )
                else:
                    logger.warning("Workable %s HTTP error: %s", company.name, e.response.status_code)
                return []
            except Exception as e:  # noqa: BLE001
                logger.debug("Workable %s fetch error: %s", company.name, e)
                return []

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all enabled Workable company accounts."""
        enabled = {k: c for k, c in self._companies.items() if c.enabled}
        if not enabled:
            logger.info("No enabled Workable companies configured.")
            return []

        logger.info("Fetching jobs from %d Workable accounts...", len(enabled))
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [
                self._fetch_company_jobs(client, company)
                for company in enabled.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[Job] = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.error("Workable account task failed unexpectedly: %s", result, exc_info=result)

        # Filter using candidate preferences if provided
        if preferences:
            all_jobs = filter_jobs(all_jobs, preferences, enable_semantic=False, enable_system1=False)

        self._last_fetch_time = time.time()
        logger.info("Workable: fetched %d total jobs (limit=%d)", len(all_jobs), limit)
        return all_jobs[:limit]

    async def fetch_job_by_ref(self, ref: JobRef) -> FetchResult:
        """Fetch a specific Workable posting by re-fetching its public account.

        Conforms to the Search Plane native refetch dispatch seam.
        """
        ref_str = str(ref)
        account = (ref.account or "").strip()
        target_locator = (ref.locator or "").strip()

        if ref.source_family != self.source_id:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic=f"Workable cannot refetch job with source_family {ref.source_family!r}",
            )

        if not account or not target_locator:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic="Workable ref must contain both account (account_subdomain) and locator (shortcode)",
            )

        # Resolve company display name if configured
        matched_company = next(
            (c for c in self._companies.values() if c.account_subdomain.lower() == account.lower()),
            None,
        )
        company_name = matched_company.name if matched_company else account

        url = f"{ACCOUNTS_API_BASE}/{account}?details=true"
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                if resp.status_code == 404:
                    return FetchResult(
                        status=FetchStatus.NOT_FOUND,
                        ref=ref_str,
                        diagnostic=f"Workable account {account!r} not found (HTTP 404)",
                    )
                resp.raise_for_status()
                data = resp.json()

            if not isinstance(data, dict):
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"Workable account {account!r} returned malformed response",
                )

            raw_jobs = data.get("jobs")
            if not isinstance(raw_jobs, list):
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"Workable account {account!r} returned non-list 'jobs'",
                )

            matching_raw = next(
                (
                    j
                    for j in raw_jobs
                    if isinstance(j, dict)
                    and str(j.get("shortcode") or j.get("id") or "").strip() == target_locator
                ),
                None,
            )
            if matching_raw is None:
                return FetchResult(
                    status=FetchStatus.NOT_FOUND,
                    ref=ref_str,
                    diagnostic=f"Posting {target_locator!r} was not found on Workable account {account!r}",
                )

            job = parse_workable_job(matching_raw, company_name, account)
            return FetchResult(
                status=FetchStatus.FOUND,
                job=job,
                ref=ref_str,
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Upstream HTTP error while fetching Workable account {account!r}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Unexpected error while fetching Workable posting: {exc}",
            )

    async def check_health(self) -> bool:
        """Check operational health by pinging the first enabled company account.

        Returns False if no enabled companies are configured to ping.
        """
        first_company = next(
            (c for c in self._companies.values() if c.enabled), None
        )
        if not first_company:
            return False
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                url = f"{ACCOUNTS_API_BASE}/{first_company.account_subdomain}"
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=8.0)
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


__all__ = [
    "ACCOUNTS_API_BASE",
    "WORKABLE_COMPANIES",
    "WorkableCompany",
    "WorkableSource",
    "parse_workable_job",
]
