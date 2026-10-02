"""Ashby ATS direct job source — free public JSON API for modern tech companies."""

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
    ASHBY_COMPANIES as _BUILTIN_COMPANIES,
)
from job_mcp.sources.company_registry.entries import (
    AshbyCompany as RegistryAshbyCompany,
)
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = os.getenv("REPO_URL", "https://github.com/TechJobMCP/TechJobMCP")
BOARDS_API_BASE = "https://api.ashbyhq.com/posting-api/job-board"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": f"TechJobMCP/1.0 (Job Aggregator; +{REPO_URL})",
}
MAX_CONCURRENT_REQUESTS = 4

# Company descriptors and curated default catalog are owned by the
# configuration-driven company registry; re-exported for backward compatibility.
AshbyCompany = RegistryAshbyCompany
ASHBY_COMPANIES: dict[str, AshbyCompany] = _BUILTIN_COMPANIES


def _strip_html(raw_html: str) -> str:
    """Remove HTML tags, unescape entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(is_remote: bool, location: str, description: str) -> WorkMode:
    """Detect work mode from Ashby isRemote flag, location, and description text."""
    if is_remote:
        return WorkMode.REMOTE
    combined = f"{location} {description}".lower()
    if "remote" in combined:
        return WorkMode.REMOTE
    if "hybrid" in combined:
        return WorkMode.HYBRID
    return WorkMode.ONSITE


def parse_ashby_job(raw: dict[str, Any], company_name: str, board_name: str) -> Job:
    """Parse raw Ashby API job dictionary into standardized Job model."""
    raw_id = str(raw.get("id") or "unknown")
    job_id = f"ashby_{board_name}_{raw_id}"
    title = str(raw.get("title") or "Untitled")
    url = str(raw.get("jobUrl") or "")
    apply_url = str(raw.get("applyUrl") or url)

    # Location
    location_str = str(raw.get("location") or "")
    sec_locations = raw.get("secondaryLocations")
    if isinstance(sec_locations, list) and sec_locations:
        extra_locs = [str(loc) for loc in sec_locations if loc]
        if extra_locs and location_str:
            location_str = f"{location_str} (also: {', '.join(extra_locs)})"
        elif extra_locs:
            location_str = ", ".join(extra_locs)

    # Description
    content_html = str(raw.get("descriptionHtml") or "")
    description_plain = str(raw.get("descriptionPlain") or "")
    sections = parse_job_sections(content_html) if content_html else parse_job_sections(description_plain)
    description = description_plain if description_plain else _strip_html(content_html)

    # Department
    department = str(raw.get("department") or "") or None

    # Tech stack extraction
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        department=department,
        fallback_text=description,
    )

    # Work mode
    is_remote = bool(raw.get("isRemote", False))
    work_mode = _detect_work_mode(is_remote, location_str, description)

    # Published date
    posted_date = str(raw.get("publishedAt")) if raw.get("publishedAt") else None

    # Compensation extraction if present
    salary_range: str | None = None
    compensation = raw.get("compensation")
    if isinstance(compensation, dict):
        tiers = compensation.get("compensationTiers")
        if isinstance(tiers, list) and tiers:
            first_tier = tiers[0]
            if isinstance(first_tier, dict):
                components = first_tier.get("tierComponents")
                if isinstance(components, list) and components:
                    first_comp = components[0]
                    if isinstance(first_comp, dict):
                        curr = str(first_comp.get("currencyCode") or "").strip()
                        min_v = first_comp.get("minValue")
                        max_v = first_comp.get("maxValue")
                        s_min: int | None = None
                        s_max: int | None = None
                        if min_v is not None:
                            try:
                                s_min = int(min_v)
                            except (ValueError, TypeError):
                                pass
                        if max_v is not None:
                            try:
                                s_max = int(max_v)
                            except (ValueError, TypeError):
                                pass

                        if s_min is not None and s_max is not None:
                            salary_range = f"{s_min:,} - {s_max:,} {curr}".strip()
                        elif s_min is not None:
                            salary_range = f"{s_min:,}+ {curr}".strip()
                        elif s_max is not None:
                            salary_range = f"Up to {s_max:,} {curr}".strip()

    return Job(
        job_id=job_id,
        title=title,
        company=company_name,
        location=location_str,
        url=url,
        apply_url=apply_url,
        description=description or "",
        tech_stack=tech_stack,
        source="ashby",
        sources=["ashby"],
        work_mode=work_mode,
        department=department,
        posted_date=posted_date,
        salary_range=salary_range,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
    )


class AshbySource(BasePublicSource):
    """Job source aggregating roles from modern tech companies using Ashby ATS.

    Uses the free, unauthenticated Ashby Job Board API to fetch open positions.
    """

    source_id = "ashby"
    display_name = "Ashby"
    description = "Ashby ATS aggregator for tech companies (public unauthenticated API)"
    timeout: float = 15.0

    def __init__(self, companies: dict[str, AshbyCompany] | None = None) -> None:
        """Initialize AshbySource.

        Args:
            companies: Optional company catalog. Defaults to the effective
                company registry catalog for 'ashby'.
        """
        self._companies = (
            companies if companies is not None else registry_catalog("ashby")
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_company_jobs(
        self,
        client: httpx.AsyncClient,
        company: AshbyCompany,
    ) -> list[Job]:
        """Fetch all jobs for a single Ashby company board."""
        async with self._semaphore:
            url = f"{BOARDS_API_BASE}/{company.board_name}?includeCompensation=true"
            try:
                response = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    logger.warning("Ashby %s returned non-dict response", company.name)
                    return []
                raw_jobs = data.get("jobs")
                if not isinstance(raw_jobs, list):
                    logger.warning("Ashby %s returned non-list 'jobs'", company.name)
                    return []
                return [
                    parse_ashby_job(j, company.name, company.board_name)
                    for j in raw_jobs
                    if isinstance(j, dict)
                ]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug("Ashby board '%s' HTTP 404 (board not found or inactive)", company.board_name)
                else:
                    logger.warning("Ashby %s HTTP error: %s", company.name, e.response.status_code)
                return []
            except Exception as e:  # noqa: BLE001
                logger.debug("Ashby %s fetch error: %s", company.name, e)
                return []

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all enabled Ashby company boards."""
        enabled = {k: c for k, c in self._companies.items() if c.enabled}
        if not enabled:
            logger.info("No enabled Ashby companies configured.")
            return []

        logger.info("Fetching jobs from %d Ashby boards...", len(enabled))
        async with httpx.AsyncClient() as client:
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
                logger.error("Ashby board task failed unexpectedly: %s", result, exc_info=result)

        # Filter using candidate preferences if provided
        if preferences:
            all_jobs = filter_jobs(all_jobs, preferences, enable_semantic=False, enable_system1=False)

        self._last_fetch_time = time.time()
        logger.info("Ashby: fetched %d total jobs (limit=%d)", len(all_jobs), limit)
        return all_jobs[:limit]

    async def fetch_job_by_ref(self, ref: JobRef) -> FetchResult:
        """Fetch a specific Ashby posting by re-fetching its public board.

        Conforms to the Search Plane native refetch dispatch seam.
        """
        ref_str = str(ref)
        board_name = (ref.account or "").strip()
        target_locator = (ref.locator or "").strip()

        if ref.source_family != self.source_id:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic=f"Ashby cannot refetch job with source_family {ref.source_family!r}",
            )

        if not board_name or not target_locator:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic="Ashby ref must contain both account (board_name) and locator (job id)",
            )

        # Resolve company display name if configured
        matched_company = next(
            (c for c in self._companies.values() if c.board_name.lower() == board_name.lower()),
            None,
        )
        company_name = matched_company.name if matched_company else board_name

        url = f"{BOARDS_API_BASE}/{board_name}?includeCompensation=true"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                if resp.status_code == 404:
                    return FetchResult(
                        status=FetchStatus.NOT_FOUND,
                        ref=ref_str,
                        diagnostic=f"Ashby board {board_name!r} not found (HTTP 404)",
                    )
                resp.raise_for_status()
                data = resp.json()

            if not isinstance(data, dict):
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"Ashby board {board_name!r} returned malformed response",
                )

            raw_jobs = data.get("jobs")
            if not isinstance(raw_jobs, list):
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"Ashby board {board_name!r} returned non-list 'jobs'",
                )

            matching_raw = next(
                (j for j in raw_jobs if isinstance(j, dict) and str(j.get("id")) == target_locator),
                None,
            )
            if matching_raw is None:
                return FetchResult(
                    status=FetchStatus.NOT_FOUND,
                    ref=ref_str,
                    diagnostic=f"Posting {target_locator!r} was not found on Ashby board {board_name!r}",
                )

            job = parse_ashby_job(matching_raw, company_name, board_name)
            return FetchResult(
                status=FetchStatus.FOUND,
                job=job,
                ref=ref_str,
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Upstream HTTP error while fetching Ashby board {board_name!r}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Unexpected error while fetching Ashby posting: {exc}",
            )

    async def check_health(self) -> bool:
        """Check operational health by pinging the first enabled company board.

        Returns False if no enabled companies are configured to ping.
        """
        first_company = next(
            (c for c in self._companies.values() if c.enabled), None
        )
        if not first_company:
            return False
        try:
            async with httpx.AsyncClient() as client:
                url = f"{BOARDS_API_BASE}/{first_company.board_name}"
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=8.0)
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


__all__ = [
    "ASHBY_COMPANIES",
    "AshbyCompany",
    "AshbySource",
    "parse_ashby_job",
]
