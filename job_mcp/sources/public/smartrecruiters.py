"""SmartRecruiters ATS direct job source — public keyless Posting API."""

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
    SMARTRECRUITERS_COMPANIES as _BUILTIN_COMPANIES,
)
from job_mcp.sources.company_registry.entries import (
    SmartRecruitersCompany as RegistrySmartRecruitersCompany,
)
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = os.getenv("REPO_URL", "https://github.com/TechJobMCP/TechJobMCP")
POSTINGS_API_BASE = "https://api.smartrecruiters.com/v1/companies"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": f"TechJobMCP/1.0 (Job Aggregator; +{REPO_URL})",
}
MAX_CONCURRENT_REQUESTS = 4

# Company descriptors and curated default catalog are owned by the
# configuration-driven company registry; re-exported for backward compatibility.
SmartRecruitersCompany = RegistrySmartRecruitersCompany
SMARTRECRUITERS_COMPANIES: dict[str, SmartRecruitersCompany] = _BUILTIN_COMPANIES


def _strip_html(raw_html: str) -> str:
    """Remove HTML tags, unescape entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(
    is_remote: bool,
    is_hybrid: bool,
    is_onsite: bool,
    location_str: str,
    description: str,
) -> WorkMode | None:
    """Detect work mode from structured location flags and textual fallback.

    Returns None when neither structured flags nor textual/location evidence is present.
    """
    if is_remote:
        return WorkMode.REMOTE
    if is_hybrid:
        return WorkMode.HYBRID
    if is_onsite:
        return WorkMode.ONSITE

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


def parse_smartrecruiters_job(
    raw: dict[str, Any], company_name: str, company_identifier: str
) -> Job:
    """Parse raw SmartRecruiters API job dictionary into standardized Job model."""
    raw_id = str(raw.get("id") or "unknown")
    job_id = f"smartrecruiters_{company_identifier}_{raw_id}"
    title = str(raw.get("name") or "Untitled")

    # Authoritative URLs
    url = f"https://jobs.smartrecruiters.com/{company_identifier}/{raw_id}"
    apply_url = str(raw.get("applyUrl") or url)

    # Location parsing
    location_str = ""
    loc = raw.get("location")
    is_remote = False
    is_hybrid = False
    is_onsite = False

    if isinstance(loc, dict):
        city = str(loc.get("city") or "").strip()
        region = str(loc.get("region") or loc.get("regionCode") or "").strip()
        country = str(loc.get("country") or loc.get("countryCode") or "").strip()
        loc_parts = [p for p in (city, region, country) if p]
        location_str = ", ".join(loc_parts)

        loc_type = str(loc.get("locationType") or "").upper().strip()
        is_remote = bool(loc.get("remote", False)) or loc_type == "REMOTE"
        is_hybrid = bool(loc.get("hybrid", False)) or loc_type == "HYBRID"
        is_onsite = loc_type == "ONSITE"

    # Description parsing from jobAd sections
    job_ad = raw.get("jobAd")
    sections_dict: dict[str, Any] = {}
    if isinstance(job_ad, dict):
        raw_sections = job_ad.get("sections")
        if isinstance(raw_sections, dict):
            sections_dict = raw_sections
        else:
            sections_dict = job_ad

    section_keys = [
        "companyDescription",
        "jobDescription",
        "qualifications",
        "additionalInformation",
    ]
    html_parts: list[str] = []
    for key in section_keys:
        sec_val = sections_dict.get(key)
        if isinstance(sec_val, dict):
            txt = sec_val.get("text")
            if isinstance(txt, str) and txt.strip():
                html_parts.append(txt.strip())
        elif isinstance(sec_val, str) and sec_val.strip():
            html_parts.append(sec_val.strip())

    combined_html = "\n\n".join(html_parts)
    description = _strip_html(combined_html) if combined_html else ""
    sections = parse_job_sections(combined_html) if combined_html else parse_job_sections(description)

    # Department / Function
    department: str | None = None
    dept_obj = raw.get("department")
    if isinstance(dept_obj, dict):
        department = str(dept_obj.get("label") or dept_obj.get("id") or "") or None
    elif isinstance(dept_obj, str) and dept_obj.strip():
        department = dept_obj.strip()

    # Tech stack extraction
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        department=department,
        fallback_text=description,
    )

    # Work mode
    work_mode = _detect_work_mode(
        is_remote, is_hybrid, is_onsite, location_str, description
    )

    # Released / Posted Date
    posted_date = str(raw.get("releasedDate")) if raw.get("releasedDate") else None

    # Compensation parsing if present
    salary_range: str | None = None
    compensation = raw.get("compensation")
    if isinstance(compensation, dict):
        min_v = compensation.get("min")
        max_v = compensation.get("max")
        curr = str(compensation.get("currency") or "").strip()
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

    # Use company display name from parameter or from raw posting
    raw_comp = raw.get("company")
    raw_name = str(raw_comp["name"]).strip() if isinstance(raw_comp, dict) and raw_comp.get("name") else ""
    resolved_company = company_name.strip() if company_name and company_name.strip() else (raw_name or company_identifier)

    return Job(
        job_id=job_id,
        title=title,
        company=resolved_company,
        location=location_str,
        url=url,
        apply_url=apply_url,
        description=description or "",
        tech_stack=tech_stack,
        source="smartrecruiters",
        sources=["smartrecruiters"],
        work_mode=work_mode,
        department=department,
        posted_date=posted_date,
        salary_range=salary_range,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
    )


class SmartRecruitersSource(BasePublicSource):
    """Job source aggregating roles from tech companies using SmartRecruiters ATS.

    Uses the public unauthenticated SmartRecruiters Posting API to search and retrieve postings.
    """

    source_id = "smartrecruiters"
    display_name = "SmartRecruiters"
    description = "SmartRecruiters ATS aggregator for tech companies (public Posting API)"
    timeout: float = 15.0

    def __init__(
        self, companies: dict[str, SmartRecruitersCompany] | None = None
    ) -> None:
        """Initialize SmartRecruitersSource.

        Args:
            companies: Optional company catalog. Defaults to the effective
                company registry catalog for 'smartrecruiters'.
        """
        self._companies = (
            companies
            if companies is not None
            else registry_catalog("smartrecruiters")
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_company_postings(
        self,
        client: httpx.AsyncClient,
        company: SmartRecruitersCompany,
        preferences: JobPreferences | None,
        limit: int,
    ) -> list[Job]:
        """Fetch job postings for a single SmartRecruiters company with bounded pagination."""
        async with self._semaphore:
            postings: list[Job] = []
            offset = 0
            page_size = min(max(limit, 10), 100)

            # Build query parameters
            query_param: str | None = None
            if preferences and preferences.keywords:
                q_tokens = [k.strip() for k in preferences.keywords if k.strip()]
                if q_tokens:
                    query_param = " ".join(q_tokens)

            while len(postings) < limit:
                params: dict[str, Any] = {
                    "destination": "PUBLIC",
                    "limit": page_size,
                    "offset": offset,
                }
                if query_param:
                    params["q"] = query_param

                url = f"{POSTINGS_API_BASE}/{company.company_identifier}/postings"
                try:
                    resp = await client.get(
                        url, params=params, headers=REQUEST_HEADERS, timeout=12.0
                    )
                    if resp.status_code == 404:
                        logger.debug(
                            "SmartRecruiters company %r not found (HTTP 404)",
                            company.company_identifier,
                        )
                        break
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "SmartRecruiters %s HTTP error: %s",
                        company.name,
                        exc.response.status_code,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "SmartRecruiters %s fetch error: %s", company.name, exc
                    )
                    break

                if not isinstance(data, dict):
                    logger.warning(
                        "SmartRecruiters %s returned non-dict response", company.name
                    )
                    break

                if "content" not in data or not isinstance(data["content"], list):
                    logger.warning(
                        "SmartRecruiters %s returned malformed response missing 'content' list: %r",
                        company.name,
                        data,
                    )
                    break

                content = data["content"]
                if not content:
                    # Legitimate empty page or no more postings
                    break

                total_found = data.get("totalFound")

                for item in content:
                    if isinstance(item, dict):
                        postings.append(
                            parse_smartrecruiters_job(
                                item, company.name, company.company_identifier
                            )
                        )
                        if len(postings) >= limit:
                            break

                # Advance offset
                offset += len(content)
                if (
                    total_found is not None
                    and isinstance(total_found, int)
                    and offset >= total_found
                ):
                    break

                # Safety bound on max pages per company
                if offset >= 1000:
                    break

            return postings

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all enabled SmartRecruiters companies."""
        enabled = {k: c for k, c in self._companies.items() if c.enabled}
        if not enabled:
            logger.info("No enabled SmartRecruiters companies configured.")
            return []

        logger.info(
            "Fetching jobs from %d SmartRecruiters companies...", len(enabled)
        )
        async with httpx.AsyncClient() as client:
            tasks = [
                self._fetch_company_postings(client, company, preferences, limit)
                for company in enabled.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[Job] = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.error(
                    "SmartRecruiters company task failed unexpectedly: %s",
                    result,
                    exc_info=result,
                )

        if preferences:
            all_jobs = filter_jobs(
                all_jobs, preferences, enable_semantic=False, enable_system1=False
            )

        self._last_fetch_time = time.time()
        logger.info(
            "SmartRecruiters: fetched %d total jobs (limit=%d)",
            len(all_jobs),
            limit,
        )
        return all_jobs[:limit]

    async def fetch_job_by_ref(self, ref: JobRef) -> FetchResult:
        """Fetch a specific SmartRecruiters posting via its native detail endpoint.

        Conforms to the Search Plane native refetch dispatch seam.
        Endpoint: GET /v1/companies/{companyIdentifier}/postings/{postingId}
        """
        ref_str = str(ref)
        company_identifier = (ref.account or "").strip()
        posting_id = (ref.locator or "").strip()

        if ref.source_family != self.source_id:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic=f"SmartRecruiters cannot refetch job with source_family {ref.source_family!r}",
            )

        if not company_identifier or not posting_id:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                ref=ref_str,
                diagnostic="SmartRecruiters ref must contain both account (companyIdentifier) and locator (postingId)",
            )

        matched_company = next(
            (
                c
                for c in self._companies.values()
                if c.company_identifier.lower() == company_identifier.lower()
            ),
            None,
        )
        company_name = (
            matched_company.name if matched_company else company_identifier
        )

        url = f"{POSTINGS_API_BASE}/{company_identifier}/postings/{posting_id}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                if resp.status_code == 404:
                    return FetchResult(
                        status=FetchStatus.NOT_FOUND,
                        ref=ref_str,
                        diagnostic=f"SmartRecruiters posting {posting_id!r} not found for company {company_identifier!r} (HTTP 404)",
                    )
                resp.raise_for_status()
                data = resp.json()

            if not isinstance(data, dict):
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"SmartRecruiters posting {posting_id!r} returned malformed non-dict response",
                )

            returned_ids = {
                str(data.get("id") or "").strip(),
                str(data.get("uuid") or "").strip(),
            }
            returned_ids.discard("")

            if not returned_ids:
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=f"SmartRecruiters posting detail response missing usable id/uuid for ref {posting_id!r}",
                )

            if posting_id not in returned_ids:
                return FetchResult(
                    status=FetchStatus.UPSTREAM_ERROR,
                    ref=ref_str,
                    diagnostic=(
                        f"SmartRecruiters posting detail identity mismatch: requested locator {posting_id!r} "
                        f"does not match response id/uuid {returned_ids!r}"
                    ),
                )

            job = parse_smartrecruiters_job(
                data, company_name, company_identifier
            )
            return FetchResult(
                status=FetchStatus.FOUND,
                job=job,
                ref=ref_str,
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Upstream HTTP error while fetching SmartRecruiters posting {posting_id!r}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                status=FetchStatus.UPSTREAM_ERROR,
                ref=ref_str,
                diagnostic=f"Unexpected error while fetching SmartRecruiters posting: {exc}",
            )

    async def check_health(self) -> bool:
        """Check operational health by pinging the first enabled company postings endpoint.

        Returns False if no enabled companies are configured.
        """
        first_company = next(
            (c for c in self._companies.values() if c.enabled), None
        )
        if not first_company:
            return False
        try:
            async with httpx.AsyncClient() as client:
                url = f"{POSTINGS_API_BASE}/{first_company.company_identifier}/postings"
                params = {"limit": 1, "destination": "PUBLIC"}
                resp = await client.get(
                    url, params=params, headers=REQUEST_HEADERS, timeout=8.0
                )
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


__all__ = [
    "SMARTRECRUITERS_COMPANIES",
    "SmartRecruitersCompany",
    "SmartRecruitersSource",
    "parse_smartrecruiters_job",
]
