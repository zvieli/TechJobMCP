"""Workday ATS direct job source implementation with concurrency control and caching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import re
import time
from typing import Any, Optional

import httpx

from job_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BaseJobSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WorkdayCompany:
    """Descriptor for a company using Workday ATS."""

    name: str
    wd_company: str
    wd_version: int = 1
    wd_suffix: str = "External"
    wd_locations: list[str] = field(default_factory=list)
    base_url: Optional[str] = None
    enabled: bool = True

    def get_base_url(self) -> str:
        """Return the base URL for the company's Workday portal."""
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"https://{self.wd_company}.wd{self.wd_version}.myworkdayjobs.com"

    def get_cxs_url(self) -> str:
        """Return the CXS search endpoint URL."""
        base = self.get_base_url()
        return f"{base}/wday/cxs/{self.wd_company}/{self.wd_suffix}/jobs"

    def get_job_url(self, external_path: str) -> str:
        """Return the public apply / details URL for a job posting."""
        base = self.get_base_url()
        if not external_path:
            return f"{base}/en-US/{self.wd_suffix}"
        if external_path.startswith("http://") or external_path.startswith("https://"):
            return external_path
        if not external_path.startswith("/"):
            external_path = f"/{external_path}"
        return f"{base}/en-US/{self.wd_suffix}{external_path}"


# Curated directory of tech enterprises using Workday ATS
WORKDAY_COMPANIES: dict[str, WorkdayCompany] = {
    "intel": WorkdayCompany(
        name="Intel",
        wd_company="intel",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=True,
    ),
    "nvidia": WorkdayCompany(
        name="NVIDIA",
        wd_company="nvidia",
        wd_version=5,
        wd_suffix="NVIDIAExternalCareerSite",
        wd_locations=[],
        enabled=True,
    ),
    "cisco": WorkdayCompany(
        name="Cisco",
        wd_company="cisco",
        wd_version=5,
        wd_suffix="Cisco_Careers",
        wd_locations=[],
        enabled=True,
    ),
    "philips": WorkdayCompany(
        name="Philips",
        wd_company="philips",
        wd_version=3,
        wd_suffix="jobs-and-careers",
        wd_locations=[],
        enabled=True,
    ),
    "dell": WorkdayCompany(
        name="Dell",
        wd_company="dell",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=True,
    ),
    "autodesk": WorkdayCompany(
        name="Autodesk",
        wd_company="autodesk",
        wd_version=1,
        wd_suffix="Ext",
        wd_locations=[],
        enabled=True,
    ),
    "microsoft": WorkdayCompany(
        name="Microsoft",
        wd_company="microsoft",
        wd_version=2,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # Microsoft uses careers.microsoft.com custom portal
    ),
    "qualcomm": WorkdayCompany(
        name="Qualcomm",
        wd_company="qualcomm",
        wd_version=5,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # Qualcomm Workday blocks direct automated POSTs
    ),
    "ptc": WorkdayCompany(
        name="PTC",
        wd_company="ptc",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # PTC endpoint changed
    ),
}

DEFAULT_WORKDAY_COMPANIES: list[WorkdayCompany] = list(WORKDAY_COMPANIES.values())


def parse_workday_position(raw: dict[str, Any], company: WorkdayCompany | str) -> Job:
    """Parse raw Workday CXS API job posting dictionary into standardized Job model.

    Args:
        raw: Job posting dictionary from Workday CXS API.
        company: WorkdayCompany instance or company name string.

    Returns:
        Job: Normalized Job object with Workday tagging and tech stack extraction.
    """
    if isinstance(company, WorkdayCompany):
        comp_name = company.name
        comp_slug = company.wd_company or re.sub(r"[^a-zA-Z0-9]+", "_", comp_name).strip("_").lower()
    else:
        comp_name = str(company)
        comp_slug = re.sub(r"[^a-zA-Z0-9]+", "_", comp_name).strip("_").lower()

    # Extract raw job ID
    bullet_fields = raw.get("bulletFields")
    raw_id: Optional[str] = None
    if isinstance(bullet_fields, list) and bullet_fields:
        for b in bullet_fields:
            if b and str(b).strip():
                raw_id = str(b).strip()
                break

    posting_info = raw.get("jobPostingInfo") if isinstance(raw.get("jobPostingInfo"), dict) else {}
    if not raw_id and posting_info:
        raw_id = posting_info.get("jobReqId") or posting_info.get("id")

    if not raw_id:
        raw_id = raw.get("jobReqId") or raw.get("id") or raw.get("jobPostingId")

    external_path = str(raw.get("externalPath") or "").strip()
    if not raw_id and external_path:
        match = re.search(r"([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|[0-9]{4,})", external_path)
        if match:
            raw_id = match.group(1)
        else:
            raw_id = hashlib.md5(external_path.encode("utf-8")).hexdigest()[:8]

    if not raw_id:
        title_hint = str(raw.get("title") or "unknown")
        raw_id = hashlib.md5(f"{comp_name}_{title_hint}_{external_path}".encode("utf-8")).hexdigest()[:8]

    # Format unique job_id
    if raw_id.startswith("workday_"):
        job_id = raw_id
    else:
        job_id = f"workday_{comp_slug}_{raw_id}"

    title = str(raw.get("title") or "Untitled").strip()

    # Location parsing
    location_str = str(
        raw.get("locationsText")
        or posting_info.get("location")
        or raw.get("location")
        or ""
    ).strip()

    # Work mode determination
    workplace_type = str(
        raw.get("workplaceType")
        or posting_info.get("workplaceType")
        or ""
    ).strip().lower()
    time_type = str(
        raw.get("timeType")
        or posting_info.get("timeType")
        or ""
    ).strip().lower()
    title_lower = title.lower()
    loc_lower = location_str.lower()

    if "remote" in workplace_type or "remote" in title_lower or "remote" in loc_lower:
        work_mode = WorkMode.REMOTE
    elif "hybrid" in workplace_type or "hybrid" in title_lower or "hybrid" in loc_lower:
        work_mode = WorkMode.HYBRID
    elif "on-site" in workplace_type or "onsite" in workplace_type or "on-site" in title_lower or "onsite" in title_lower:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = WorkMode.ONSITE if location_str else None

    # Description parsing
    raw_desc = str(
        posting_info.get("jobDescription")
        or raw.get("jobDescription")
        or raw.get("description")
        or raw.get("summary")
        or ""
    )
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Department / Subfunction
    department = (
        posting_info.get("subfunction")
        or posting_info.get("department")
        or raw.get("subfunction")
        or raw.get("department")
        or raw.get("jobCategory")
        or raw.get("category")
    )
    if department:
        department = str(department).strip()

    # Posted date
    posted_date = str(
        raw.get("postedOn")
        or posting_info.get("postedOn")
        or raw.get("startDate")
        or ""
    ).strip() or None

    # URL resolution
    if isinstance(company, WorkdayCompany):
        url = company.get_job_url(external_path)
    elif external_path.startswith("http://") or external_path.startswith("https://"):
        url = external_path
    elif external_path:
        url = f"https://{comp_slug}.myworkdayjobs.com/en-US/External{external_path if external_path.startswith('/') else '/' + external_path}"
    else:
        url = None

    apply_url = url

    # Tech stack extraction
    search_text = f"{title} {clean_desc} {department or ''}"
    tech_stack = _extract_text_tech_keywords(search_text)

    return Job(
        job_id=job_id,
        title=title,
        company=comp_name,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=clean_desc,
        posted_date=posted_date,
        url=url,
        apply_url=apply_url,
        department=department or None,
        source="workday",
        sources=["workday"],
    )


class WorkdaySource(BaseJobSource):
    """Direct ATS job source querying Workday CXS APIs for enterprise tech companies."""

    source_id: str = "workday"
    display_name: str = "Workday (Direct ATS)"
    description: str = "Direct career listings from tech companies using Workday ATS"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        companies: Optional[
            list[dict[str, Any] | WorkdayCompany]
            | dict[str, dict[str, Any] | WorkdayCompany]
        ] = None,
        max_concurrency: int = 5,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        """Initialize WorkdaySource.

        Args:
            companies: Optional list or dict of companies to query. Defaults to WORKDAY_COMPANIES.
            max_concurrency: Maximum number of concurrent HTTP requests (semaphore limit).
            cache_ttl_seconds: Time-to-live for per-company job caching in seconds.
            client: Optional shared httpx.AsyncClient instance.
            timeout: HTTP request timeout in seconds.
        """
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout = timeout
        self._client = client
        self._companies: dict[str, WorkdayCompany] = {}
        self._cache: dict[str, tuple[float, list[Job]]] = {}

        if companies is None:
            init_companies = list(WORKDAY_COMPANIES.values())
        elif isinstance(companies, dict):
            init_companies = list(companies.values())
        else:
            init_companies = list(companies)

        for comp in init_companies:
            if isinstance(comp, WorkdayCompany):
                self.add_company(comp)
            elif isinstance(comp, dict):
                self.add_company(
                    WorkdayCompany(
                        name=comp["name"],
                        wd_company=comp["wd_company"],
                        wd_version=comp.get("wd_version", 1),
                        wd_suffix=comp.get("wd_suffix", "External"),
                        wd_locations=comp.get("wd_locations", []),
                        base_url=comp.get("base_url"),
                        enabled=comp.get("enabled", True),
                    )
                )

    def add_company(self, company: WorkdayCompany) -> None:
        """Register a new company using Workday ATS."""
        self._companies[company.wd_company] = company

    def remove_company(self, wd_company: str) -> Optional[WorkdayCompany]:
        """Remove a company from the directory."""
        self._cache.pop(wd_company, None)
        return self._companies.pop(wd_company, None)

    def get_companies(self) -> list[WorkdayCompany]:
        """Return list of currently registered companies."""
        return list(self._companies.values())

    def clear_cache(self, wd_company: Optional[str] = None) -> None:
        """Clear cached positions for a specific company or all companies."""
        if wd_company:
            self._cache.pop(wd_company, None)
        else:
            self._cache.clear()

    async def _fetch_company_positions(
        self,
        company: WorkdayCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch positions for a single Workday company with caching and semaphore throttling."""
        now = time.time()
        cache_key = company.wd_company
        has_keywords = bool(preferences and preferences.keywords)
        if not has_keywords and cache_key in self._cache:
            ts, cached_jobs = self._cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        async with self.semaphore:
            cxs_url = company.get_cxs_url()
            
            search_text = ""
            if preferences and preferences.keywords:
                if len(preferences.keywords) > 2:
                    search_text = ""
                else:
                    joined = " ".join(preferences.keywords)
                    if len(joined) <= 30:
                        search_text = joined
                    else:
                        search_text = preferences.keywords[0][:30]

            applied_facets: dict[str, list[str]] = {}
            if company.wd_locations:
                applied_facets["locations"] = list(company.wd_locations)

            payload: dict[str, Any] = {
                "appliedFacets": applied_facets,
                "limit": min(limit, 50),
                "offset": 0,
                "searchText": search_text,
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }

            t0 = time.perf_counter()
            response = await client.post(cxs_url, json=payload, headers=headers, timeout=self.timeout)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code != 200:
                logger.warning(
                    "Workday CXS API error for %s (%s): HTTP %d",
                    company.name,
                    company.wd_company,
                    response.status_code,
                )
                return []

            data = response.json()
            postings_raw = data.get("jobPostings", []) if isinstance(data, dict) else []
            parsed_jobs: list[Job] = []
            for raw_pos in postings_raw:
                try:
                    job = parse_workday_position(raw_pos, company=company)
                    parsed_jobs.append(job)
                except Exception as exc:
                    logger.warning(
                        "Error parsing Workday position for %s: %s",
                        company.name,
                        exc,
                    )

            logger.info(
                "HTTP ATS request completed",
                url=cxs_url,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
                company=company.name,
                positions_count=len(parsed_jobs),
                source="workday",
            )

            if not has_keywords:
                self._cache[cache_key] = (now, parsed_jobs)

            return parsed_jobs

    async def _fetch_company_positions_safe(
        self,
        company: WorkdayCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Wrap company fetch with exception safety and stale cache fallback."""
        try:
            return await self._fetch_company_positions(
                company=company,
                client=client,
                preferences=preferences,
                limit=limit,
            )
        except Exception as exc:
            logger.warning(
                "Exception fetching Workday positions for %s (%s): %s",
                company.name,
                company.wd_company,
                exc,
            )
            if company.wd_company in self._cache:
                return self._cache[company.wd_company][1]
            return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings matching optional preferences across all registered Workday companies."""
        if not self._companies:
            return []

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            tasks = [
                self._fetch_company_positions_safe(
                    company=company,
                    client=client,
                    preferences=preferences,
                    limit=limit,
                )
                for company in self._companies.values()
                if company.enabled
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_jobs: list[Job] = []
            for res in results:
                if isinstance(res, list):
                    all_jobs.extend(res)
                elif isinstance(res, Exception):
                    logger.warning("Error in Workday fetch gather: %s", res)
        finally:
            if should_close_client:
                await client.aclose()

        # Tag source
        tagged_jobs: list[Job] = []
        for j in all_jobs:
            j.source = "workday"
            if "workday" not in j.sources:
                j.sources.insert(0, "workday")
            tagged_jobs.append(j)

        # Preferences filtering
        if preferences:
            tagged_jobs = filter_jobs(tagged_jobs, preferences)

        # Truncate
        if limit and len(tagged_jobs) > limit:
            tagged_jobs = tagged_jobs[:limit]

        return tagged_jobs

    async def check_health(self) -> bool:
        """Check operational readiness of Workday CXS endpoint with primary company test query."""
        test_company = (
            self._companies.get("cisco")
            or (next(iter(self._companies.values())) if self._companies else None)
            or WORKDAY_COMPANIES.get("cisco")
        )
        if not test_company:
            return False

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            cxs_url = test_company.get_cxs_url()
            payload = {
                "appliedFacets": {},
                "limit": 1,
                "offset": 0,
                "searchText": "",
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }
            response = await client.post(cxs_url, json=payload, headers=headers, timeout=self.timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Workday health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()
