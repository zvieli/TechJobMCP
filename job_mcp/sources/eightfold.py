"""Eightfold AI ATS direct job source implementation with concurrency control and caching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import re
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from job_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BaseJobSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EightfoldCompany:
    """Descriptor for a company using Eightfold AI ATS."""

    name: str
    hostname: str
    domain: str
    locations: list[str] = field(default_factory=list)
    filter_distance: Optional[str] = "16"
    enabled: bool = True

    def get_search_url(self) -> str:
        """Return the PCSX search endpoint URL."""
        return f"https://{self.hostname.rstrip('/')}/api/pcsx/search"

    def get_job_url(self, position_url: str) -> str:
        """Return the public apply / details URL for a job posting."""
        if not position_url:
            return f"https://{self.hostname.rstrip('/')}"
        if position_url.startswith("http://") or position_url.startswith("https://"):
            return position_url
        if not position_url.startswith("/"):
            position_url = f"/{position_url}"
        return f"https://{self.hostname.rstrip('/')}{position_url}"


# Curated directory of tech enterprises using Eightfold AI ATS
EIGHTFOLD_COMPANIES: dict[str, EightfoldCompany] = {
    "nvidia": EightfoldCompany(
        name="NVIDIA",
        hostname="nvidia.eightfold.ai",
        domain="nvidia.com",
        locations=["Yokne'am Illit", "Tel Aviv", "Israel"],
        filter_distance="16",
    ),
    "intel": EightfoldCompany(
        name="Intel",
        hostname="intel.eightfold.ai",
        domain="intel.com",
        locations=["Israel", "Haifa", "Petach Tikva", "Jerusalem"],
        filter_distance="16",
    ),
    "elbit_systems": EightfoldCompany(
        name="Elbit Systems",
        hostname="elbitsystems.eightfold.ai",
        domain="elbitsystems.com",
        locations=["Israel"],
        filter_distance="16",
    ),
    "micron": EightfoldCompany(
        name="Micron",
        hostname="micron.eightfold.ai",
        domain="micron.com",
        locations=[],
        filter_distance="16",
    ),
    "paypal": EightfoldCompany(
        name="PayPal",
        hostname="paypal.eightfold.ai",
        domain="paypal.com",
        locations=["Israel", "Tel Aviv"],
        filter_distance="16",
    ),
}

DEFAULT_EIGHTFOLD_COMPANIES: list[EightfoldCompany] = list(EIGHTFOLD_COMPANIES.values())


def parse_eightfold_position(raw: dict[str, Any], company: EightfoldCompany | str) -> Job:
    """Parse raw Eightfold AI PCSX API job posting dictionary into standardized Job model.

    Args:
        raw: Job posting dictionary from Eightfold AI PCSX API.
        company: EightfoldCompany instance or company name string.

    Returns:
        Job: Normalized Job object with Eightfold tagging and tech stack extraction.
    """
    if isinstance(company, EightfoldCompany):
        comp_name = company.name
        comp_slug = re.sub(r"[^a-zA-Z0-9]+", "_", comp_name).strip("_").lower()
    else:
        comp_name = str(company)
        comp_slug = re.sub(r"[^a-zA-Z0-9]+", "_", comp_name).strip("_").lower()

    # Extract raw job ID
    raw_id: Optional[str] = None
    for id_field in ("id", "position_id", "job_id", "req_id", "jobReqId", "display_job_id"):
        val = raw.get(id_field)
        if val is not None and str(val).strip():
            raw_id = str(val).strip()
            break

    position_url = str(
        raw.get("positionUrl")
        or raw.get("position_url")
        or raw.get("url")
        or raw.get("apply_url")
        or raw.get("canonical_url")
        or ""
    ).strip()

    if not raw_id and position_url:
        try:
            parsed_u = urlparse(position_url)
            qs = parse_qs(parsed_u.query)
            if "pid" in qs and qs["pid"]:
                raw_id = qs["pid"][0].strip()
            elif "job_id" in qs and qs["job_id"]:
                raw_id = qs["job_id"][0].strip()
        except Exception:
            pass

    if not raw_id and position_url:
        match = re.search(r"/(?:job|positions|careers)/([A-Za-z0-9_-]+)", position_url)
        if match:
            raw_id = match.group(1)
        else:
            raw_id = hashlib.md5(position_url.encode("utf-8")).hexdigest()[:8]

    if not raw_id:
        title_hint = str(raw.get("name") or raw.get("title") or "unknown")
        raw_id = hashlib.md5(f"{comp_name}_{title_hint}_{position_url}".encode("utf-8")).hexdigest()[:8]

    # Format unique job_id
    if raw_id.startswith("eightfold_"):
        job_id = raw_id
    else:
        job_id = f"eightfold_{comp_slug}_{raw_id}"

    title = str(raw.get("name") or raw.get("title") or raw.get("job_title") or "Untitled").strip()

    # Location parsing
    locations_raw = raw.get("locations")
    location_str = ""
    if isinstance(locations_raw, list) and locations_raw:
        loc_parts: list[str] = []
        for loc in locations_raw:
            if isinstance(loc, str) and loc.strip():
                loc_parts.append(loc.strip())
            elif isinstance(loc, dict):
                city = loc.get("city") or loc.get("name")
                country = loc.get("country")
                part = ", ".join([str(p) for p in (city, country) if p])
                if part:
                    loc_parts.append(part)
        location_str = "; ".join(loc_parts)
    elif isinstance(raw.get("location"), str):
        location_str = str(raw["location"]).strip()
    elif isinstance(raw.get("location"), dict):
        loc_dict = raw["location"]
        city = loc_dict.get("city") or loc_dict.get("name")
        country = loc_dict.get("country")
        location_str = ", ".join([str(p) for p in (city, country) if p])
    elif raw.get("locationsText"):
        location_str = str(raw["locationsText"]).strip()

    # Work mode determination
    workplace_type = str(
        raw.get("workplace_type")
        or raw.get("workplaceType")
        or raw.get("work_mode")
        or raw.get("work_type")
        or ""
    ).strip().lower()
    is_remote = bool(raw.get("is_remote") or raw.get("remote"))
    title_lower = title.lower()
    loc_lower = location_str.lower()

    if is_remote or "remote" in workplace_type or "remote" in title_lower or "remote" in loc_lower:
        work_mode = WorkMode.REMOTE
    elif "hybrid" in workplace_type or "hybrid" in title_lower or "hybrid" in loc_lower:
        work_mode = WorkMode.HYBRID
    elif (
        "on-site" in workplace_type
        or "onsite" in workplace_type
        or "on-site" in title_lower
        or "onsite" in title_lower
    ):
        work_mode = WorkMode.ONSITE
    else:
        work_mode = WorkMode.ONSITE if location_str else None

    # Description parsing
    raw_desc = str(
        raw.get("job_description")
        or raw.get("jobDescription")
        or raw.get("description")
        or raw.get("summary")
        or ""
    )
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Department
    department_raw = (
        raw.get("department")
        or raw.get("department_name")
        or raw.get("subfunction")
        or raw.get("category")
        or raw.get("job_category")
    )
    department: Optional[str] = None
    if isinstance(department_raw, dict):
        department = str(department_raw.get("name") or "").strip() or None
    elif department_raw:
        department = str(department_raw).strip() or None

    # Posted date
    posted_date = str(
        raw.get("posted_date")
        or raw.get("postedDate")
        or raw.get("posted_on")
        or raw.get("postedOn")
        or raw.get("created_at")
        or raw.get("createdAt")
        or raw.get("startDate")
        or ""
    ).strip() or None

    # URL resolution
    if isinstance(company, EightfoldCompany):
        url = company.get_job_url(position_url) if position_url else f"https://{company.hostname}"
    elif position_url.startswith("http://") or position_url.startswith("https://"):
        url = position_url
    elif position_url:
        url = f"https://{comp_slug}.eightfold.ai{position_url if position_url.startswith('/') else '/' + position_url}"
    else:
        url = None

    apply_url = str(raw.get("apply_url") or raw.get("applyUrl") or "").strip() or url

    # Tech stack extraction
    skills_raw = raw.get("skills") or raw.get("standard_skills") or raw.get("tags") or []
    skills_text = ""
    if isinstance(skills_raw, list):
        skills_text = " ".join(str(s) for s in skills_raw if s)

    search_text = f"{title} {clean_desc} {department or ''} {skills_text}".strip()
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
        department=department,
        source="eightfold",
        sources=["eightfold"],
    )


class EightfoldAISource(BaseJobSource):
    """Direct ATS job source querying Eightfold AI PCSX search APIs for enterprise tech companies."""

    source_id: str = "eightfold"
    display_name: str = "Eightfold AI (Direct ATS)"
    description: str = "Direct career listings from tech companies using Eightfold AI ATS"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        companies: Optional[
            list[dict[str, Any] | EightfoldCompany]
            | dict[str, dict[str, Any] | EightfoldCompany]
        ] = None,
        max_concurrency: int = 5,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        """Initialize EightfoldAISource.

        Args:
            companies: Optional list or dict of companies to query. Defaults to EIGHTFOLD_COMPANIES.
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
        self._companies: dict[str, EightfoldCompany] = {}
        self._cache: dict[str, tuple[float, list[Job]]] = {}

        if companies is None:
            init_companies = list(EIGHTFOLD_COMPANIES.values())
        elif isinstance(companies, dict):
            init_companies = list(companies.values())
        else:
            init_companies = list(companies)

        for comp in init_companies:
            if isinstance(comp, EightfoldCompany):
                self.add_company(comp)
            elif isinstance(comp, dict):
                locations = comp.get("locations")
                if locations is None:
                    if "efai_location" in comp:
                        locations = [comp["efai_location"]]
                    elif "location" in comp:
                        locations = [comp["location"]]
                    else:
                        locations = []
                self.add_company(
                    EightfoldCompany(
                        name=comp["name"],
                        hostname=comp.get("hostname") or comp.get("efai_hostname", ""),
                        domain=comp.get("domain") or comp.get("efai_domain", ""),
                        locations=locations,
                        filter_distance=comp.get("filter_distance", "16"),
                        enabled=comp.get("enabled", True),
                    )
                )

    def _get_company_key(self, company: EightfoldCompany | str) -> str:
        """Return unique dictionary lookup key for a company."""
        if isinstance(company, EightfoldCompany):
            return company.domain or company.hostname or company.name.lower()
        return str(company).strip().lower()

    def add_company(self, company: EightfoldCompany) -> None:
        """Register a new company using Eightfold AI ATS."""
        key = self._get_company_key(company)
        self._companies[key] = company

    def remove_company(self, company_identifier: str) -> Optional[EightfoldCompany]:
        """Remove a company from the directory by domain, hostname, or name."""
        key = company_identifier.strip().lower()
        if key in self._companies:
            self._cache.pop(key, None)
            return self._companies.pop(key)
        for k, comp in list(self._companies.items()):
            if (
                comp.domain.lower() == key
                or comp.hostname.lower() == key
                or comp.name.lower() == key
            ):
                self._cache.pop(k, None)
                return self._companies.pop(k)
        return None

    def get_companies(self) -> list[EightfoldCompany]:
        """Return list of currently registered companies."""
        return list(self._companies.values())

    def clear_cache(self, company_key: Optional[str] = None) -> None:
        """Clear cached positions for a specific company or all companies."""
        if company_key:
            key = company_key.strip().lower()
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    async def _fetch_company_positions(
        self,
        company: EightfoldCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch positions for a single Eightfold company with caching and semaphore throttling."""
        now = time.time()
        cache_key = self._get_company_key(company)
        has_keywords = bool(preferences and preferences.keywords)
        if not has_keywords and cache_key in self._cache:
            ts, cached_jobs = self._cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        async with self.semaphore:
            search_url = company.get_search_url()
            query_str = " ".join(preferences.keywords) if (preferences and preferences.keywords) else ""
            location_str = (
                preferences.location
                if (preferences and preferences.location)
                else (company.locations[0] if company.locations else "")
            )

            params: dict[str, Any] = {
                "domain": company.domain,
                "query": query_str,
                "location": location_str,
                "filter_distance": company.filter_distance or "16",
                "start": 0,
                "num": min(limit, 100),
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }

            t0 = time.perf_counter()
            response = await client.get(search_url, params=params, headers=headers, timeout=self.timeout)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code != 200:
                logger.warning(
                    "Eightfold PCSX API error for %s (%s): HTTP %d",
                    company.name,
                    company.hostname,
                    response.status_code,
                )
                return []

            data = response.json()
            if isinstance(data, dict):
                data_inner = data.get("data")
                if isinstance(data_inner, dict) and "positions" in data_inner:
                    postings_raw = data_inner.get("positions", [])
                else:
                    postings_raw = data.get("positions", [])
            elif isinstance(data, list):
                postings_raw = data
            else:
                postings_raw = []

            parsed_jobs: list[Job] = []
            for raw_pos in postings_raw:
                try:
                    job = parse_eightfold_position(raw_pos, company=company)
                    parsed_jobs.append(job)
                except Exception as exc:
                    logger.warning(
                        "Error parsing Eightfold position for %s: %s",
                        company.name,
                        exc,
                    )

            logger.info(
                "HTTP ATS request completed",
                url=search_url,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
                company=company.name,
                positions_count=len(parsed_jobs),
                source="eightfold",
            )

            if not has_keywords:
                self._cache[cache_key] = (now, parsed_jobs)

            return parsed_jobs

    async def _fetch_company_positions_safe(
        self,
        company: EightfoldCompany,
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
                "Exception fetching Eightfold positions for %s (%s): %s",
                company.name,
                company.hostname,
                exc,
            )
            cache_key = self._get_company_key(company)
            if cache_key in self._cache:
                return self._cache[cache_key][1]
            return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings matching optional preferences across all registered Eightfold companies."""
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
                    logger.warning("Error in Eightfold fetch gather: %s", res)
        finally:
            if should_close_client:
                await client.aclose()

        # Tag source
        tagged_jobs: list[Job] = []
        for j in all_jobs:
            j.source = "eightfold"
            if "eightfold" not in j.sources:
                j.sources.insert(0, "eightfold")
            tagged_jobs.append(j)

        # Preferences filtering
        if preferences:
            tagged_jobs = filter_jobs(tagged_jobs, preferences)

        # Truncate
        if limit and len(tagged_jobs) > limit:
            tagged_jobs = tagged_jobs[:limit]

        return tagged_jobs

    async def check_health(self) -> bool:
        """Check operational readiness of Eightfold endpoint with primary company test query."""
        test_company = (
            self._companies.get("nvidia.com")
            or (next(iter(self._companies.values())) if self._companies else None)
            or EIGHTFOLD_COMPANIES.get("nvidia")
        )
        if not test_company:
            return False

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            search_url = test_company.get_search_url()
            params = {
                "domain": test_company.domain,
                "query": "",
                "location": "",
                "start": 0,
                "num": 1,
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }
            response = await client.get(search_url, params=params, headers=headers, timeout=self.timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Eightfold health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()
