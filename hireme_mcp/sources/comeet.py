"""Comeet ATS direct job source implementation with concurrency control and caching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import time
from typing import Any, Optional

import httpx

from hireme_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode
from hireme_mcp.sources.base import BaseJobSource
from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComeetCompany:
    """Descriptor for a company using Comeet ATS."""

    uid: str
    name: str
    token: str


# Curated directory of leading Israeli tech companies using Comeet
DEFAULT_COMEET_COMPANIES: list[dict[str, str]] = [
    {
        "uid": "76.008",
        "name": "Comm-IT",
        "token": "67826D067833C0CF002D48020581368",
    },
]


def parse_comeet_position(raw: dict[str, Any], company_name: str) -> Job:
    """Parse raw Comeet API position dictionary into standardized Job model.

    Args:
        raw: Position payload dictionary from Comeet Careers API.
        company_name: Name of the company offering the position.

    Returns:
        Job: Normalized Job object with Comeet tagging and tech stack extraction.
    """
    uid_val = raw.get("position_uid") or raw.get("uid") or raw.get("name") or "unknown"
    job_id = f"comeet_{uid_val}"
    title = str(raw.get("name") or "Untitled")
    department = raw.get("department")

    url = (
        raw.get("url_comeet_hosted_page")
        or raw.get("url_recruit_hosted_page")
        or raw.get("url_active_page")
        or ""
    )
    apply_url = (
        raw.get("url_comeet_hosted_page")
        or raw.get("url_recruit_hosted_page")
        or ""
    )

    # Location parsing
    location_data = raw.get("location")
    location_str = ""
    is_remote = False

    if isinstance(location_data, dict):
        city = location_data.get("city")
        country = location_data.get("country") or location_data.get("name")
        is_remote = bool(location_data.get("is_remote", False))
        parts = [str(p) for p in [city, country] if p]
        if len(parts) == 2 and parts[0] == parts[1]:
            parts = [parts[0]]
        location_str = ", ".join(parts) if parts else str(location_data.get("name") or "")
    elif isinstance(location_data, str):
        location_str = location_data.strip()
        if "remote" in location_str.lower():
            is_remote = True

    # Work mode determination
    workplace_type = str(raw.get("workplace_type") or "").strip().lower()
    title_lower = title.lower()
    loc_lower = location_str.lower()

    if is_remote or workplace_type == "remote" or "remote" in title_lower or "remote" in loc_lower:
        work_mode = WorkMode.REMOTE
    elif workplace_type == "hybrid" or "hybrid" in title_lower or "hybrid" in loc_lower:
        work_mode = WorkMode.HYBRID
    elif workplace_type in ("on-site", "onsite") or "on-site" in title_lower or "onsite" in title_lower:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = WorkMode.ONSITE if location_str else None

    # Details / Description parsing
    details = raw.get("details")
    desc_parts: list[str] = []
    if isinstance(details, dict):
        for k in ("description", "requirements", "about", "value"):
            if k in details and details[k]:
                desc_parts.append(str(details[k]))
    elif isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and item.get("value"):
                desc_parts.append(str(item["value"]))
            elif isinstance(item, str) and item:
                desc_parts.append(item)
    elif isinstance(details, str) and details:
        desc_parts.append(details)

    if not desc_parts and raw.get("description"):
        desc_parts.append(str(raw["description"]))

    raw_description = "\n\n".join(desc_parts)
    clean_description = re.sub(r"<[^>]+>", " ", raw_description)
    clean_description = re.sub(r"\s+", " ", clean_description).strip()

    # Tech stack extraction
    search_text = f"{title} {clean_description} {department or ''}"
    tech_stack = _extract_text_tech_keywords(search_text)

    return Job(
        job_id=job_id,
        title=title,
        company=company_name,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=clean_description,
        url=url or None,
        apply_url=apply_url or None,
        department=department,
        source="comeet",
        sources=["comeet"],
    )


class ComeetSource(BaseJobSource):
    """Direct ATS job source querying Comeet Careers API for tech companies."""

    source_id: str = "comeet"
    display_name: str = "Comeet (Direct ATS)"
    description: str = "Direct career listings from tech companies using Comeet ATS"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        companies: Optional[list[dict[str, str] | ComeetCompany]] = None,
        max_concurrency: int = 5,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        """Initialize ComeetSource.

        Args:
            companies: Optional list of companies to query. Defaults to DEFAULT_COMEET_COMPANIES.
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
        self._companies: dict[str, ComeetCompany] = {}
        self._cache: dict[str, tuple[float, list[Job]]] = {}

        init_companies = companies if companies is not None else DEFAULT_COMEET_COMPANIES
        for comp in init_companies:
            if isinstance(comp, ComeetCompany):
                self.add_company(comp.uid, comp.name, comp.token)
            elif isinstance(comp, dict):
                self.add_company(comp["uid"], comp["name"], comp["token"])

    def add_company(self, uid: str, name: str, token: str) -> None:
        """Register a new company using Comeet ATS."""
        self._companies[uid] = ComeetCompany(uid=uid, name=name, token=token)

    def remove_company(self, uid: str) -> Optional[ComeetCompany]:
        """Remove a company from the directory."""
        self._cache.pop(uid, None)
        return self._companies.pop(uid, None)

    def get_companies(self) -> list[ComeetCompany]:
        """Return list of currently registered companies."""
        return list(self._companies.values())

    def clear_cache(self, uid: Optional[str] = None) -> None:
        """Clear cached positions for a specific company or all companies."""
        if uid:
            self._cache.pop(uid, None)
        else:
            self._cache.clear()

    async def _fetch_company_positions(
        self,
        company: ComeetCompany,
        client: httpx.AsyncClient,
    ) -> list[Job]:
        """Fetch positions for a single company with caching and semaphore throttling."""
        now = time.time()
        if company.uid in self._cache:
            ts, cached_jobs = self._cache[company.uid]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        async with self.semaphore:
            url = f"https://www.comeet.co/careers-api/2.0/company/{company.uid}/positions"
            params = {"token": company.token, "details": "true"}
            response = await client.get(url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(
                    "Comeet API error for %s (%s): HTTP %d",
                    company.name,
                    company.uid,
                    response.status_code,
                )
                return []

            data = response.json()
            positions_raw = data if isinstance(data, list) else data.get("positions", [])
            parsed_jobs: list[Job] = []
            for raw_pos in positions_raw:
                try:
                    job = parse_comeet_position(raw_pos, company_name=company.name)
                    parsed_jobs.append(job)
                except Exception as exc:
                    logger.warning(
                        "Error parsing Comeet position for company %s: %s",
                        company.name,
                        exc,
                    )

            self._cache[company.uid] = (now, parsed_jobs)
            return parsed_jobs

    async def _fetch_company_positions_safe(
        self,
        company: ComeetCompany,
        client: httpx.AsyncClient,
    ) -> list[Job]:
        """Wrap company fetch with exception safety and stale cache fallback."""
        try:
            return await self._fetch_company_positions(company, client)
        except Exception as exc:
            logger.warning(
                "Exception fetching Comeet positions for %s (%s): %s",
                company.name,
                company.uid,
                exc,
            )
            if company.uid in self._cache:
                return self._cache[company.uid][1]
            return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings matching optional preferences across all registered companies.

        Args:
            preferences: Optional JobPreferences filter.
            limit: Maximum number of jobs to retrieve.

        Returns:
            list[Job]: Standardized Job models.
        """
        if not self._companies:
            return []

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            tasks = [
                self._fetch_company_positions_safe(company, client)
                for company in self._companies.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_jobs: list[Job] = []
            for res in results:
                if isinstance(res, list):
                    all_jobs.extend(res)
                elif isinstance(res, Exception):
                    logger.warning("Error in Comeet fetch gather: %s", res)
        finally:
            if should_close_client:
                await client.aclose()

        # Tag source
        tagged_jobs: list[Job] = []
        for j in all_jobs:
            j.source = "comeet"
            if "comeet" not in j.sources:
                j.sources.insert(0, "comeet")
            tagged_jobs.append(j)

        # Preferences filtering
        if preferences:
            tagged_jobs = filter_jobs(tagged_jobs, preferences)

        # Truncate
        if limit and len(tagged_jobs) > limit:
            tagged_jobs = tagged_jobs[:limit]

        return tagged_jobs

    async def check_health(self) -> bool:
        """Check operational readiness of Comeet endpoint with primary company test query.

        Returns:
            bool: True if test endpoint returns HTTP 200, False otherwise.
        """
        test_company = self._companies.get("76.008")
        if not test_company and self._companies:
            test_company = next(iter(self._companies.values()))
        if not test_company:
            test_company = ComeetCompany(
                uid="76.008",
                name="Comm-IT",
                token="67826D067833C0CF002D48020581368",
            )

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            url = f"https://www.comeet.co/careers-api/2.0/company/{test_company.uid}/positions"
            params = {"token": test_company.token}
            response = await client.get(url, params=params, timeout=self.timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Comeet health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()
