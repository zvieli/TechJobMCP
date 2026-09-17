"""Greenhouse ATS direct job source — free public JSON API for Israeli AI startups."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import html
import re
import time
from typing import Any, Optional

import httpx

from job_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BasePublicSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

BOARDS_API_BASE = "https://boards-api.greenhouse.io/v1/boards"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "TechJobMCP/1.0 (Job Aggregator; +https://github.com/zvieli/TechJobMCP)",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}
MAX_CONCURRENT_REQUESTS = 4


@dataclass
class GreenhouseCompany:
    """Descriptor for a company using Greenhouse ATS."""

    name: str
    board_token: str
    enabled: bool = True


# Curated directory of Israeli AI/tech companies using Greenhouse
GREENHOUSE_COMPANIES: dict[str, GreenhouseCompany] = {
    "ai21labs": GreenhouseCompany(name="AI21 Labs", board_token="ai21labs"),
    "lightricks": GreenhouseCompany(name="Lightricks", board_token="lightricks"),
    "tabnine": GreenhouseCompany(name="Tabnine", board_token="tabnine"),
    "bria": GreenhouseCompany(name="Bria AI", board_token="baborstudio"),
    "gong": GreenhouseCompany(name="Gong", board_token="gong"),
    "wiz": GreenhouseCompany(name="Wiz", board_token="wiz"),
    "appsflyer": GreenhouseCompany(name="AppsFlyer", board_token="appsflyer"),
    "orcaai": GreenhouseCompany(name="Orca AI", board_token="orcaai"),
    "lemonade": GreenhouseCompany(name="Lemonade", board_token="lemonade"),
    "monday": GreenhouseCompany(name="monday.com", board_token="mondaydotcom"),
    "fiverr": GreenhouseCompany(name="Fiverr", board_token="fiverr"),
    "deepchecks": GreenhouseCompany(name="Deepchecks", board_token="deepchecks"),
    "datadog": GreenhouseCompany(name="Datadog Israel", board_token="datadog"),
    "snyk": GreenhouseCompany(name="Snyk", board_token="snyk"),
    "jfrog": GreenhouseCompany(name="JFrog", board_token="jfrog"),
    "orca_security": GreenhouseCompany(name="Orca Security", board_token="orcasecurity"),
}


def _strip_html(raw_html: str) -> str:
    """Remove HTML tags, unescape HTML entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(location: str, description: str) -> WorkMode:
    """Detect work mode from location and description text."""
    combined = f"{location} {description}".lower()
    if "remote" in combined:
        return WorkMode.REMOTE
    if "hybrid" in combined:
        return WorkMode.HYBRID
    return WorkMode.ONSITE


def parse_greenhouse_job(raw: dict[str, Any], company_name: str) -> Job:
    """Parse raw Greenhouse API job dictionary into standardized Job model."""
    job_id = f"greenhouse_{raw.get('id', 'unknown')}"
    title = str(raw.get("title") or "Untitled")
    url = str(raw.get("absolute_url") or "")

    # Location
    location_data = raw.get("location")
    location_str = ""
    if isinstance(location_data, dict):
        location_str = str(location_data.get("name") or "")
    elif isinstance(location_data, str):
        location_str = location_data

    # Description — strip HTML for tech stack extraction
    content_html = str(raw.get("content") or "")
    description = _strip_html(content_html)

    # Department
    departments = raw.get("departments") or []
    dept_names = [d.get("name", "") for d in departments if isinstance(d, dict) and d.get("name")]
    department = ", ".join(dept_names) if dept_names else None

    # Tech stack extraction
    tech_stack = _extract_text_tech_keywords(f"{title} {description}")

    # Work mode
    work_mode = _detect_work_mode(location_str, description)

    # Posted date from updated_at timestamp
    posted_date = str(raw["updated_at"]) if raw.get("updated_at") else None

    return Job(
        job_id=job_id,
        title=title,
        company=company_name,
        location=location_str,
        url=url,
        apply_url=url,
        description=description[:2000] if description else "",
        tech_stack=tech_stack,
        source="greenhouse",
        work_mode=work_mode,
        department=department,
        posted_date=posted_date,
    )


class GreenhouseSource(BasePublicSource):
    """Job source aggregating roles from Israeli tech companies using Greenhouse ATS.

    Uses the free, unauthenticated Greenhouse Job Board API to fetch open positions
    from a curated directory of Israeli AI and tech startups.
    """

    source_id = "greenhouse"
    display_name = "Greenhouse"
    description = "Greenhouse ATS aggregator for Israeli AI startups (AI21 Labs, Lightricks, Tabnine, etc.)"
    supports_auto_apply = False

    def __init__(self, companies: Optional[dict[str, GreenhouseCompany]] = None) -> None:
        self._companies = companies if companies is not None else GREENHOUSE_COMPANIES
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_company_jobs(
        self,
        client: httpx.AsyncClient,
        company: GreenhouseCompany,
    ) -> list[Job]:
        """Fetch all jobs for a single Greenhouse company board."""
        async with self._semaphore:
            url = f"{BOARDS_API_BASE}/{company.board_token}/jobs?content=true"
            try:
                response = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                response.raise_for_status()
                data = response.json()
                raw_jobs = data.get("jobs") or []
                return [parse_greenhouse_job(j, company.name) for j in raw_jobs]
            except httpx.HTTPStatusError as e:
                logger.warning("Greenhouse %s HTTP error: %s", company.name, e.response.status_code)
                return []
            except Exception as e:
                logger.warning("Greenhouse %s fetch error: %s", company.name, e)
                return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all enabled Greenhouse company boards."""
        enabled = {k: c for k, c in self._companies.items() if c.enabled}
        if not enabled:
            logger.warning("No enabled Greenhouse companies configured.")
            return []

        logger.info("Fetching jobs from %d Greenhouse boards...", len(enabled))
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
                logger.error("Greenhouse board task failed unexpectedly: %s", result, exc_info=result)

        # Filter using candidate preferences if provided
        if preferences:
            all_jobs = filter_jobs(all_jobs, preferences)

        self._last_fetch_time = time.time()
        logger.info("Greenhouse: fetched %d total jobs (limit=%d)", len(all_jobs), limit)
        return all_jobs[:limit]

    async def check_health(self) -> bool:
        """Check health by pinging the first enabled company board."""
        first_company = next(
            (c for c in self._companies.values() if c.enabled), None
        )
        if not first_company:
            return False
        try:
            async with httpx.AsyncClient() as client:
                url = f"{BOARDS_API_BASE}/{first_company.board_token}/jobs"
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=8.0)
                return resp.status_code == 200
        except Exception:
            return False


__all__ = [
    "GREENHOUSE_COMPANIES",
    "GreenhouseCompany",
    "GreenhouseSource",
    "parse_greenhouse_job",
]

