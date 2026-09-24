"""Lever ATS direct job source — free public JSON API for Israeli startup career boards."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import html
import os
import re
import time
from typing import Any, Optional

import httpx

from job_mcp.core.api_client import filter_jobs
from job_mcp.core.section_parser import extract_clean_job_tech_stack, parse_job_sections
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BasePublicSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = os.getenv("REPO_URL", "https://github.com/TechJobMCP/TechJobMCP")
POSTINGS_API_BASE = "https://api.lever.co/v0/postings"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": f"TechJobMCP/1.0 (Job Aggregator; +{REPO_URL})",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}
MAX_CONCURRENT_REQUESTS = 4


@dataclass
class LeverCompany:
    """Descriptor for a company using Lever ATS."""

    name: str
    slug: str
    enabled: bool = True


# Curated directory of Israeli tech companies and startups using Lever
LEVER_COMPANIES: dict[str, LeverCompany] = {
    "drivenets": LeverCompany(name="DriveNets", slug="drivenets", enabled=True),
    "here": LeverCompany(name="HERE Technologies", slug="here", enabled=True),
    "yotpo": LeverCompany(name="Yotpo", slug="yotpo", enabled=True),
    "redis": LeverCompany(name="Redis", slug="redis", enabled=True),
    "melio": LeverCompany(name="Melio", slug="melio", enabled=True),
    "papaya_global": LeverCompany(name="Papaya Global", slug="papayaglobal", enabled=True),
    "hibob": LeverCompany(name="HiBob", slug="hibob", enabled=True),
    "palo_alto_networks": LeverCompany(name="Palo Alto Networks Israel", slug="paloaltonetworks", enabled=True),
    "riskified": LeverCompany(name="Riskified", slug="riskified", enabled=True),
    "k_health": LeverCompany(name="K Health", slug="khealth", enabled=True),
    "tipalti": LeverCompany(name="Tipalti", slug="tipalti", enabled=True),
    "deel": LeverCompany(name="Deel Israel", slug="deel", enabled=True),
    "run_ai": LeverCompany(name="Run:ai", slug="runai", enabled=True),
    "deci": LeverCompany(name="Deci AI", slug="deci", enabled=True),
}


def _strip_html(raw_html: str) -> str:
    """Remove HTML tags, unescape HTML entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(workplace_type: str, location: str, description: str) -> WorkMode:
    """Detect work mode from workplaceType, location, and description text."""
    wpt = workplace_type.lower().strip()
    if wpt in ("remote", "telecommute"):
        return WorkMode.REMOTE
    if wpt == "hybrid":
        return WorkMode.HYBRID
    if wpt in ("onsite", "on-site"):
        return WorkMode.ONSITE

    combined = f"{location} {description}".lower()
    if "remote" in combined:
        return WorkMode.REMOTE
    if "hybrid" in combined:
        return WorkMode.HYBRID
    return WorkMode.ONSITE


def parse_lever_job(raw: dict[str, Any], company_name: str) -> Job:
    """Parse raw Lever API job dictionary into standardized Job model."""
    job_id = f"lever_{raw.get('id', 'unknown')}"
    title = str(raw.get("text") or "Untitled")

    # Categories
    categories = raw.get("categories") or {}
    if not isinstance(categories, dict):
        categories = {}

    location_str = str(categories.get("location") or "")
    team = str(categories.get("team") or "")
    department = team if team else (str(categories.get("department") or "") if categories.get("department") else None)

    # Description and structured sections
    parts_for_parsing: list[str] = []
    description_plain = raw.get("descriptionPlain")
    description_html = raw.get("description")

    if description_html:
        parts_for_parsing.append(str(description_html))
    elif description_plain:
        parts_for_parsing.append(str(description_plain))

    lists = raw.get("lists")
    list_clean_texts: list[str] = []
    if isinstance(lists, list):
        for lst in lists:
            if isinstance(lst, dict):
                heading = str(lst.get("text") or "").strip()
                content = str(lst.get("content") or "").strip()
                if heading and content:
                    parts_for_parsing.append(f"<h3>{heading}</h3>\n{content}")
                    list_clean_texts.append(f"{heading}:\n{_strip_html(content)}")
                elif content:
                    parts_for_parsing.append(content)
                    list_clean_texts.append(_strip_html(content))

    add_html = raw.get("additional")
    add_plain = raw.get("additionalPlain")
    if add_html:
        parts_for_parsing.append(str(add_html))
    elif add_plain:
        parts_for_parsing.append(str(add_plain))

    combined_for_sections = "\n\n".join(parts_for_parsing)
    sections = parse_job_sections(combined_for_sections)

    if description_plain:
        description = str(description_plain)
    else:
        description = _strip_html(str(description_html or ""))

    if list_clean_texts:
        description = f"{description}\n\n" + "\n\n".join(list_clean_texts)
        description = description.strip()
    if add_plain or add_html:
        add_text = str(add_plain) if add_plain else _strip_html(str(add_html))
        if add_text:
            description = f"{description}\n\n{add_text}".strip()

    # URLs
    hosted_url = str(raw.get("hostedUrl") or "")
    apply_url_val = str(raw.get("applyUrl") or "")
    url = hosted_url if hosted_url else apply_url_val
    apply_url = apply_url_val if apply_url_val else hosted_url

    # Workplace / Work mode
    workplace_type = str(raw.get("workplaceType") or "")
    work_mode = _detect_work_mode(workplace_type, location_str, description)

    # Posted date from createdAt timestamp
    posted_date = str(raw["createdAt"]) if raw.get("createdAt") is not None else None

    # Tech stack extraction
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        department=department,
        fallback_text=description,
    )

    return Job(
        job_id=job_id,
        title=title,
        company=company_name,
        location=location_str,
        url=url,
        apply_url=apply_url,
        description=description[:2000] if description else "",
        tech_stack=tech_stack,
        source="lever",
        work_mode=work_mode,
        department=department,
        posted_date=posted_date,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
    )


class LeverSource(BasePublicSource):
    """Job source aggregating roles from Israeli tech companies using Lever ATS.

    Uses the free, unauthenticated Lever Postings API to fetch open positions
    from a curated directory of Israeli tech startups.
    """

    source_id = "lever"
    display_name = "Lever"
    description = "Lever ATS aggregator for Israeli tech startups (DriveNets, Redis, Melio, HiBob, etc.)"
    supports_auto_apply = True
    timeout: float = 15.0

    def __init__(self, companies: Optional[dict[str, LeverCompany]] = None) -> None:
        self._companies = companies if companies is not None else LEVER_COMPANIES
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_company_jobs(
        self,
        client: httpx.AsyncClient,
        company: LeverCompany,
    ) -> list[Job]:
        """Fetch all jobs for a single Lever company board."""
        async with self._semaphore:
            url = f"{POSTINGS_API_BASE}/{company.slug}?mode=json&limit=100"
            try:
                response = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list):
                    logger.warning("Lever %s returned non-list response: %s", company.name, type(data))
                    return []
                return [parse_lever_job(j, company.name) for j in data if isinstance(j, dict)]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug("Lever %s HTTP 404 (board inactive or migrated)", company.name)
                else:
                    logger.warning("Lever %s HTTP error: %s", company.name, e.response.status_code)
                return []
            except Exception as e:
                logger.debug("Lever %s fetch error: %s", company.name, e)
                return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all enabled Lever company boards."""
        enabled = {k: c for k, c in self._companies.items() if c.enabled}
        if not enabled:
            logger.warning("No enabled Lever companies configured.")
            return []

        logger.info("Fetching jobs from %d Lever boards...", len(enabled))
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
                logger.error("Lever board task failed unexpectedly: %s", result, exc_info=result)

        # Filter using candidate preferences if provided
        if preferences:
            all_jobs = filter_jobs(all_jobs, preferences, enable_semantic=False, enable_system1=False)

        self._last_fetch_time = time.time()
        logger.info("Lever: fetched %d total jobs (limit=%d)", len(all_jobs), limit)
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
                url = f"{POSTINGS_API_BASE}/{first_company.slug}?mode=json&limit=1"
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=8.0)
                return resp.status_code == 200
        except Exception:
            return False


__all__ = [
    "LEVER_COMPANIES",
    "LeverCompany",
    "LeverSource",
    "parse_lever_job",
]
