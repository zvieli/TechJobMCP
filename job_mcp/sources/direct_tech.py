"""Direct tech enterprise career sources (Google, Amazon, Apple, IBM) with concurrency and caching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import Any, Optional

import httpx

from job_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BaseJobSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class DirectTechCompany:
    """Descriptor for a direct tech company career endpoint."""

    provider_id: str
    name: str
    search_url: str
    default_query: str = "student"
    default_location: str = ""
    locations: list[str] = field(default_factory=list)
    enabled: bool = True


# Curated directory of direct tech company career endpoints
DIRECT_TECH_COMPANIES: dict[str, DirectTechCompany] = {
    "google": DirectTechCompany(
        provider_id="google",
        name="Google",
        search_url="https://www.google.com/about/careers/applications/jobs/results/",
        default_query="student",
        default_location="Haifa, Israel",
    ),
    "amazon": DirectTechCompany(
        provider_id="amazon",
        name="Amazon",
        search_url="https://www.amazon.jobs/api/jobs/search",
        default_query="student",
        locations=["Haifa"],
    ),
    "apple": DirectTechCompany(
        provider_id="apple",
        name="Apple",
        search_url="https://jobs.apple.com/api/v1/search",
        default_query="student",
        locations=["postLocation-state1312"],
    ),
    "ibm": DirectTechCompany(
        provider_id="ibm",
        name="IBM",
        search_url="https://www-api.ibm.com/search/api/v2",
        default_query="student",
        default_location="Israel",
    ),
}

DEFAULT_DIRECT_TECH_COMPANIES: list[DirectTechCompany] = list(DIRECT_TECH_COMPANIES.values())


def _determine_work_mode(title: str, location: str, description: str) -> Optional[WorkMode]:
    """Helper to detect WorkMode from title, location, and description text."""
    combined = f"{title} {location} {description}".lower()
    if "remote" in combined:
        return WorkMode.REMOTE
    if "hybrid" in combined:
        return WorkMode.HYBRID
    if "on-site" in combined or "onsite" in combined:
        return WorkMode.ONSITE
    return WorkMode.ONSITE if location else None


# ---------------------------------------------------------------------------
# Google Careers Parser
# ---------------------------------------------------------------------------


def parse_google_job(obj: list[Any], company: DirectTechCompany | str = "Google") -> Job:
    """Parse raw Google Careers data array item into a Job model.

    Args:
        obj: Array item extracted from Google Careers ds:1 data blob.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        Job: Normalized Job object.
    """
    comp_name = company.name if isinstance(company, DirectTechCompany) else str(company)
    raw_id = str(obj[0]).strip() if len(obj) > 0 and obj[0] is not None else ""
    clean_id = raw_id.replace("jobs/", "").strip("/")
    job_id = f"direct_google_{clean_id}"

    title = str(obj[1]).strip() if len(obj) > 1 and obj[1] else "Untitled"
    apply_url = str(obj[2]).strip() if len(obj) > 2 and obj[2] else None
    url = f"https://www.google.com/about/careers/applications/jobs/results/{clean_id}" if clean_id else None
    if not apply_url:
        apply_url = url

    # Locations
    location_parts: list[str] = []
    if len(obj) > 9 and isinstance(obj[9], list):
        for loc in obj[9]:
            if isinstance(loc, list) and loc and isinstance(loc[0], str):
                location_parts.append(loc[0].strip())
            elif isinstance(loc, str) and loc.strip():
                location_parts.append(loc.strip())
    location_str = "; ".join(location_parts)

    # Description
    desc_chunks: list[str] = []
    if len(obj) > 3 and isinstance(obj[3], list) and len(obj[3]) > 1 and obj[3][1]:
        desc_chunks.append(str(obj[3][1]))
    if len(obj) > 4 and isinstance(obj[4], list) and len(obj[4]) > 1 and obj[4][1]:
        desc_chunks.append(str(obj[4][1]))
    if len(obj) > 10 and isinstance(obj[10], list) and len(obj[10]) > 1 and obj[10][1]:
        desc_chunks.append(str(obj[10][1]))

    raw_desc = " ".join(desc_chunks)
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Work mode & tech stack
    work_mode = _determine_work_mode(title, location_str, clean_desc)
    search_text = f"{title} {clean_desc}"
    tech_stack = _extract_text_tech_keywords(search_text)

    return Job(
        job_id=job_id,
        title=title,
        company=comp_name,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=clean_desc,
        url=url,
        apply_url=apply_url,
        source="direct_tech",
        sources=["direct_tech"],
    )


def parse_google_positions(html_text: str, company: DirectTechCompany | str = "Google") -> list[Job]:
    """Parse Google Careers search result HTML and extract job postings from data blobs.

    Args:
        html_text: HTML response text from Google Careers endpoint.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        list[Job]: Parsed list of Job objects.
    """
    pattern = r"'ds:1', hash: '\d+', data:\[(.*?)\]\],null"
    match = re.search(pattern, html_text, re.DOTALL)
    if not match:
        # Fallback regex pattern
        match = re.search(r"data:\[(.*?)\]\],null", html_text, re.DOTALL)
    if not match:
        logger.debug("Could not find Google careers data blob in HTML response")
        return []

    matched_content = match.group(1).strip()
    raw_jobs = None
    for suffix, prefix in [("]]", ""), ("]", ""), ("", ""), ("]", "["), ("]]", "[["), ("]]]", "[")]:
        try:
            candidate = prefix + matched_content + suffix
            raw_jobs = json.loads(candidate)
            break
        except Exception:
            continue

    if raw_jobs is None:
        logger.warning("Failed parsing Google careers data blob JSON")
        return []

    parsed: list[Job] = []
    if isinstance(raw_jobs, list) and raw_jobs:
        if isinstance(raw_jobs[0], list) and len(raw_jobs[0]) > 1:
            items = raw_jobs
        elif isinstance(raw_jobs[0], (str, int)):
            items = [raw_jobs]
        else:
            items = raw_jobs

        for item in items:
            if isinstance(item, list) and item:
                try:
                    parsed.append(parse_google_job(item, company=company))
                except Exception as exc:
                    logger.warning("Error parsing Google job item: %s", exc)
    return parsed


# ---------------------------------------------------------------------------
# Amazon Jobs Parser
# ---------------------------------------------------------------------------


def parse_amazon_position(raw: dict[str, Any], company: DirectTechCompany | str = "Amazon") -> Job:
    """Parse raw Amazon Jobs API search hit dictionary into a Job model.

    Args:
        raw: Job posting dictionary from Amazon Jobs API.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        Job: Normalized Job object.
    """
    comp_name = company.name if isinstance(company, DirectTechCompany) else str(company)
    fields = raw.get("fields", {}) if isinstance(raw.get("fields"), dict) else raw

    raw_id: Optional[str] = None
    for k in ("icimsJobId", "job_id", "id_icims"):
        val = fields.get(k)
        if isinstance(val, list) and val:
            raw_id = str(val[0]).strip()
            break
        elif isinstance(val, str) and val.strip():
            raw_id = val.strip()
            break

    if not raw_id:
        raw_id = str(raw.get("id") or raw.get("job_id") or "").strip()

    job_id = f"direct_amazon_{raw_id}" if raw_id else f"direct_amazon_{hashlib.md5(str(raw).encode()).hexdigest()[:8]}"

    # Title
    title_raw = fields.get("title")
    if isinstance(title_raw, list) and title_raw:
        title = str(title_raw[0]).strip()
    elif isinstance(title_raw, str):
        title = title_raw.strip()
    else:
        title = str(raw.get("title") or "Untitled").strip()

    # Location
    loc_raw = fields.get("location") or fields.get("city")
    if isinstance(loc_raw, list) and loc_raw:
        location_str = str(loc_raw[0]).strip()
    elif isinstance(loc_raw, str):
        location_str = loc_raw.strip()
    else:
        location_str = str(raw.get("location") or raw.get("city") or "").strip()

    # Department / Category
    dept_raw = fields.get("jobCategory") or fields.get("category")
    department = None
    if isinstance(dept_raw, list) and dept_raw:
        department = str(dept_raw[0]).strip()
    elif isinstance(dept_raw, str) and dept_raw.strip():
        department = dept_raw.strip()

    # Posted date
    date_raw = fields.get("postedDate") or fields.get("posted_date")
    posted_date = None
    if isinstance(date_raw, list) and date_raw:
        posted_date = str(date_raw[0]).strip()
    elif isinstance(date_raw, str) and date_raw.strip():
        posted_date = date_raw.strip()

    # Description
    desc_chunks: list[str] = []
    for k in ("description", "basicQualifications", "preferredQualifications"):
        v = fields.get(k)
        if isinstance(v, list):
            desc_chunks.extend([str(item) for item in v if item])
        elif isinstance(v, str) and v.strip():
            desc_chunks.append(v.strip())

    raw_desc = " ".join(desc_chunks) or str(raw.get("description") or "")
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # URLs
    url = f"https://amazon.jobs/jobs/{raw_id}" if raw_id else "https://amazon.jobs"
    apply_url = url

    # Work mode & tech stack
    work_mode = _determine_work_mode(title, location_str, clean_desc)
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
        department=department,
        source="direct_tech",
        sources=["direct_tech"],
    )


def parse_amazon_positions(data: dict[str, Any] | list[Any], company: DirectTechCompany | str = "Amazon") -> list[Job]:
    """Parse Amazon Jobs API search response into a list of Job models.

    Args:
        data: Amazon Jobs API response dictionary or list of hits.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        list[Job]: Parsed list of Job objects.
    """
    if isinstance(data, dict):
        hits = data.get("searchHits") or data.get("jobs") or []
    elif isinstance(data, list):
        hits = data
    else:
        hits = []

    parsed: list[Job] = []
    for item in hits:
        if isinstance(item, dict):
            try:
                parsed.append(parse_amazon_position(item, company=company))
            except Exception as exc:
                logger.warning("Error parsing Amazon job hit: %s", exc)
    return parsed


# ---------------------------------------------------------------------------
# Apple Careers Parser
# ---------------------------------------------------------------------------


def parse_apple_position(raw: dict[str, Any], company: DirectTechCompany | str = "Apple") -> Job:
    """Parse raw Apple Careers API job posting dictionary into a Job model.

    Args:
        raw: Job posting dictionary from Apple Careers API.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        Job: Normalized Job object.
    """
    comp_name = company.name if isinstance(company, DirectTechCompany) else str(company)
    raw_id = str(raw.get("id") or raw.get("positionId") or "").strip()
    job_id = f"direct_apple_{raw_id}" if raw_id else f"direct_apple_{hashlib.md5(str(raw).encode()).hexdigest()[:8]}"

    title = str(raw.get("postingTitle") or raw.get("title") or "Untitled").strip()
    transformed_title = str(raw.get("transformedPostingTitle") or "").strip()

    if raw_id and transformed_title:
        url = f"https://jobs.apple.com/en-il/details/{raw_id}/{transformed_title}"
    elif raw_id:
        url = f"https://jobs.apple.com/en-il/details/{raw_id}"
    else:
        url = "https://jobs.apple.com"
    apply_url = url

    # Locations
    locs_raw = raw.get("locations")
    location_parts: list[str] = []
    if isinstance(locs_raw, list):
        for loc in locs_raw:
            if isinstance(loc, dict) and "name" in loc:
                location_parts.append(str(loc["name"]).strip())
            elif isinstance(loc, str) and loc.strip():
                location_parts.append(loc.strip())
    elif isinstance(locs_raw, str):
        location_parts.append(locs_raw.strip())
    location_str = "; ".join(location_parts) or str(raw.get("location") or "").strip()

    # Department / Team
    team_data = raw.get("team")
    department = None
    if isinstance(team_data, dict) and "teamName" in team_data:
        department = str(team_data["teamName"]).strip()
    elif isinstance(team_data, str) and team_data.strip():
        department = team_data.strip()

    # Posted date
    posted_date = str(
        raw.get("postDateInFormat") or raw.get("postingDate") or raw.get("postDate") or ""
    ).strip() or None

    # Description
    raw_desc = str(raw.get("jobSummary") or raw.get("description") or "")
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Work mode & tech stack
    work_mode = _determine_work_mode(title, location_str, clean_desc)
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
        department=department,
        source="direct_tech",
        sources=["direct_tech"],
    )


def parse_apple_positions(data: dict[str, Any] | list[Any], company: DirectTechCompany | str = "Apple") -> list[Job]:
    """Parse Apple Careers API search response into a list of Job models.

    Args:
        data: Apple Careers API response dictionary or list of search results.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        list[Job]: Parsed list of Job objects.
    """
    if isinstance(data, dict):
        res = data.get("res", {}) if isinstance(data.get("res"), dict) else {}
        results = res.get("searchResults") or data.get("searchResults") or []
    elif isinstance(data, list):
        results = data
    else:
        results = []

    parsed: list[Job] = []
    for item in results:
        if isinstance(item, dict):
            try:
                parsed.append(parse_apple_position(item, company=company))
            except Exception as exc:
                logger.warning("Error parsing Apple job result: %s", exc)
    return parsed


# ---------------------------------------------------------------------------
# IBM Careers Parser
# ---------------------------------------------------------------------------


def parse_ibm_position(raw: dict[str, Any], company: DirectTechCompany | str = "IBM") -> Job:
    """Parse raw IBM Careers Search API hit dictionary into a Job model.

    Args:
        raw: Search hit dictionary from IBM Search API.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        Job: Normalized Job object.
    """
    comp_name = company.name if isinstance(company, DirectTechCompany) else str(company)
    source_data = raw.get("_source", {}) if isinstance(raw.get("_source"), dict) else raw

    url = str(source_data.get("url") or raw.get("url") or "").strip()
    raw_id = ""
    if "jobId=" in url:
        raw_id = url.split("jobId=")[1].split("&")[0].strip()
    if not raw_id:
        raw_id = str(raw.get("_id") or raw.get("id") or "").strip()

    job_id = f"direct_ibm_{raw_id}" if raw_id else f"direct_ibm_{hashlib.md5(str(raw).encode()).hexdigest()[:8]}"

    title = str(source_data.get("title") or raw.get("title") or "Untitled").strip()

    # Location
    loc1 = source_data.get("field_keyword_05")
    loc2 = source_data.get("field_keyword_08")
    loc_parts = [str(p).strip() for p in (loc2, loc1) if p and str(p).strip()]
    location_str = ", ".join(loc_parts) if loc_parts else str(raw.get("location") or "").strip()

    # Description
    raw_desc = str(source_data.get("description") or raw.get("description") or "")
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Work mode & tech stack
    work_mode = _determine_work_mode(title, location_str, clean_desc)
    search_text = f"{title} {clean_desc}"
    tech_stack = _extract_text_tech_keywords(search_text)

    return Job(
        job_id=job_id,
        title=title,
        company=comp_name,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=clean_desc,
        url=url or None,
        apply_url=url or None,
        source="direct_tech",
        sources=["direct_tech"],
    )


def parse_ibm_positions(data: dict[str, Any] | list[Any], company: DirectTechCompany | str = "IBM") -> list[Job]:
    """Parse IBM Search API response into a list of Job models.

    Args:
        data: IBM Search API response dictionary or list of hits.
        company: DirectTechCompany descriptor or company name string.

    Returns:
        list[Job]: Parsed list of Job objects.
    """
    if isinstance(data, dict):
        hits_obj = data.get("hits", {}) if isinstance(data.get("hits"), dict) else {}
        hits = hits_obj.get("hits") or data.get("hits") or []
    elif isinstance(data, list):
        hits = data
    else:
        hits = []

    parsed: list[Job] = []
    for item in hits:
        if isinstance(item, dict):
            try:
                parsed.append(parse_ibm_position(item, company=company))
            except Exception as exc:
                logger.warning("Error parsing IBM job hit: %s", exc)
    return parsed


# ---------------------------------------------------------------------------
# DirectTechSource Implementation
# ---------------------------------------------------------------------------


class DirectTechSource(BaseJobSource):
    """Direct career search source querying Google, Amazon, Apple, and IBM endpoints."""

    source_id: str = "direct_tech"
    display_name: str = "Direct Tech Companies"
    description: str = "Direct career endpoints for major tech enterprises (Google, Amazon, Apple, IBM)"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        companies: Optional[
            list[dict[str, Any] | DirectTechCompany]
            | dict[str, dict[str, Any] | DirectTechCompany]
        ] = None,
        max_concurrency: int = 5,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        """Initialize DirectTechSource.

        Args:
            companies: Optional list or dict of DirectTechCompany presets. Defaults to DIRECT_TECH_COMPANIES.
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
        self._companies: dict[str, DirectTechCompany] = {}
        self._cache: dict[str, tuple[float, list[Job]]] = {}

        if companies is None:
            init_companies = list(DIRECT_TECH_COMPANIES.values())
        elif isinstance(companies, dict):
            init_companies = list(companies.values())
        else:
            init_companies = list(companies)

        for comp in init_companies:
            if isinstance(comp, DirectTechCompany):
                self.add_company(comp)
            elif isinstance(comp, dict):
                self.add_company(
                    DirectTechCompany(
                        provider_id=comp["provider_id"],
                        name=comp["name"],
                        search_url=comp["search_url"],
                        default_query=comp.get("default_query", "student"),
                        default_location=comp.get("default_location", ""),
                        locations=comp.get("locations", []),
                        enabled=comp.get("enabled", True),
                    )
                )

    def add_company(self, company: DirectTechCompany) -> None:
        """Register a direct tech company provider."""
        self._companies[company.provider_id] = company

    def remove_company(self, provider_id: str) -> Optional[DirectTechCompany]:
        """Remove a direct tech company provider by provider ID."""
        self._cache.pop(provider_id, None)
        return self._companies.pop(provider_id, None)

    def get_companies(self) -> list[DirectTechCompany]:
        """Return list of currently registered company providers."""
        return list(self._companies.values())

    def clear_cache(self, provider_id: Optional[str] = None) -> None:
        """Clear cached positions for a specific provider or all providers."""
        if provider_id:
            self._cache.pop(provider_id, None)
        else:
            self._cache.clear()

    async def _fetch_google(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch listings from Google Careers search results HTML."""
        query_str = (
            " ".join(preferences.keywords)
            if (preferences and preferences.keywords)
            else company.default_query
        )
        location_str = (
            preferences.location
            if (preferences and preferences.location)
            else company.default_location
        )
        params: dict[str, str] = {
            "hl": "en-US",
        }
        if query_str:
            params["q"] = query_str
        if location_str:
            params["location"] = location_str

        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": USER_AGENT,
        }

        response = await client.get(company.search_url, params=params, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            logger.warning("Google Careers API error: HTTP %d", response.status_code)
            return []

        return parse_google_positions(response.text, company=company)

    async def _fetch_amazon(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch listings from Amazon Jobs search API."""
        query_str = (
            " ".join(preferences.keywords)
            if (preferences and preferences.keywords)
            else company.default_query
        )
        locations = (
            [preferences.location]
            if (preferences and preferences.location)
            else company.locations
        )

        payload: dict[str, Any] = {
            "query": query_str or "",
            "treatment": "OM",
        }
        if locations:
            payload["locationFacets"] = [
                [
                    {
                        "name": "normalizedCityName",
                        "requestedFacetCount": 9999,
                        "values": [{"name": loc} for loc in locations if loc],
                    }
                ]
            ]

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        response = await client.post(company.search_url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            logger.warning("Amazon Jobs API error: HTTP %d", response.status_code)
            return []

        return parse_amazon_positions(response.json(), company=company)

    async def _fetch_apple(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch listings from Apple Careers search API."""
        query_str = (
            " ".join(preferences.keywords)
            if (preferences and preferences.keywords)
            else company.default_query
        )

        payload: dict[str, Any] = {
            "query": query_str or "",
            "filters": {
                "locations": list(company.locations),
            },
            "page": 1,
            "locale": "en-il",
            "sort": "relevance",
            "format": {
                "longDate": "MMMM D, YYYY",
                "mediumDate": "MMM D, YYYY",
            },
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        response = await client.post(company.search_url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            logger.warning("Apple Careers API error: HTTP %d", response.status_code)
            return []

        return parse_apple_positions(response.json(), company=company)

    async def _fetch_ibm(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch listings from IBM Careers Search API."""
        query_str = (
            " ".join(preferences.keywords)
            if (preferences and preferences.keywords)
            else company.default_query
        )
        query_val = query_str or "*"
        loc_val = (
            preferences.location
            if (preferences and preferences.location)
            else company.default_location
        )

        payload: dict[str, Any] = {
            "appId": "careers",
            "scopes": ["careers2"],
            "query": {
                "bool": {
                    "must": [
                        {
                            "simple_query_string": {
                                "query": query_val,
                                "fields": [
                                    "keywords^1",
                                    "body^1",
                                    "url^2",
                                    "description^2",
                                    "h1s_content^2",
                                    "title^3",
                                    "field_text_01",
                                ],
                            }
                        }
                    ]
                }
            },
            "size": min(limit, 50),
            "sm": {
                "query": query_val,
                "lang": "zz",
            },
            "_source": [
                "_id",
                "title",
                "url",
                "description",
                "field_keyword_05",
                "field_keyword_08",
            ],
        }
        if loc_val:
            payload["post_filter"] = {
                "term": {
                    "field_keyword_05": loc_val,
                }
            }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        response = await client.post(company.search_url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            logger.warning("IBM Careers API error: HTTP %d", response.status_code)
            return []

        return parse_ibm_positions(response.json(), company=company)

    async def _fetch_company_positions(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch listings for a single direct company with caching and semaphore throttling."""
        now = time.time()
        cache_key = company.provider_id
        has_keywords = bool(preferences and preferences.keywords)
        if not has_keywords and cache_key in self._cache:
            ts, cached_jobs = self._cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        async with self.semaphore:
            t0 = time.perf_counter()
            pid = company.provider_id.lower()
            if pid == "google":
                jobs = await self._fetch_google(company, client, preferences, limit)
            elif pid == "amazon" or "amazon" in pid:
                jobs = await self._fetch_amazon(company, client, preferences, limit)
            elif pid == "apple" or "apple" in pid:
                jobs = await self._fetch_apple(company, client, preferences, limit)
            elif pid == "ibm" or "ibm" in pid:
                jobs = await self._fetch_ibm(company, client, preferences, limit)
            else:
                # Fallback to amazon-style or default json search if custom provider
                jobs = await self._fetch_amazon(company, client, preferences, limit)

            duration_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "Direct tech company fetch completed",
                company=company.name,
                provider_id=company.provider_id,
                duration_ms=round(duration_ms, 2),
                positions_count=len(jobs),
                source="direct_tech",
            )

            if not has_keywords:
                self._cache[cache_key] = (now, jobs)

            return jobs

    async def _fetch_company_positions_safe(
        self,
        company: DirectTechCompany,
        client: httpx.AsyncClient,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Wrap provider fetch with exception safety and stale cache fallback."""
        try:
            return await self._fetch_company_positions(
                company=company,
                client=client,
                preferences=preferences,
                limit=limit,
            )
        except Exception as exc:
            logger.warning(
                "Exception fetching positions for %s (%s): %s",
                company.name,
                company.provider_id,
                exc,
            )
            if company.provider_id in self._cache:
                return self._cache[company.provider_id][1]
            return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings matching preferences across all registered Direct Tech companies."""
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
                    logger.warning("Error in Direct Tech fetch gather: %s", res)
        finally:
            if should_close_client:
                await client.aclose()

        # Tag source
        tagged_jobs: list[Job] = []
        for j in all_jobs:
            j.source = "direct_tech"
            if "direct_tech" not in j.sources:
                j.sources.insert(0, "direct_tech")
            tagged_jobs.append(j)

        # Preferences filtering
        if preferences:
            tagged_jobs = filter_jobs(tagged_jobs, preferences)

        # Truncate
        if limit and len(tagged_jobs) > limit:
            tagged_jobs = tagged_jobs[:limit]

        return tagged_jobs

    async def check_health(self) -> bool:
        """Check operational readiness of direct tech company endpoints."""
        test_company = (
            self._companies.get("google")
            or (next(iter(self._companies.values())) if self._companies else None)
            or DIRECT_TECH_COMPANIES.get("google")
        )
        if not test_company:
            return False

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            headers = {"User-Agent": USER_AGENT}
            if test_company.provider_id == "google":
                response = await client.get(test_company.search_url, headers=headers, timeout=self.timeout)
            else:
                response = await client.post(test_company.search_url, json={}, headers=headers, timeout=self.timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Direct tech health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()
