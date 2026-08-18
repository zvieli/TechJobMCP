"""AllJobs Israel job source implementation with browser-grade headers and category caching."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, Optional

import httpx

from hireme_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode
from hireme_mcp.sources.base import BaseJobSource
from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)

ALLJOBS_BASE_URL = "https://www.alljobs.co.il"
SEARCH_MOBILE_ENDPOINT = "/SearchResultsMobile.ashx"

ALLJOBS_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Known Israeli tech categories on AllJobs
DEFAULT_TECH_CATEGORIES: dict[str, int] = {
    "software": 235,
    "ai": 1998,
    "computers_networks": 357,
    "qa": 237,
    "internet": 1563,
}


def parse_alljobs_position(raw: dict[str, Any]) -> Job:
    """Parse a raw AllJobs job listing payload into a standardized Job model.

    Args:
        raw: Raw dictionary from AllJobs search feeds.

    Returns:
        Job: Normalized Job object tagged with 'alljobs'.
    """
    raw_id = (
        raw.get("JobID")
        or raw.get("job_id")
        or raw.get("id")
        or raw.get("JobCode")
        or raw.get("JobId")
    )
    title = str(
        raw.get("JobTitle")
        or raw.get("title")
        or raw.get("Title")
        or raw.get("JobName")
        or "Untitled"
    ).strip()
    company = str(
        raw.get("CompanyName")
        or raw.get("company")
        or raw.get("Company")
        or raw.get("EmployerName")
        or "Confidential"
    ).strip()

    # Location parsing
    city = str(raw.get("JobCity") or raw.get("city") or raw.get("City") or raw.get("CityName") or "").strip()
    region = str(raw.get("JobRegion") or raw.get("region") or raw.get("Region") or raw.get("RegionName") or "").strip()
    if city and region and city != region:
        location_str = f"{city}, {region}"
    elif city or region:
        location_str = city or region
    else:
        location_str = str(raw.get("location") or raw.get("Location") or "").strip()

    # ID resolution
    if raw_id is not None and str(raw_id).strip():
        str_raw_id = str(raw_id).strip()
        job_id = str_raw_id if str_raw_id.startswith("alljobs_") else f"alljobs_{str_raw_id}"
        numeric_id = str_raw_id.replace("alljobs_", "")
    else:
        hash_seed = f"{title}_{company}_{location_str}".encode("utf-8")
        hash_val = hashlib.md5(hash_seed).hexdigest()[:10]
        job_id = f"alljobs_{hash_val}"
        numeric_id = hash_val

    # Description parsing & cleaning
    desc_raw = (
        raw.get("JobDescription")
        or raw.get("description")
        or raw.get("Description")
        or raw.get("JobRequirements")
        or raw.get("requirements")
        or ""
    )
    clean_desc = re.sub(r"<[^>]+>", " ", str(desc_raw))
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Work mode determination
    is_remote = bool(raw.get("IsRemote") or raw.get("is_remote") or False)
    is_hybrid = bool(raw.get("IsHybrid") or raw.get("is_hybrid") or False)
    workplace = str(raw.get("WorkMode") or raw.get("work_mode") or raw.get("workplace_type") or "").strip().lower()
    combined_search_text = f"{title} {location_str} {clean_desc}".lower()

    if (
        is_remote
        or workplace == "remote"
        or "remote" in combined_search_text
        or "מהבית" in combined_search_text
        or "עבודה מהבית" in combined_search_text
    ):
        work_mode = WorkMode.REMOTE
    elif (
        is_hybrid
        or workplace == "hybrid"
        or "hybrid" in combined_search_text
        or "היברידי" in combined_search_text
    ):
        work_mode = WorkMode.HYBRID
    elif workplace in ("on-site", "onsite") or "onsite" in combined_search_text:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = WorkMode.ONSITE if location_str else None

    # Salary parsing
    sal = raw.get("Salary") or raw.get("salary") or raw.get("SalaryRange") or raw.get("SalaryText") or raw.get("salary_range")
    salary_range: Optional[str] = None
    if isinstance(sal, dict):
        min_val = sal.get("min") or sal.get("from") or ""
        max_val = sal.get("max") or sal.get("to") or ""
        formatted = f"{min_val} - {max_val}".strip(" -")
        salary_range = formatted if formatted else None
    elif sal is not None:
        salary_range = str(sal).strip() or None

    # URLs
    raw_url = raw.get("url") or raw.get("JobURL") or raw.get("JobUrl")
    if raw_url and str(raw_url).startswith("http"):
        url = str(raw_url).strip()
    else:
        url = f"https://www.alljobs.co.il/User/ShowJob.aspx?JobID={numeric_id}"

    apply_url_raw = raw.get("ApplyURL") or raw.get("apply_url")
    apply_url = str(apply_url_raw).strip() if apply_url_raw else url

    posted_date_raw = raw.get("Date") or raw.get("JobDate") or raw.get("posted_date") or raw.get("CreateDate")
    posted_date = str(posted_date_raw).strip() if posted_date_raw else None

    department_raw = raw.get("Department") or raw.get("department") or raw.get("CategoryName")
    department = str(department_raw).strip() if department_raw else None

    # Tech stack extraction
    search_keywords_text = f"{title} {clean_desc} {department or ''}"
    tech_stack = _extract_text_tech_keywords(search_keywords_text)

    return Job(
        job_id=job_id,
        title=title,
        company=company,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=clean_desc,
        salary_range=salary_range,
        posted_date=posted_date,
        url=url,
        apply_url=apply_url,
        department=department,
        source="alljobs",
        sources=["alljobs"],
    )


class AllJobsSource(BaseJobSource):
    """Job source querying AllJobs Israel index with anti-blocking headers and category feeds."""

    source_id: str = "alljobs"
    display_name: str = "AllJobs Israel"
    description: str = "Israel's largest tech and general job index"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        base_url: str = ALLJOBS_BASE_URL,
        headers: Optional[dict[str, str]] = None,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
        health_timeout: float = 5.0,
    ) -> None:
        """Initialize AllJobsSource.

        Args:
            base_url: Base domain for AllJobs.
            headers: HTTP headers to emulate browser. Defaults to ALLJOBS_HEADERS.
            cache_ttl_seconds: TTL for category and feed caching.
            client: Optional shared httpx.AsyncClient instance.
            timeout: Default HTTP query timeout in seconds.
            health_timeout: Health check HTTP timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or dict(ALLJOBS_HEADERS)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = client
        self.timeout = timeout
        self.health_timeout = health_timeout

        self._categories_cache: dict[str, int] = {}
        self._categories_cache_ts: float = 0.0
        self._feed_cache: dict[tuple[tuple[str, str], ...], tuple[float, list[Job]]] = {}

    def clear_categories_cache(self) -> None:
        """Clear cached category mappings."""
        self._categories_cache.clear()
        self._categories_cache_ts = 0.0

    def clear_cache(self, clear_categories: bool = True) -> None:
        """Clear cached search feeds and optionally categories.

        Args:
            clear_categories: Whether to also clear categories cache.
        """
        self._feed_cache.clear()
        if clear_categories:
            self.clear_categories_cache()

    async def fetch_categories(
        self,
        force_refresh: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, int]:
        """Fetch and cache categories from AllJobs search engine data.

        Args:
            force_refresh: Whether to ignore cached category dictionary.
            client: Optional httpx.AsyncClient to reuse.

        Returns:
            dict[str, int]: Mapping from category name to category ID.
        """
        now = time.time()
        if not force_refresh and self._categories_cache and (now - self._categories_cache_ts < self.cache_ttl_seconds):
            return dict(self._categories_cache)

        should_close_client = False
        active_client = client or self._client
        if active_client is None:
            active_client = httpx.AsyncClient()
            should_close_client = True

        url = f"{self.base_url}{SEARCH_MOBILE_ENDPOINT}"
        params = {"action": "getSearchEngineData", "categories": "true"}

        try:
            response = await active_client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.warning("AllJobs getSearchEngineData returned status %d", response.status_code)
                return dict(self._categories_cache) if self._categories_cache else dict(DEFAULT_TECH_CATEGORIES)

            data = response.json()
            cats_raw: list[dict[str, Any]] = []
            if isinstance(data, list):
                cats_raw = data
            elif isinstance(data, dict):
                cats_raw = data.get("Categories") or data.get("categories") or []

            parsed_cats: dict[str, int] = {}
            for item in cats_raw:
                if not isinstance(item, dict):
                    continue
                cid = item.get("CategoryID") or item.get("id") or item.get("CategoryId") or item.get("code")
                cname = item.get("CategoryName") or item.get("name") or item.get("Name")
                if cid is not None and cname:
                    try:
                        parsed_cats[str(cname)] = int(cid)
                    except (ValueError, TypeError):
                        pass

            if parsed_cats:
                self._categories_cache = parsed_cats
                self._categories_cache_ts = now
                return dict(self._categories_cache)

            return dict(self._categories_cache) if self._categories_cache else dict(DEFAULT_TECH_CATEGORIES)
        except Exception as exc:
            logger.warning("Failed to fetch AllJobs categories: %s", exc)
            return dict(self._categories_cache) if self._categories_cache else dict(DEFAULT_TECH_CATEGORIES)
        finally:
            if should_close_client:
                await active_client.aclose()

    async def check_health(self) -> bool:
        """Check the operational health of AllJobs mobile endpoint.

        Returns:
            bool: True if endpoint returns 200 and valid categories, False otherwise.
        """
        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        url = f"{self.base_url}{SEARCH_MOBILE_ENDPOINT}"
        params = {"action": "getSearchEngineData", "categories": "true"}

        try:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.health_timeout,
            )
            if response.status_code != 200:
                return False

            data = response.json()
            if isinstance(data, dict):
                cats = data.get("Categories") or data.get("categories")
                return bool(cats and len(cats) > 0)
            elif isinstance(data, list):
                return len(data) > 0
            return False
        except Exception as exc:
            logger.warning("AllJobs health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        params: dict[str, Any],
    ) -> list[Job]:
        """Query a single AllJobs search feed and parse results with caching and error isolation."""
        now = time.time()
        param_key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if param_key in self._feed_cache:
            ts, cached_jobs = self._feed_cache[param_key]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        url = f"{self.base_url}{SEARCH_MOBILE_ENDPOINT}"
        try:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.warning(
                    "AllJobs search feed returned status %d for params %s",
                    response.status_code,
                    params,
                )
                return []

            data = response.json()
            jobs_raw: list[dict[str, Any]] = []
            if isinstance(data, list):
                jobs_raw = data
            elif isinstance(data, dict):
                jobs_raw = (
                    data.get("Jobs")
                    or data.get("jobs")
                    or data.get("Results")
                    or (data.get("d", {}).get("Jobs") if isinstance(data.get("d"), dict) else [])
                    or []
                )

            parsed_jobs: list[Job] = []
            for raw_job in jobs_raw:
                if not isinstance(raw_job, dict):
                    continue
                try:
                    job = parse_alljobs_position(raw_job)
                    parsed_jobs.append(job)
                except Exception as exc:
                    logger.warning("Error parsing AllJobs position: %s", exc)

            self._feed_cache[param_key] = (now, parsed_jobs)
            return parsed_jobs
        except Exception as exc:
            logger.warning("AllJobs fetch query error for params %s: %s", params, exc)
            return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from AllJobs matching optional preferences up to limit.

        Args:
            preferences: Optional JobPreferences filter.
            limit: Maximum number of jobs to return.

        Returns:
            list[Job]: Standardized Job listings tagged with 'alljobs'.
        """
        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient()
            should_close_client = True

        try:
            # Determine query parameters
            query_params_list: list[dict[str, Any]] = []

            # 1. Category-based search params
            categories = await self.fetch_categories(client=client)
            # Default tech category IDs: Software (235), AI (1998), Computers/Networks (357)
            primary_cat_ids = [235, 1998, 357]
            if categories:
                # Add any matched category IDs from preferences
                for cat_id in primary_cat_ids:
                    query_params_list.append({"action": "getJobs", "cat": cat_id, "page": 1})
            else:
                query_params_list.append({"action": "getJobs", "page": 1})

            # 2. Add keyword-specific query params if preferences specified keywords/stack
            if preferences:
                search_terms = list(preferences.tech_stack) + list(preferences.keywords)
                for term in search_terms[:3]:  # Top 3 terms
                    query_params_list.append({"action": "getJobs", "keyword": term, "page": 1})

            # Fetch queries concurrently
            tasks = [self._fetch_feed(client, params) for params in query_params_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_jobs: list[Job] = []
            seen_ids: set[str] = set()

            for res in results:
                if isinstance(res, list):
                    for job in res:
                        if job.job_id not in seen_ids:
                            seen_ids.add(job.job_id)
                            all_jobs.append(job)
                elif isinstance(res, Exception):
                    logger.warning("Exception in AllJobs fetch gather: %s", res)

            # Apply preferences filtering if provided
            if preferences:
                all_jobs = filter_jobs(all_jobs, preferences)

            # Truncate to limit
            if limit and len(all_jobs) > limit:
                all_jobs = all_jobs[:limit]

            return all_jobs
        except Exception as exc:
            logger.warning("Unhandled error during AllJobs fetch_jobs: %s", exc)
            return []
        finally:
            if should_close_client:
                await client.aclose()
