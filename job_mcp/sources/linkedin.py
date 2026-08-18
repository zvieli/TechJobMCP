"""Lightweight LinkedIn job source implementation using LinkedIn Guest Jobs Search & Detail APIs."""

from __future__ import annotations

import asyncio
import hashlib
import html
import re
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from job_mcp.core.api_client import _extract_text_tech_keywords, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BaseJobSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

LINKEDIN_SEARCH_API_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
LINKEDIN_JOB_DETAIL_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting"
LINKEDIN_DEFAULT_LOCATION = "Israel"
LINKEDIN_REQUEST_TIMEOUT: float = 5.0
LINKEDIN_HEALTH_TIMEOUT: float = 5.0
LINKEDIN_DEFAULT_RATE_LIMIT_DELAY: float = 0.5
LINKEDIN_DEFAULT_MAX_RETRIES: int = 3

LINKEDIN_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,he-IL;q=0.8,he;q=0.7",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

# Work mode mapping to LinkedIn f_WT query parameter:
# 1 = On-site, 2 = Remote, 3 = Hybrid
WORK_MODE_TO_FWT: dict[WorkMode, str] = {
    WorkMode.ONSITE: "1",
    WorkMode.REMOTE: "2",
    WorkMode.HYBRID: "3",
}

FWT_TO_WORK_MODE: dict[str, WorkMode] = {
    "1": WorkMode.ONSITE,
    "2": WorkMode.REMOTE,
    "3": WorkMode.HYBRID,
}


def _clean_html_text(raw_text: Optional[str]) -> str:
    """Remove HTML tags, unescape entities, and normalize whitespace."""
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw_text))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_job_id_from_text(card_html: str) -> Optional[str]:
    """Extract numeric LinkedIn job ID from various card attributes and link formats."""
    # 1. data-entity-urn="urn:li:jobPosting:4152839402"
    urn_match = re.search(r'data-entity-urn="urn:li:jobPosting:(\d+)"', card_html)
    if urn_match:
        return urn_match.group(1)

    # 2. data-job-id="4152839402" or data-id="4152839402"
    job_id_attr = re.search(r'data-(?:job-id|id)="(\d+)"', card_html)
    if job_id_attr:
        return job_id_attr.group(1)

    # 3. href with /jobs/view/...-4152839402 or /jobs/view/4152839402
    view_url_match = re.search(r'/jobs/view/(?:[^\?\"\'\s/]+-)?(\d+)', card_html)
    if view_url_match:
        return view_url_match.group(1)

    # 4. href with currentJobId=4152839402
    current_job_id_match = re.search(r'[?&]currentJobId=(\d+)', card_html)
    if current_job_id_match:
        return current_job_id_match.group(1)

    return None


def _clean_linkedin_url(raw_url: Optional[str], numeric_id: Optional[str]) -> str:
    """Construct clean canonical LinkedIn job URL without tracking parameters."""
    if numeric_id:
        return f"https://www.linkedin.com/jobs/view/{numeric_id}"
    if not raw_url:
        return ""
    # Strip tracking query params
    parsed = urlparse(raw_url)
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return clean_url or raw_url


def parse_linkedin_job_card(card_html: str) -> Optional[Job]:
    """Parse a single LinkedIn job card HTML fragment into a standardized Job model.

    Args:
        card_html: HTML string representing a single job card.

    Returns:
        Optional[Job]: Normalized Job object or None if invalid/empty.
    """
    if not card_html or not card_html.strip():
        return None

    # Title extraction
    title_match = (
        re.search(r'<h3[^>]*class="[^"]*(?:base-search-card__title|job-search-card__title)[^"]*"[^>]*>(.*?)</h3>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<span[^>]*class="[^"]*sr-only[^"]*"[^>]*>(.*?)</span>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<a[^>]*class="[^"]*base-card__full-link[^"]*"[^>]*>(.*?)</a>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<h3[^>]*>(.*?)</h3>', card_html, re.DOTALL | re.IGNORECASE)
    )
    title = _clean_html_text(title_match.group(1)) if title_match else ""

    # Company extraction
    company_match = (
        re.search(r'<h4[^>]*class="[^"]*base-search-card__subtitle[^"]*"[^>]*>(.*?)</h4>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<a[^>]*class="[^"]*hidden-nested-link[^"]*"[^>]*>(.*?)</a>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<div[^>]*class="[^"]*base-search-card__subtitle[^"]*"[^>]*>(.*?)</div>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<h4[^>]*>(.*?)</h4>', card_html, re.DOTALL | re.IGNORECASE)
    )
    company = _clean_html_text(company_match.group(1)) if company_match else ""

    # If neither title nor company exists or card has no job info, discard
    if not title and not company:
        return None
    if not title:
        title = "Untitled"
    if not company:
        company = "Confidential"

    # Job ID extraction
    numeric_id = _extract_job_id_from_text(card_html)
    if numeric_id:
        job_id = f"linkedin_{numeric_id}"
    else:
        hash_seed = f"{title}_{company}".encode("utf-8")
        hash_val = hashlib.md5(hash_seed).hexdigest()[:10]
        job_id = f"linkedin_{hash_val}"
        numeric_id = hash_val

    # Location extraction
    location_match = (
        re.search(r'<span[^>]*class="[^"]*(?:job-search-card__location|base-search-card__metadata-item)[^"]*"[^>]*>(.*?)</span>', card_html, re.DOTALL | re.IGNORECASE)
        or re.search(r'<div[^>]*class="[^"]*base-search-card__metadata[^"]*"[^>]*>\s*<span[^>]*>(.*?)</span>', card_html, re.DOTALL | re.IGNORECASE)
    )
    location_str = _clean_html_text(location_match.group(1)) if location_match else ""

    # Posted date extraction
    date_match = re.search(r'<time[^>]*datetime="([^"]+)"[^>]*>', card_html, re.IGNORECASE)
    if date_match:
        posted_date = date_match.group(1).strip()
    else:
        time_text_match = (
            re.search(r'<time[^>]*class="[^"]*(?:job-search-card__listdate|job-search-card__listdate--new)[^"]*"[^>]*>(.*?)</time>', card_html, re.DOTALL | re.IGNORECASE)
            or re.search(r'<span[^>]*class="[^"]*posted-time-ago__text[^"]*"[^>]*>(.*?)</span>', card_html, re.DOTALL | re.IGNORECASE)
        )
        posted_date = _clean_html_text(time_text_match.group(1)) if time_text_match else None

    # Snippet / Description extraction
    snippet_match = re.search(r'<p[^>]*class="[^"]*(?:job-search-card__snippet|base-search-card__snippet)[^"]*"[^>]*>(.*?)</p>', card_html, re.DOTALL | re.IGNORECASE)
    snippet = _clean_html_text(snippet_match.group(1)) if snippet_match else ""

    # URL & Apply URL
    url_match = re.search(r'href="([^"]*(?:jobs/view|linkedin.com/jobs)[^"]*)"', card_html, re.IGNORECASE)
    raw_url = url_match.group(1) if url_match else None
    canonical_url = _clean_linkedin_url(raw_url, numeric_id)
    apply_url = canonical_url

    # Work mode determination
    combined_text = f"{title} {location_str} {snippet}".lower()
    if "remote" in combined_text or "מהבית" in combined_text or "עבודה מהבית" in combined_text:
        work_mode = WorkMode.REMOTE
    elif "hybrid" in combined_text or "היברידי" in combined_text:
        work_mode = WorkMode.HYBRID
    elif "on-site" in combined_text or "onsite" in combined_text:
        work_mode = WorkMode.ONSITE
    elif location_str:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = None

    # Tech stack extraction
    search_keywords_text = f"{title} {location_str} {snippet}"
    tech_stack = _extract_text_tech_keywords(search_keywords_text)

    return Job(
        job_id=job_id,
        title=title,
        company=company,
        location=location_str,
        work_mode=work_mode,
        tech_stack=tech_stack,
        description=snippet,
        posted_date=posted_date,
        url=canonical_url,
        apply_url=apply_url,
        source="linkedin",
        sources=["linkedin"],
    )


def parse_linkedin_search_results(html_text: str) -> list[Job]:
    """Parse LinkedIn search results HTML page into a list of standardized Job models.

    Args:
        html_text: Raw HTML string of LinkedIn search results.

    Returns:
        list[Job]: List of parsed Job objects, deduplicated by job_id.
    """
    if not html_text or not html_text.strip():
        return []

    # 1. Extract all <li>...</li> card items
    li_cards = re.findall(r'<li[\s>].*?</li>', html_text, re.DOTALL | re.IGNORECASE)

    # 2. Remove matched <li> elements and extract any standalone card <div> elements
    remaining_html = re.sub(r'<li[\s>].*?</li>', ' ', html_text, flags=re.DOTALL | re.IGNORECASE)
    div_cards = re.split(
        r'(?=<div[^>]*class="[^"]*(?:base-card|base-search-card|job-search-card)[^"]*")',
        remaining_html,
        flags=re.IGNORECASE,
    )

    all_chunks = list(li_cards) + [d for d in div_cards if d and d.strip()]

    jobs: list[Job] = []
    seen_ids: set[str] = set()

    for chunk in all_chunks:
        job = parse_linkedin_job_card(chunk)
        if job and job.job_id not in seen_ids:
            seen_ids.add(job.job_id)
            jobs.append(job)

    return jobs


def parse_linkedin_job_details(html_text: str) -> dict[str, Any]:
    """Parse raw HTML from LinkedIn guest job posting endpoint into structured details dict.

    Args:
        html_text: Raw HTML from https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{id}

    Returns:
        dict[str, Any]: Dictionary containing extracted title, company, location,
            description, posted_date, apply_url, seniority_level, employment_type,
            department, work_mode, and tech_stack.
    """
    if not html_text or not html_text.strip():
        return {}

    # Title
    title_match = (
        re.search(r'<h1[^>]*class="[^"]*top-card-layout__title[^"]*"[^>]*>(.*?)</h1>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<h2[^>]*class="[^"]*top-card-layout__title[^"]*"[^>]*>(.*?)</h2>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<h1[^>]*>(.*?)</h1>', html_text, re.DOTALL | re.IGNORECASE)
    )
    title = _clean_html_text(title_match.group(1)) if title_match else ""

    # Company
    company_match = (
        re.search(r'<a[^>]*class="[^"]*topcard__org-name-link[^"]*"[^>]*>(.*?)</a>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<span[^>]*class="[^"]*topcard__flavor[^"]*"[^>]*>(.*?)</span>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<a[^>]*class="[^"]*top-card-layout__first-sub-headline[^"]*"[^>]*>(.*?)</a>', html_text, re.DOTALL | re.IGNORECASE)
    )
    company = _clean_html_text(company_match.group(1)) if company_match else ""

    # Location
    location_match = (
        re.search(r'<span[^>]*class="[^"]*topcard__flavor--bullet[^"]*"[^>]*>(.*?)</span>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<span[^>]*class="[^"]*top-card-layout__second-sub-headline[^"]*"[^>]*>(.*?)</span>', html_text, re.DOTALL | re.IGNORECASE)
    )
    location = _clean_html_text(location_match.group(1)) if location_match else ""

    # Posted Date
    date_match = re.search(r'<time[^>]*datetime="([^"]+)"[^>]*>', html_text, re.IGNORECASE)
    if date_match:
        posted_date = date_match.group(1).strip()
    else:
        time_match = (
            re.search(r'<span[^>]*class="[^"]*posted-time-ago__text[^"]*"[^>]*>(.*?)</span>', html_text, re.DOTALL | re.IGNORECASE)
            or re.search(r'<time[^>]*>(.*?)</time>', html_text, re.DOTALL | re.IGNORECASE)
        )
        posted_date = _clean_html_text(time_match.group(1)) if time_match else None

    # Apply URL
    apply_url_match = (
        re.search(r'<a[^>]*class="[^"]*(?:apply-button|apply-btn)[^"]*"[^>]*href="([^"]+)"', html_text, re.IGNORECASE)
        or re.search(r'<a[^>]*href="([^"]+)"[^>]*>\s*Apply', html_text, re.IGNORECASE)
    )
    apply_url = apply_url_match.group(1).strip() if apply_url_match else None
    if apply_url:
        apply_url = html.unescape(apply_url)

    # Description
    desc_match = (
        re.search(r'<div[^>]*class="[^"]*show-more-less-html__markup[^"]*"[^>]*>(.*?)</div>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<section[^>]*class="[^"]*show-more-less-html[^"]*"[^>]*>(.*?)</section>', html_text, re.DOTALL | re.IGNORECASE)
        or re.search(r'<div[^>]*class="[^"]*decorated-job-posting__details[^"]*"[^>]*>(.*?)</div>', html_text, re.DOTALL | re.IGNORECASE)
    )
    description = _clean_html_text(desc_match.group(1)) if desc_match else ""

    # Criteria parsing (Seniority level, Employment type, Job function, Industries)
    criteria_items = re.findall(
        r'<li[^>]*>\s*<h3[^>]*class="[^"]*description__job-criteria-subheader[^"]*"[^>]*>(.*?)</h3>\s*<span[^>]*class="[^"]*description__job-criteria-text[^"]*"[^>]*>(.*?)</span>\s*</li>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    criteria_dict: dict[str, str] = {}
    for subheader, text_val in criteria_items:
        clean_sub = _clean_html_text(subheader).lower()
        clean_val = _clean_html_text(text_val)
        criteria_dict[clean_sub] = clean_val

    seniority_level = criteria_dict.get("seniority level") or criteria_dict.get("seniority")
    employment_type = criteria_dict.get("employment type")
    department = criteria_dict.get("job function") or criteria_dict.get("function")
    industries = criteria_dict.get("industries")

    # Work mode determination
    combined_text = f"{title} {location} {description} {employment_type or ''}".lower()
    if "remote" in combined_text or "מהבית" in combined_text:
        work_mode = WorkMode.REMOTE
    elif "hybrid" in combined_text or "היברידי" in combined_text:
        work_mode = WorkMode.HYBRID
    elif "on-site" in combined_text or "onsite" in combined_text:
        work_mode = WorkMode.ONSITE
    elif location:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = None

    # Tech stack extraction
    tech_text = f"{title} {description} {department or ''}"
    tech_stack = _extract_text_tech_keywords(tech_text)

    return {
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "posted_date": posted_date,
        "apply_url": apply_url,
        "seniority_level": seniority_level,
        "employment_type": employment_type,
        "department": department,
        "industries": industries,
        "work_mode": work_mode,
        "tech_stack": tech_stack,
    }


async def search_linkedin_jobs_api(
    keywords: str = "",
    location: str = LINKEDIN_DEFAULT_LOCATION,
    start: int = 0,
    work_mode: Optional[WorkMode] = None,
    f_WT: Optional[str] = None,
    f_TPR: Optional[str] = None,
    f_AL: Optional[bool] = None,
    f_E: Optional[str] = None,
    sort_by: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = LINKEDIN_REQUEST_TIMEOUT,
    max_retries: int = LINKEDIN_DEFAULT_MAX_RETRIES,
    backoff_factor: float = 1.0,
    base_url: str = LINKEDIN_SEARCH_API_URL,
) -> list[Job]:
    """Search LinkedIn jobs using the high-speed guest seeMoreJobPostings endpoint with backoff.

    Args:
        keywords: Search keywords or query string.
        location: Geographic location filter (defaults to Israel).
        start: Pagination offset integer.
        work_mode: Optional WorkMode enum value mapped to f_WT.
        f_WT: Optional raw workplace type parameter ('1'=onsite, '2'=remote, '3'=hybrid).
        f_TPR: Time posted range filter (e.g. 'r86400', 'r604800').
        f_AL: Easy apply filter toggle.
        f_E: Experience level filter string.
        sort_by: Sorting parameter ('R'=relevant, 'DD'=date posted).
        client: Optional shared httpx.AsyncClient instance.
        headers: Optional HTTP headers dict.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts on 429 rate limit or 5xx errors.
        backoff_factor: Multiplier for exponential backoff sleep.
        base_url: Base search API endpoint URL.

    Returns:
        list[Job]: Standardized Job listings tagged with 'linkedin'.
    """
    params: dict[str, Any] = {
        "keywords": keywords,
        "location": location,
        "start": start,
    }

    # Workplace / Work mode filter
    if f_WT:
        params["f_WT"] = f_WT
    elif work_mode and work_mode in WORK_MODE_TO_FWT:
        params["f_WT"] = WORK_MODE_TO_FWT[work_mode]

    if f_TPR:
        params["f_TPR"] = f_TPR
    if f_AL is not None:
        params["f_AL"] = "true" if f_AL else "false"
    if f_E:
        params["f_E"] = f_E
    if sort_by:
        params["sortBy"] = sort_by

    req_headers = headers or dict(LINKEDIN_HEADERS)
    should_close_client = False
    active_client = client
    if active_client is None:
        active_client = httpx.AsyncClient(timeout=timeout)
        should_close_client = True

    try:
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                response = await active_client.get(
                    base_url,
                    params=params,
                    headers=req_headers,
                    timeout=timeout,
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 200:
                    jobs = parse_linkedin_search_results(response.text)
                    logger.info(
                        "LinkedIn search request completed",
                        url=str(response.url),
                        status=response.status_code,
                        duration_ms=round(duration_ms, 2),
                        items_count=len(jobs),
                        source="linkedin",
                    )
                    return jobs

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_time = float(retry_after)
                        except ValueError:
                            sleep_time = backoff_factor * (2 ** attempt)
                    else:
                        sleep_time = min(10.0, backoff_factor * (2 ** attempt))

                    logger.warning(
                        "LinkedIn 429 Rate limited (attempt %d/%d), sleeping %.2fs",
                        attempt + 1,
                        max_retries,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    continue

                if response.status_code in (500, 502, 503, 504):
                    logger.warning(
                        "LinkedIn server error %d (attempt %d/%d)",
                        response.status_code,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                    continue

                logger.warning(
                    "LinkedIn search returned unexpected status %d for params %s",
                    response.status_code,
                    params,
                )
                return []

            except (httpx.RequestError, httpx.TimeoutException) as net_err:
                logger.warning(
                    "LinkedIn network error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    net_err,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                else:
                    return []

        return []
    finally:
        if should_close_client:
            await active_client.aclose()


class LinkedInSource(BaseJobSource):
    """Job source querying LinkedIn guest APIs with backoff rate-limiting and caching."""

    source_id: str = "linkedin"
    display_name: str = "LinkedIn"
    description: str = "LinkedIn Jobs guest search & job details API"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def __init__(
        self,
        base_url: str = LINKEDIN_SEARCH_API_URL,
        detail_url: str = LINKEDIN_JOB_DETAIL_URL,
        default_location: str = LINKEDIN_DEFAULT_LOCATION,
        headers: Optional[dict[str, str]] = None,
        cache_ttl_seconds: int = 3600,
        client: Optional[httpx.AsyncClient] = None,
        session_manager: Optional[Any] = None,
        timeout: float = LINKEDIN_REQUEST_TIMEOUT,
        health_timeout: float = LINKEDIN_HEALTH_TIMEOUT,
        max_retries: int = LINKEDIN_DEFAULT_MAX_RETRIES,
        rate_limit_delay_seconds: float = LINKEDIN_DEFAULT_RATE_LIMIT_DELAY,
        max_concurrency: int = 3,
    ) -> None:
        """Initialize LinkedInSource.

        Args:
            base_url: Search API endpoint URL.
            detail_url: Job posting detail endpoint base URL.
            default_location: Default geographic location filter.
            headers: HTTP headers dict. Defaults to LINKEDIN_HEADERS.
            cache_ttl_seconds: TTL for caching in seconds.
            client: Optional shared httpx.AsyncClient instance.
            session_manager: Optional Playwright SessionManager for authenticated actions.
            timeout: HTTP timeout in seconds.
            health_timeout: Health probe timeout in seconds.
            max_retries: Maximum retry attempts for 429 rate limit backoff.
            rate_limit_delay_seconds: Pacing delay between sequential requests.
            max_concurrency: Maximum concurrent request tasks.
        """
        self.base_url = base_url
        self.detail_url = detail_url.rstrip("/")
        self.default_location = default_location
        self.headers = headers or dict(LINKEDIN_HEADERS)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = client
        self.session_manager = session_manager
        self.timeout = timeout
        self.health_timeout = health_timeout
        self.max_retries = max_retries
        self.rate_limit_delay_seconds = rate_limit_delay_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)

        if self.session_manager is not None:
            self.supports_bookmarks = True

        self._search_cache: dict[tuple[tuple[str, str], ...], tuple[float, list[Job]]] = {}
        self._details_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def clear_cache(self) -> None:
        """Clear all in-memory search and details caches."""
        self._search_cache.clear()
        self._details_cache.clear()

    async def fetch_job_details(
        self,
        job_id: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch detailed information for a specific LinkedIn job posting.

        Args:
            job_id: Job identifier (with or without 'linkedin_' prefix).
            client: Optional httpx.AsyncClient to reuse.

        Returns:
            Optional[dict[str, Any]]: Detailed fields or None on failure.
        """
        raw_numeric_id = job_id.replace("linkedin_", "").strip()
        if not raw_numeric_id:
            return None

        now = time.time()
        if raw_numeric_id in self._details_cache:
            ts, cached_details = self._details_cache[raw_numeric_id]
            if now - ts < self.cache_ttl_seconds:
                return cached_details

        should_close_client = False
        active_client = client or self._client
        if active_client is None:
            active_client = httpx.AsyncClient(timeout=self.timeout)
            should_close_client = True

        url = f"{self.detail_url}/{raw_numeric_id}"
        try:
            async with self.semaphore:
                response = await active_client.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    details = parse_linkedin_job_details(response.text)
                    self._details_cache[raw_numeric_id] = (now, details)
                    return details
                logger.warning(
                    "LinkedIn detail endpoint returned status %d for job %s",
                    response.status_code,
                    raw_numeric_id,
                )
                return None
        except Exception as exc:
            logger.warning("Failed to fetch LinkedIn details for job %s: %s", raw_numeric_id, exc)
            return None
        finally:
            if should_close_client:
                await active_client.aclose()

    async def _query_feed(
        self,
        client: httpx.AsyncClient,
        keywords: str,
        location: str,
        start: int,
        work_mode: Optional[WorkMode],
    ) -> list[Job]:
        """Query a single page of results with cache and concurrency control."""
        cache_key = tuple(sorted([
            ("keywords", keywords),
            ("location", location),
            ("start", str(start)),
            ("work_mode", work_mode.value if work_mode else ""),
        ]))
        now = time.time()
        if cache_key in self._search_cache:
            ts, cached_jobs = self._search_cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return cached_jobs

        async with self.semaphore:
            jobs = await search_linkedin_jobs_api(
                keywords=keywords,
                location=location,
                start=start,
                work_mode=work_mode,
                client=client,
                headers=self.headers,
                timeout=self.timeout,
                max_retries=self.max_retries,
                base_url=self.base_url,
            )
            self._search_cache[cache_key] = (now, jobs)
            return jobs

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from LinkedIn matching optional preferences up to limit.

        Args:
            preferences: Optional JobPreferences filter.
            limit: Maximum number of jobs to return.

        Returns:
            list[Job]: Standardized Job listings tagged with 'linkedin'.
        """
        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close_client = True

        try:
            location = (
                (preferences.location.strip() if preferences and preferences.location else "")
                or self.default_location
            )
            work_mode = preferences.work_mode if preferences else None

            # Determine search keyword queries
            search_terms: list[str] = []
            if preferences:
                combined_terms = list(preferences.tech_stack) + list(preferences.keywords)
                if combined_terms:
                    search_terms = combined_terms[:3]

            if not search_terms:
                search_terms = [""]

            # Calculate pages needed
            pages_per_term = 1
            if limit > 25:
                pages_per_term = min(3, (limit + 24) // 25)

            tasks = []
            for term in search_terms:
                for page_idx in range(pages_per_term):
                    start_offset = page_idx * 25
                    tasks.append(
                        self._query_feed(
                            client=client,
                            keywords=term,
                            location=location,
                            start=start_offset,
                            work_mode=work_mode,
                        )
                    )

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
                    logger.warning("Error in LinkedIn fetch query: %s", res)

            # Apply preferences filtering if provided
            if preferences:
                all_jobs = filter_jobs(all_jobs, preferences)

            # Truncate to limit
            if limit and len(all_jobs) > limit:
                all_jobs = all_jobs[:limit]

            return all_jobs
        finally:
            if should_close_client:
                await client.aclose()

    async def check_health(self) -> bool:
        """Check the operational health of the LinkedIn Guest Jobs endpoint.

        Returns:
            bool: True if endpoint is reachable and responsive, False otherwise.
        """
        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.health_timeout)
            should_close_client = True

        try:
            params = {
                "keywords": "software",
                "location": self.default_location,
                "start": 0,
            }
            response = await client.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=self.health_timeout,
            )
            if response.status_code == 200 and response.text:
                return True
            logger.warning("LinkedIn health check returned status %d", response.status_code)
            return False
        except Exception as exc:
            logger.warning("LinkedIn health check failed: %s", exc)
            return False
        finally:
            if should_close_client:
                await client.aclose()

    async def bookmark_job(self, job_id: str) -> bool:
        """Bookmark a job listing if browser session manager is available.

        Args:
            job_id: ID of the job listing.

        Returns:
            bool: True if bookmarked successfully, False otherwise.
        """
        if not self.supports_bookmarks or not self.session_manager:
            return False
        if hasattr(self.session_manager, "bookmark_job"):
            try:
                return await self.session_manager.bookmark_job(job_id)
            except Exception as exc:
                logger.warning("Failed to bookmark LinkedIn job %s: %s", job_id, exc)
                return False
        return False
