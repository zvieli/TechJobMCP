"""Jobify (jobify360.co.il) job source implementation with JSON-LD parsing and snowball crawling."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx

from job_mcp.core.api_client import filter_jobs
from job_mcp.core.section_parser import extract_clean_job_tech_stack, parse_job_sections
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BasePublicSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

JOBIFY_BASE_URL: str = "https://jobify360.co.il"

JOBIFY_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}

DEFAULT_SEED_URLS: list[str] = [
    "https://jobify360.co.il/",
    "https://jobify360.co.il/myjob-roles/ai-engineer-5752190",
    "https://jobify360.co.il/myjob-roles/software-engineer",
    "https://jobify360.co.il/jobs/191_302995-emp",
    "https://jobify360.co.il/jobs/8821105-aj",
]

DEFAULT_JOBIFY_SEED_URLS: list[str] = DEFAULT_SEED_URLS


_ASSET_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".webp",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
)


def extract_jsonld_job_postings(html_content: str) -> list[dict[str, Any]]:
    """Extract all schema.org/JobPosting objects from JSON-LD script blocks in HTML.

    Args:
        html_content: Raw HTML page text.

    Returns:
        list[dict[str, Any]]: Parsed JobPosting JSON dictionaries.
    """
    postings: list[dict[str, Any]] = []
    script_pattern = re.compile(
        r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.DOTALL | re.IGNORECASE,
    )

    for match in script_pattern.finditer(html_content):
        script_text = match.group(1).strip()
        if not script_text:
            continue
        try:
            data = json.loads(script_text)
        except Exception:
            continue

        if isinstance(data, dict):
            # Check direct @type
            t = data.get("@type")
            if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                postings.append(data)
            # Check @graph array
            graph = data.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict):
                        item_t = item.get("@type")
                        if item_t == "JobPosting" or (isinstance(item_t, list) and "JobPosting" in item_t):
                            postings.append(item)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item_t = item.get("@type")
                    if item_t == "JobPosting" or (isinstance(item_t, list) and "JobPosting" in item_t):
                        postings.append(item)

    return postings


def extract_related_job_urls(html_content: str, base_url: str = JOBIFY_BASE_URL) -> list[str]:
    """Extract recommended and related job URLs from HTML for snowball crawling.

    Args:
        html_content: Raw HTML page text.
        base_url: Base domain to prepend to relative URLs.

    Returns:
        list[str]: Deduplicated absolute job detail URLs.
    """
    href_pattern = re.compile(
        r'href=[\"\'](/jobs/[^\"\'#\s]+|https?://(?:www\.)?jobify360\.co\.il/jobs/[^\"\'#\s]+)[\"\']',
        re.IGNORECASE,
    )

    urls: list[str] = []
    seen: set[str] = set()

    for match in href_pattern.finditer(html_content):
        raw_href = match.group(1).strip()
        if any(raw_href.lower().endswith(ext) for ext in _ASSET_EXTENSIONS):
            continue

        # Build absolute URL
        if raw_href.startswith("/"):
            full_url = urljoin(base_url, raw_href)
        else:
            full_url = raw_href

        # Strip fragment
        parsed = urlparse(full_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url = f"{clean_url}?{parsed.query}"

        # Filter out generic /jobs index page itself
        if parsed.path.rstrip("/") in ("/jobs", ""):
            continue

        if clean_url not in seen:
            seen.add(clean_url)
            urls.append(clean_url)

    return urls


def parse_jobify_position(jsonld: dict[str, Any], url: str = "") -> Job:
    """Parse schema.org/JobPosting dictionary into a standardized Job model.

    Args:
        jsonld: Parsed schema.org/JobPosting JSON-LD dictionary.
        url: URL of the job page if available.

    Returns:
        Job: Normalized Job object tagged with 'jobify'.
    """
    # 1. Title
    raw_title = str(jsonld.get("title") or "Untitled").strip()
    title = html.unescape(raw_title)

    # 2. Company
    hiring_org = jsonld.get("hiringOrganization")
    if isinstance(hiring_org, dict):
        raw_company = str(hiring_org.get("name") or "Confidential").strip()
    elif hiring_org is not None:
        raw_company = str(hiring_org).strip()
    else:
        raw_company = "Confidential"
    company = html.unescape(raw_company)

    # 3. Identifier / Job ID
    raw_id: Optional[str] = None
    identifier = jsonld.get("identifier")
    if isinstance(identifier, dict):
        raw_id = identifier.get("value") or identifier.get("name")
    elif identifier is not None:
        raw_id = str(identifier).strip()

    if not raw_id and url:
        m = re.search(r"/jobs/([a-zA-Z0-9_\-]+)", url)
        if m:
            raw_id = m.group(1)

    if not raw_id:
        raw_id = hashlib.md5(f"{title}_{company}_{url}".encode("utf-8")).hexdigest()[:10]

    raw_id_str = str(raw_id).strip()
    job_id = raw_id_str if raw_id_str.startswith("jobify_") else f"jobify_{raw_id_str}"

    # 4. Location
    location_data = jsonld.get("jobLocation")
    loc_parts: list[str] = []
    if isinstance(location_data, dict):
        addr = location_data.get("address")
        if isinstance(addr, dict):
            for key in ("addressLocality", "addressRegion", "addressCountry"):
                val = addr.get(key)
                if val and str(val).strip() and str(val).strip() not in loc_parts:
                    loc_parts.append(str(val).strip())
        elif isinstance(addr, str) and addr.strip():
            loc_parts.append(addr.strip())
        elif location_data.get("name"):
            loc_parts.append(str(location_data["name"]).strip())
    elif isinstance(location_data, str) and location_data.strip():
        loc_parts.append(location_data.strip())
    location_str = html.unescape(", ".join(loc_parts)) if loc_parts else ""

    # 5. Description
    raw_desc = str(jsonld.get("description") or "")
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = html.unescape(clean_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # 6. Work mode
    job_loc_type = str(jsonld.get("jobLocationType") or "").strip().upper()
    is_remote = job_loc_type == "TELECOMMUTE"
    combined = f"{title} {clean_desc} {location_str}".lower()

    if (
        is_remote
        or "remote" in combined
        or "מהבית" in combined
        or "עבודה מהבית" in combined
        or "מרחוק" in combined
    ):
        work_mode = WorkMode.REMOTE
    elif "hybrid" in combined or "היברידי" in combined or "משולב" in combined:
        work_mode = WorkMode.HYBRID
    elif "onsite" in combined or "on-site" in combined:
        work_mode = WorkMode.ONSITE
    else:
        work_mode = WorkMode.ONSITE if location_str else None

    # 7. Salary range
    sal = jsonld.get("baseSalary") or jsonld.get("estimatedSalary")
    salary_range: Optional[str] = None
    if isinstance(sal, dict):
        val = sal.get("value")
        if isinstance(val, dict):
            min_val = val.get("minValue") or ""
            max_val = val.get("maxValue") or ""
            curr = sal.get("currency") or ""
            unit = sal.get("unitText") or ""
            salary_range = f"{min_val} - {max_val} {curr} {unit}".strip()
        elif val is not None:
            salary_range = str(val).strip()
    elif sal is not None:
        salary_range = str(sal).strip()

    # 8. Posted date
    date_posted = jsonld.get("datePosted")
    posted_date = str(date_posted).strip() if date_posted else None

    # 9. Section parsing & Tech stack extraction
    sections = parse_job_sections(raw_desc or clean_desc)
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        fallback_text=clean_desc,
    )

    # 10. URL and apply URL
    resolved_url = url or (str(jsonld.get("url")) if jsonld.get("url") else None)
    apply_url = resolved_url

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
        url=resolved_url,
        apply_url=apply_url,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
        source="jobify",
        sources=["jobify"],
    )


class JobifySource(BasePublicSource):
    """Job source implementation for Jobify (jobify360.co.il) platform."""

    source_id: str = "jobify"
    display_name: str = "Jobify"
    description: str = "Israeli AI job matching platform with JSON-LD structured postings"
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False
    timeout: float = 18.0

    def __init__(
        self,
        seed_urls: Optional[list[str]] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 18.0,
        max_crawl_pages: int = 15,
        concurrency: int = 4,
    ) -> None:
        """Initialize JobifySource with seed URLs and optional HTTP client.

        Args:
            seed_urls: Initial URLs to crawl. Defaults to DEFAULT_JOBIFY_SEED_URLS.
            client: Optional shared httpx.AsyncClient instance.
            timeout: HTTP request timeout in seconds.
            max_crawl_pages: Maximum number of pages to crawl during snowball fetch.
            concurrency: Number of concurrent HTTP requests when crawling pages.
        """
        self.seed_urls = list(seed_urls) if seed_urls is not None else list(DEFAULT_JOBIFY_SEED_URLS)
        self._client = client
        self.timeout = timeout
        self.max_crawl_pages = max_crawl_pages
        self.concurrency = concurrency

    async def check_health(self) -> bool:
        """Check the operational health of Jobify platform.

        Returns:
            bool: True if Jobify is reachable and returns HTTP 200, False otherwise.
        """
        should_close = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(headers=JOBIFY_HEADERS, timeout=self.timeout)
            should_close = True

        try:
            target_url = self.seed_urls[0] if self.seed_urls else JOBIFY_BASE_URL
            response = await client.get(target_url, headers=JOBIFY_HEADERS, timeout=self.timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Jobify health check failed: %s", exc)
            return False
        finally:
            if should_close:
                await client.aclose()

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from Jobify using snowball crawling and JSON-LD extraction.

        Args:
            preferences: Optional JobPreferences for filtering.
            limit: Maximum number of jobs to fetch.

        Returns:
            list[Job]: Standardized Job listings tagged with 'jobify'.
        """
        should_close = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(headers=JOBIFY_HEADERS, timeout=self.timeout, follow_redirects=True)
            should_close = True

        jobs: list[Job] = []
        seen_job_ids: set[str] = set()
        visited_urls: set[str] = set()
        url_queue: list[str] = list(self.seed_urls)
        sem = asyncio.Semaphore(self.concurrency)

        async def _fetch_page(target_url: str) -> tuple[str, Optional[str]]:
            async with sem:
                try:
                    resp = await client.get(target_url, headers=JOBIFY_HEADERS, timeout=self.timeout)
                    if resp.status_code == 200:
                        return target_url, resp.text
                    logger.debug("Jobify crawl URL %s returned status %d", target_url, resp.status_code)
                except Exception as exc:
                    logger.warning("Error fetching Jobify page %s: %s", target_url, exc)
                return target_url, None

        try:
            while url_queue and len(visited_urls) < self.max_crawl_pages:
                if not preferences and limit and len(jobs) >= limit:
                    break

                batch_size = min(self.concurrency, self.max_crawl_pages - len(visited_urls))
                batch_urls: list[str] = []
                while url_queue and len(batch_urls) < batch_size:
                    candidate = url_queue.pop(0)
                    if candidate not in visited_urls and candidate not in batch_urls:
                        batch_urls.append(candidate)

                if not batch_urls:
                    break

                for u in batch_urls:
                    visited_urls.add(u)

                page_results = await asyncio.gather(*(_fetch_page(u) for u in batch_urls))
                for page_url, html_content in page_results:
                    if not html_content:
                        continue

                    # Extract JSON-LD job postings
                    postings = extract_jsonld_job_postings(html_content)
                    for posting in postings:
                        try:
                            job = parse_jobify_position(posting, url=page_url)
                            if job.job_id not in seen_job_ids:
                                seen_job_ids.add(job.job_id)
                                jobs.append(job)
                        except Exception as exc:
                            logger.warning("Error parsing Jobify position from %s: %s", page_url, exc)

                    # Extract recommended/related job links for snowball crawling
                    related_urls = extract_related_job_urls(html_content)
                    for r_url in related_urls:
                        if r_url not in visited_urls and r_url not in url_queue:
                            url_queue.append(r_url)

                if preferences and limit:
                    filtered_preview = filter_jobs(jobs, preferences, enable_semantic=False, enable_system1=False)
                    if len(filtered_preview) >= limit:
                        break
        except Exception as exc:
            logger.warning("Unhandled error during Jobify fetch_jobs: %s", exc)
        finally:
            if should_close:
                await client.aclose()

        # Apply preferences filtering if provided
        if preferences:
            jobs = filter_jobs(jobs, preferences, enable_semantic=False, enable_system1=False)

        # Truncate to limit
        if limit and len(jobs) > limit:
            jobs = jobs[:limit]

        return jobs


__all__ = [
    "DEFAULT_JOBIFY_SEED_URLS",
    "DEFAULT_SEED_URLS",
    "JOBIFY_BASE_URL",
    "JOBIFY_HEADERS",
    "JobifySource",
    "extract_jsonld_job_postings",
    "extract_related_job_urls",
    "parse_jobify_position",
]
