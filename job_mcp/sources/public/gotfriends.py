"""GotFriends job source — Israel's top tech recruiter with dedicated ML/AI job categories."""

from __future__ import annotations

import asyncio
import html
import os
import random
import re
import time
from typing import Optional
from urllib.parse import urljoin

import httpx

from job_mcp.core.api_client import filter_jobs
from job_mcp.core.section_parser import extract_clean_job_tech_stack, parse_job_sections
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources.base import BasePublicSource
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = os.getenv("REPO_URL", "https://github.com/TechJobMCP/TechJobMCP")
GOTFRIENDS_BASE_URL: str = "https://www.gotfriends.co.il"

# Key Israeli tech categories to crawl
GOTFRIENDS_CATEGORIES: dict[str, str] = {
    "ai": "/jobslobby/ai/",
    "ai_engineer": "/jobslobby/ai/ai-engineer/",
    "llm_engineer": "/jobslobby/ai/llm-engineer/",
    "algorithm": "/jobslobby/algorithm/",
    "machine_learning": "/jobslobby/algorithm/machine-learning/",
    "algorithm_engineer": "/jobslobby/algorithm/algorithm-engineer/",
    "data_scientist": "/jobslobby/algorithm/data-scientist/",
    "deep_learning": "/jobslobby/algorithm/deep-learning-engineer/",
    "software": "/jobslobby/software/",
    "backend": "/jobslobby/backend/",
}

REQUEST_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": f"TechJobMCP/1.0 (Job Aggregator; +{REPO_URL})",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}

MAX_CONCURRENT_REQUESTS: int = 2


def _strip_html(html_content: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html_content)
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_work_mode(location: str, description: str) -> WorkMode:
    """Detect work mode from location and description text."""
    combined = f"{location} {description}".lower()
    if "hybrid" in combined or "היברידי" in combined or "היברידית" in combined:
        return WorkMode.HYBRID
    if "remote" in combined or "מהבית" in combined or "עבודה מרחוק" in combined or "מרחוק" in combined:
        return WorkMode.REMOTE
    return WorkMode.ONSITE


def _extract_desc_html(html_text: str) -> str:
    """Extract inner content of <div class="desc"> preserving text within nested tags."""
    m = re.search(r'<div[^>]*class=["\'][^"\']*\bdesc\b[^"\']*["\'][^>]*>', html_text, re.IGNORECASE)
    if not m:
        return ""
    start_pos = m.end()
    depth = 1
    div_pattern = re.compile(r'<\s*(/)?\s*div\b[^>]*>', re.IGNORECASE)
    for tag in div_pattern.finditer(html_text, start_pos):
        if tag.group(1):  # </div>
            depth -= 1
            if depth == 0:
                inner = html_text[start_pos:tag.start()]
                inner = re.sub(r'<div[^>]*class=["\'][^"\']*title_c[^"\']*["\'][^>]*>.*?</div>', '', inner, flags=re.DOTALL | re.IGNORECASE)
                return inner
        else:
            depth += 1
    inner = html_text[start_pos:]
    inner = re.sub(r'<div[^>]*class=["\'][^"\']*title_c[^"\']*["\'][^>]*>.*?</div>', '', inner, flags=re.DOTALL | re.IGNORECASE)
    return inner


def parse_gotfriends_job_item(
    item_html: str,
    base_url: str = GOTFRIENDS_BASE_URL,
    category: Optional[str] = None,
) -> Optional[Job]:
    """Parse a single GotFriends HTML job card into a standardized Job model."""
    # Find link with href
    href_match = re.search(
        r'<a[^>]*href=["\'](/jobslobby/[^"\'\s>]+|https?://(?:www\.)?gotfriends\.co\.il/jobslobby/[^"\'\s>]+)["\']',
        item_html,
        re.IGNORECASE,
    )
    if not href_match:
        href_match = re.search(r'<a[^>]*href=["\']([^"\'\s>]+)["\']', item_html, re.IGNORECASE)
        if not href_match:
            return None

    raw_href = href_match.group(1).strip()

    # Extract job ID: /jobslobby/.../(\d+)/
    id_match = re.search(r"/(\d+)/?", raw_href)
    if id_match:
        match_id = id_match.group(1)
        job_id = f"gotfriends_{match_id}"
    else:
        p_match = re.search(r'\bp-(\d+)\b', item_html)
        if p_match:
            job_id = f"gotfriends_{p_match.group(1)}"
        else:
            return None

    # Full URL
    if raw_href.startswith("http://") or raw_href.startswith("https://"):
        full_url = raw_href
    else:
        full_url = urljoin(base_url, raw_href)

    # Title: <h2 class="title">(.*?)</h2>
    title_match = re.search(
        r'<h2[^>]*class=["\'][^"\']*title[^"\']*["\'][^>]*>(.*?)</h2>',
        item_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not title_match:
        title_match = re.search(r'<h2[^>]*>(.*?)</h2>', item_html, re.DOTALL | re.IGNORECASE)

    if title_match:
        raw_title = _strip_html(title_match.group(1))
        title = html.unescape(raw_title).strip()
    else:
        title = "Untitled"

    if not title or (title == "Untitled" and not title_match):
        return None

    # Location: <span class="info-data">(.*?)</span>
    loc_match = re.search(
        r'<span[^>]*class=["\'][^"\']*info-data[^"\']*["\'][^>]*>(.*?)</span>',
        item_html,
        re.DOTALL | re.IGNORECASE,
    )
    if loc_match:
        loc_raw = _strip_html(loc_match.group(1))
        loc_clean = html.unescape(loc_raw).strip()
        location = loc_clean if loc_clean else "Israel"
    else:
        location = "Israel"

    # Description
    desc_inner = _extract_desc_html(item_html)
    if not desc_inner:
        desc_match = re.search(
            r'<div[^>]*class=["\'][^"\']*desc[^"\']*["\'][^>]*>(.*?)</div>',
            item_html,
            re.DOTALL | re.IGNORECASE,
        )
        if desc_match:
            desc_inner = desc_match.group(1)

    raw_desc = _strip_html(desc_inner)
    description = html.unescape(raw_desc).strip()
    description = re.sub(r"^תיאור המשרה:\s*", "", description).strip()

    # Work mode
    work_mode = _detect_work_mode(location, description)

    # Section parsing & clean tech stack extraction
    sections = parse_job_sections(desc_inner or description)
    tech_stack = extract_clean_job_tech_stack(
        title=title,
        sections=sections,
        department=category,
        fallback_text=description,
    )

    return Job(
        job_id=job_id,
        title=title,
        company="GotFriends",
        location=location,
        url=full_url,
        apply_url=full_url,
        description=description[:2000] if description else "",
        tech_stack=tech_stack,
        source="gotfriends",
        work_mode=work_mode,
        department=category,
        requirements=sections.requirements or None,
        responsibilities=sections.responsibilities or None,
        company_overview=sections.company_overview or None,
    )


def extract_gotfriends_items(html_content: str) -> list[str]:
    """Split HTML into raw item card chunks."""
    chunks = re.split(r'<div[^>]*class=["\'][^"\']*\bitem\b[^"\']*["\'][^>]*>', html_content)
    return chunks[1:] if len(chunks) > 1 else []


class GotFriendsSource(BasePublicSource):
    """Job source aggregating roles from GotFriends (gotfriends.co.il), Israel's top tech recruiter.

    Features dedicated category hubs for AI, Machine Learning, Algorithms,
    Data Science, Deep Learning, and Software Engineering.
    """

    source_id = "gotfriends"
    display_name = "GotFriends"
    description = "Israel's top tech recruiter with dedicated AI, Machine Learning, and Algorithm job hubs"
    supports_auto_apply = True

    def __init__(
        self,
        categories: Optional[dict[str, str]] = None,
        base_url: str = GOTFRIENDS_BASE_URL,
        request_delay: float = 1.0,
    ) -> None:
        self._categories = categories if categories is not None else GOTFRIENDS_CATEGORIES
        self._base_url = base_url.rstrip("/")
        self._request_delay = request_delay
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_fetch_time: float = 0.0

    async def _fetch_category_jobs(
        self,
        client: httpx.AsyncClient,
        cat_name: str,
        cat_path: str,
    ) -> list[Job]:
        """Fetch and parse jobs for a single GotFriends category."""
        async with self._semaphore:
            if self._request_delay > 0:
                jitter = random.uniform(self._request_delay, self._request_delay * 1.5)
                await asyncio.sleep(jitter)

            url = f"{self._base_url}{cat_path if cat_path.startswith('/') else '/' + cat_path}"
            try:
                response = await client.get(url, headers=REQUEST_HEADERS, timeout=12.0)
                response.raise_for_status()
                html_text = response.text
                items = extract_gotfriends_items(html_text)
                jobs: list[Job] = []
                for item_html in items:
                    job = parse_gotfriends_job_item(item_html, base_url=self._base_url, category=cat_name)
                    if job is not None:
                        jobs.append(job)
                return jobs
            except httpx.HTTPStatusError as e:
                logger.warning("GotFriends %s HTTP error: %s", cat_name, e.response.status_code)
                return []
            except Exception as e:
                logger.warning("GotFriends %s fetch error: %s", cat_name, e)
                return []

    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings from all configured GotFriends category hubs."""
        if not self._categories:
            logger.warning("No GotFriends categories configured.")
            return []

        logger.info("Fetching jobs from %d GotFriends categories...", len(self._categories))
        async with httpx.AsyncClient() as client:
            tasks = [
                self._fetch_category_jobs(client, cat_name, cat_path)
                for cat_name, cat_path in self._categories.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[Job] = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.error("GotFriends category task failed unexpectedly: %s", result, exc_info=result)

        # Deduplicate jobs by job_id across categories
        seen_job_ids: set[str] = set()
        unique_jobs: list[Job] = []
        for job in all_jobs:
            if job.job_id not in seen_job_ids:
                seen_job_ids.add(job.job_id)
                unique_jobs.append(job)

        # Filter using candidate preferences if provided
        if preferences:
            unique_jobs = filter_jobs(unique_jobs, preferences)

        self._last_fetch_time = time.time()
        logger.info("GotFriends: fetched %d unique jobs (limit=%d)", len(unique_jobs), limit)
        return unique_jobs[:limit]

    async def check_health(self) -> bool:
        """Check health by fetching the AI jobs lobby page."""
        try:
            url = f"{self._base_url}/jobslobby/ai/"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=REQUEST_HEADERS, timeout=8.0)
                return resp.status_code == 200
        except Exception as e:
            logger.warning("GotFriends health check error: %s", e)
            return False


__all__ = [
    "GOTFRIENDS_BASE_URL",
    "GOTFRIENDS_CATEGORIES",
    "GotFriendsSource",
    "parse_gotfriends_job_item",
]
