"""Caching, CV keyword extraction, and direct API client logic for HireMeTech MCP server."""

from __future__ import annotations

import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from job_mcp.core.auth import BASE_URL
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

# Curated tech dictionary for CV and listing parsing
CURATED_TECH_KEYWORDS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Go", "Golang", "Rust", "Java", "Kotlin",
    "Scala", "C++", "C#", "C", ".NET", "Ruby", "PHP", "Swift", "Objective-C",
    "SQL", "HTML", "CSS", "Bash", "Shell", "R", "Dart", "Elixir", "Clojure",
    # Frameworks & Libraries
    "React", "Next.js", "Vue", "Vue.js", "Nuxt", "Angular", "Svelte",
    "Node.js", "Node", "Express", "FastAPI", "Django", "Flask", "Spring", "Spring Boot",
    "Ruby on Rails", "Rails", "Laravel", "ASP.NET", "GraphQL", "gRPC", "REST", "RESTful",
    "Redux", "Zustand", "TailwindCSS", "Bootstrap", "Prisma", "SQLAlchemy", "Pydantic",
    # Databases & Storage
    "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "SQLite", "MariaDB", "Neo4j", "Kafka", "RabbitMQ", "Celery",
    # DevOps, Cloud & Infrastructure
    "Docker", "Kubernetes", "K8s", "AWS", "Amazon Web Services", "GCP", "Google Cloud",
    "Azure", "Terraform", "Ansible", "Helm", "CI/CD", "GitHub Actions", "GitLab CI",
    "Jenkins", "CircleCI", "Linux", "Git", "Nginx", "Prometheus", "Grafana",
    # AI / ML / Data
    "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Pandas", "NumPy",
    "OpenAI", "LLM", "LangChain", "LlamaIndex", "Hugging Face", "NLP",
    "FastMCP", "Playwright", "Selenium", "Airflow", "Spark", "Hadoop",
]


class JobCache:
    """In-memory cache for job listings with TTL expiration."""

    def __init__(self, ttl_minutes: Optional[int] = None) -> None:
        """Initialize cache with TTL in minutes.

        Args:
            ttl_minutes: Optional cache time-to-live in minutes. Defaults to CACHE_TTL_MINUTES env or 60.
        """
        if ttl_minutes is None:
            env_ttl = os.getenv("CACHE_TTL_MINUTES", "60").strip()
            try:
                ttl_minutes = int(env_ttl)
            except ValueError:
                ttl_minutes = 60

        self.ttl_seconds: int = ttl_minutes * 60
        self._jobs: list[Job] = []
        self._last_updated: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Check if the cache has expired or contains no data."""
        if not self._jobs or self._last_updated <= 0:
            return True
        return (time.time() - self._last_updated) > self.ttl_seconds

    def update(self, jobs: list[Job]) -> None:
        """Update cached jobs and reset expiration timestamp.

        Args:
            jobs: New list of Job instances to cache.
        """
        self._jobs = list(jobs)
        self._last_updated = time.time()
        logger.info("JobCache updated with %d jobs (TTL: %ds)", len(self._jobs), self.ttl_seconds)

    def get_all(self) -> list[Job]:
        """Return all cached jobs.

        Returns:
            list[Job]: List of currently cached jobs.
        """
        return list(self._jobs)

    def get_by_id(self, job_id: str) -> Optional[Job]:
        """Find a job by its ID in the cache.

        Args:
            job_id: ID of the job to retrieve.

        Returns:
            Optional[Job]: Matching Job or None if not found.
        """
        for job in self._jobs:
            if job.job_id == job_id:
                return job
        return None

    def clear(self) -> None:
        """Clear all cached jobs and reset timestamp."""
        self._jobs = []
        self._last_updated = 0.0
        logger.info("JobCache cleared.")


def _extract_text_from_pdf(path: Path) -> str:
    """Safely extract text from a PDF file using available libraries or fallback."""
    # Attempt pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pypdf extraction failed: %s", exc)

    # Attempt fitz (PyMuPDF)
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(path))
        pages_text = [page.get_text() for page in doc]
        return "\n".join(pages_text)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("fitz extraction failed: %s", exc)

    # Fallback to binary string extraction
    try:
        with open(path, "rb") as f:
            content = f.read()
        # Find printable ASCII / UTF-8 sequences
        matches = re.findall(rb"[\x20-\x7E\r\n]{4,}", content)
        return "\n".join(m.decode("ascii", errors="ignore") for m in matches)
    except Exception as exc:
        logger.warning("PDF fallback text extraction failed for '%s': %s", path, exc)
        return ""


def _extract_text_from_docx(path: Path) -> str:
    """Extract text from a DOCX file using built-in zipfile and XML parsing."""
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # Find all text elements in Word XML
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = [node.text for node in tree.findall(".//w:t", namespaces) if node.text]
            return " ".join(texts)
    except Exception as exc:
        logger.warning("DOCX text extraction failed for '%s': %s", path, exc)
        return ""


def extract_cv_keywords(cv_path: str) -> list[str]:
    """Read a CV/resume file (text, pdf, docx) and extract technology keywords.

    Args:
        cv_path: Path to the resume/CV file.

    Returns:
        list[str]: Sorted, deduplicated list of detected technology keywords.
    """
    path = Path(cv_path).expanduser().resolve()
    if not path.exists():
        logger.warning("CV file path '%s' does not exist.", path)
        return []

    logger.info("Extracting keywords from CV: %s", path)
    text_content = ""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text_content = _extract_text_from_pdf(path)
    elif suffix == ".docx":
        text_content = _extract_text_from_docx(path)
    else:
        # Plain text, markdown, etc.
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        except Exception as exc:
            logger.warning("Failed reading text CV '%s': %s", path, exc)
            return []

    if not text_content:
        logger.warning("No readable text could be extracted from '%s'.", path)
        return []

    found_keywords: set[str] = set()
    text_lower = text_content.lower()

    for tech in CURATED_TECH_KEYWORDS:
        # Regex matching with word boundaries
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(tech.lower()) + r"(?![a-zA-Z0-9_])"
        if re.search(pattern, text_lower):
            found_keywords.add(tech)

    result = sorted(list(found_keywords), key=lambda s: s.lower())
    logger.info("Extracted %d keywords from CV '%s': %s", len(result), path.name, result)
    return result


def _extract_text_tech_keywords(text: str) -> list[str]:
    """Extract curated tech keywords present in a given string.

    Args:
        text: Input text string.

    Returns:
        list[str]: Matched tech keywords.
    """
    if not text:
        return []
    found: list[str] = []
    text_lower = text.lower()
    for tech in CURATED_TECH_KEYWORDS:
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(tech.lower()) + r"(?![a-zA-Z0-9_])"
        if re.search(pattern, text_lower):
            found.append(tech)
    return found


def _parse_salary_number(salary_str: str) -> Optional[int]:
    """Extract numeric salary value from salary text string."""
    clean = salary_str.lower().replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    # Look for '120k' or '120000'
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k", clean)
    if k_match:
        return int(float(k_match.group(1)) * 1000)

    num_match = re.findall(r"\b\d{4,7}\b", clean)
    if num_match:
        return int(num_match[-1])  # Use the highest number if range
    return None


def filter_jobs(jobs: list[Job], prefs: JobPreferences) -> list[Job]:
    """Filter and score job listings according to user preferences.

    Args:
        jobs: List of Job instances.
        prefs: JobPreferences configuration.

    Returns:
        list[Job]: Filtered and ranked list of Job instances sorted by match_score descending.
    """
    filtered: list[Job] = []

    # Prepare normalized exclusion keywords
    exclude_set = {k.strip().lower() for k in prefs.exclude_keywords if k.strip()}

    # Prepare desired skills from tech_stack, keywords, and CV
    desired_tech = {t.strip().lower() for t in prefs.tech_stack if t.strip()}
    desired_keywords = {k.strip().lower() for k in prefs.keywords if k.strip()}
    cv_keywords = set()
    if prefs.cv_path:
        extracted = extract_cv_keywords(prefs.cv_path)
        cv_keywords = {c.strip().lower() for c in extracted if c.strip()}

    total_desired = desired_tech | desired_keywords | cv_keywords

    for job in jobs:
        job_full_text = f"{job.title} {job.company} {job.location} {job.description} {' '.join(job.tech_stack)}".lower()

        # 1. Exclude keywords check
        if any(exc in job_full_text for exc in exclude_set):
            continue

        # 2. Work mode filter
        if prefs.work_mode:
            pref_mode = prefs.work_mode.value if isinstance(prefs.work_mode, WorkMode) else str(prefs.work_mode).lower()
            job_mode = job.work_mode.value if isinstance(job.work_mode, WorkMode) else str(job.work_mode or "").lower()

            if job_mode:
                if job_mode != pref_mode and pref_mode not in job_full_text:
                    continue
            else:
                # If job has no explicit work_mode, verify if text mentions preferred mode
                if pref_mode not in job_full_text:
                    continue

        # 3. Location filter
        if prefs.location:
            loc_pref = prefs.location.strip().lower()
            job_loc = job.location.lower()
            # If job is fully remote and preference didn't forbid remote, allow; otherwise check location string
            is_remote_job = job.work_mode == WorkMode.REMOTE or "remote" in job_loc
            if loc_pref not in job_loc and loc_pref not in job_full_text:
                if not (is_remote_job and "remote" in loc_pref):
                    continue

        # 4. Minimum salary filter
        if prefs.min_salary and job.salary_range:
            parsed_salary = _parse_salary_number(job.salary_range)
            if parsed_salary is not None and parsed_salary < prefs.min_salary:
                continue

        # 5. Compute match score (0 to 100)
        job_tech_tokens = {t.strip().lower() for t in job.tech_stack}
        job_words = set(re.findall(r"[a-zA-Z0-9_\.\#\+]+", job_full_text))
        all_job_tokens = job_tech_tokens | job_words

        if not total_desired:
            # If no skills or keywords specified in preferences, score is 100.0
            score = 100.0
        else:
            # Weights breakdown
            matched_tech = desired_tech & all_job_tokens
            matched_kw = desired_keywords & all_job_tokens
            matched_cv = cv_keywords & all_job_tokens

            score_components = []
            if desired_tech:
                score_components.append((len(matched_tech) / len(desired_tech)) * 50.0)
            if desired_keywords:
                score_components.append((len(matched_kw) / len(desired_keywords)) * 30.0)
            if cv_keywords:
                score_components.append((len(matched_cv) / len(cv_keywords)) * 20.0)

            raw_score = sum(score_components)
            # Normalize to 100 scale based on active components
            active_weights = (50.0 if desired_tech else 0.0) + (30.0 if desired_keywords else 0.0) + (20.0 if cv_keywords else 0.0)
            if active_weights > 0:
                score = (raw_score / active_weights) * 100.0
            else:
                score = 100.0

            # Bonus for exact title match
            title_lower = job.title.lower()
            if any(t in title_lower for t in desired_tech | desired_keywords):
                score = min(100.0, score + 5.0)

            score = round(max(0.0, min(100.0, score)), 1)

        job.match_score = score
        filtered.append(job)

    # Sort descending by match_score
    filtered.sort(key=lambda j: (j.match_score or 0.0), reverse=True)
    logger.info("Filtered %d jobs down to %d matching jobs.", len(jobs), len(filtered))
    return filtered


def parse_api_job_dict(raw: dict) -> Job:
    """Map a raw API job dictionary payload into a Job model.

    Args:
        raw: Dictionary representing a job listing from the HireMeTech API.

    Returns:
        Job: Populated Pydantic Job model instance.
    """
    job_id = str(raw.get("id") or raw.get("job_id") or raw.get("_id") or "").strip()
    title = str(raw.get("title") or "").strip()

    # Company resolution
    company = ""
    if raw.get("company_name"):
        company = str(raw["company_name"]).strip()
    elif isinstance(raw.get("company"), dict):
        company = str(raw["company"].get("name") or "").strip()
    elif isinstance(raw.get("company"), str):
        company = raw["company"].strip()

    # Location & Work Mode resolution
    location_str = ""
    work_mode: Optional[WorkMode] = None

    loc_obj = raw.get("location")
    if isinstance(loc_obj, dict):
        basic = loc_obj.get("basic") if isinstance(loc_obj.get("basic"), dict) else {}
        city = loc_obj.get("city") or basic.get("city") or ""
        display = basic.get("display_name") or loc_obj.get("full_address") or city
        location_str = str(display or city or "").strip()

        # Work model resolution inside location dict
        work_model = loc_obj.get("work_model")
        if isinstance(work_model, dict):
            wm_type = str(work_model.get("type") or "").strip().lower()
            if work_model.get("is_remote") is True or wm_type == "remote":
                work_mode = WorkMode.REMOTE
            elif work_model.get("is_hybrid") is True or wm_type == "hybrid":
                work_mode = WorkMode.HYBRID
            elif wm_type in ("onsite", "on-site", "office"):
                work_mode = WorkMode.ONSITE
        elif isinstance(work_model, str):
            wm_str = work_model.strip().lower()
            if "remote" in wm_str:
                work_mode = WorkMode.REMOTE
            elif "hybrid" in wm_str:
                work_mode = WorkMode.HYBRID
            elif "onsite" in wm_str or "on-site" in wm_str or "office" in wm_str:
                work_mode = WorkMode.ONSITE
    elif isinstance(loc_obj, str):
        location_str = loc_obj.strip()

    # Fallback work mode resolution if not already set
    if work_mode is None:
        raw_wm = raw.get("work_model") or raw.get("work_mode")
        if isinstance(raw_wm, dict):
            wm_type = str(raw_wm.get("type") or "").strip().lower()
            if raw_wm.get("is_remote") is True or wm_type == "remote":
                work_mode = WorkMode.REMOTE
            elif raw_wm.get("is_hybrid") is True or wm_type == "hybrid":
                work_mode = WorkMode.HYBRID
            elif wm_type in ("onsite", "on-site", "office"):
                work_mode = WorkMode.ONSITE
        elif isinstance(raw_wm, str):
            wm_str = raw_wm.strip().lower()
            if "remote" in wm_str:
                work_mode = WorkMode.REMOTE
            elif "hybrid" in wm_str:
                work_mode = WorkMode.HYBRID
            elif "onsite" in wm_str or "on-site" in wm_str or "office" in wm_str:
                work_mode = WorkMode.ONSITE
        elif raw.get("is_remote") is True:
            work_mode = WorkMode.REMOTE
        elif raw.get("is_hybrid") is True:
            work_mode = WorkMode.HYBRID
        elif "remote" in location_str.lower():
            work_mode = WorkMode.REMOTE
        elif "hybrid" in location_str.lower():
            work_mode = WorkMode.HYBRID
        elif "onsite" in location_str.lower() or "on-site" in location_str.lower():
            work_mode = WorkMode.ONSITE

    # Description and requirements combination
    desc = str(raw.get("description") or "").strip()
    reqs = str(raw.get("requirements") or "").strip()
    if desc and reqs and reqs not in desc:
        combined_desc = f"{desc}\n\n{reqs}"
    else:
        combined_desc = desc or reqs

    # Tech stack extraction
    tech_candidates: list[str] = []
    for field in ("skills_required", "skills", "tech_stack"):
        val = raw.get(field)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    tech_candidates.append(item.strip())
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("title") or item.get("skill")
                    if name and isinstance(name, str) and name.strip():
                        tech_candidates.append(name.strip())

    # Heuristic tech keyword extraction from title & description
    text_tech = _extract_text_tech_keywords(f"{title} {combined_desc}")
    tech_candidates.extend(text_tech)

    # Normalize & deduplicate preserving order/casing
    seen_lower = set()
    final_tech_stack: list[str] = []
    for t in tech_candidates:
        t_low = t.lower()
        if t_low not in seen_lower:
            seen_lower.add(t_low)
            final_tech_stack.append(t)

    # Salary resolution
    salary_range: Optional[str] = None
    sal_obj = raw.get("salary")
    if isinstance(sal_obj, dict):
        if sal_obj.get("formatted"):
            salary_range = str(sal_obj["formatted"]).strip()
        elif sal_obj.get("min") is not None or sal_obj.get("max") is not None:
            min_v = sal_obj.get("min")
            max_v = sal_obj.get("max")
            curr = sal_obj.get("currency", "ILS")
            if min_v is not None and max_v is not None:
                salary_range = f"{min_v:,} - {max_v:,} {curr}"
            elif min_v is not None:
                salary_range = f"{min_v:,}+ {curr}"
            elif max_v is not None:
                salary_range = f"Up to {max_v:,} {curr}"
    elif isinstance(sal_obj, str) and sal_obj.strip():
        salary_range = sal_obj.strip()
    elif raw.get("salary_range"):
        salary_range = str(raw["salary_range"]).strip()

    posted_date = raw.get("posted_date") or raw.get("reposted_at") or raw.get("created_at")
    if posted_date is not None:
        posted_date = str(posted_date).strip()

    url = raw.get("job_url") or raw.get("url") or raw.get("link")
    if url is not None:
        url = str(url).strip()

    is_bookmarked = bool(raw.get("is_saved") or raw.get("is_bookmarked") or False)
    match_score = float(raw["match_score"]) if raw.get("match_score") is not None else None

    return Job(
        job_id=job_id,
        title=title,
        company=company,
        location=location_str,
        work_mode=work_mode,
        tech_stack=final_tech_stack,
        description=combined_desc,
        salary_range=salary_range,
        posted_date=posted_date,
        url=url,
        is_bookmarked=is_bookmarked,
        match_score=match_score,
    )


async def fetch_jobs_via_api(
    request_context: Any,
    page: int = 1,
    size: int = 50,
    sort_by: str = "posted_date",
    sort_order: str = "desc",
) -> list[Job]:
    """Fetch jobs directly via the HireMeTech REST API using an active APIRequestContext.

    Args:
        request_context: Playwright APIRequestContext instance.
        page: Page number (1-indexed).
        size: Number of jobs per page.
        sort_by: Field to sort by ('posted_date', 'relevance', etc.).
        sort_order: Sort direction ('desc' or 'asc').

    Returns:
        list[Job]: Parsed list of Job objects.

    Raises:
        RuntimeError: If the API returns a non-200 status code.
    """
    url = f"{BASE_URL}/api/jobs/search?page={page}&size={size}&sort_by={sort_by}&sort_order={sort_order}&country=Israel&israeli=true"
    logger.info("Fetching jobs from API: %s", url)
    resp = await request_context.get(url)
    if resp.status != 200:
        error_msg = f"HireMe API returned status {resp.status} for {url}"
        logger.warning(error_msg)
        raise RuntimeError(error_msg)

    data = await resp.json()
    raw_jobs = data.get("jobs", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if isinstance(data, dict) and not raw_jobs and "data" in data:
        raw_jobs = data["data"]

    jobs = [parse_api_job_dict(item) for item in raw_jobs if isinstance(item, dict)]
    logger.info("Successfully fetched and parsed %d jobs from API", len(jobs))
    return jobs


async def fetch_saved_jobs_batch(
    request_context: Any,
    job_ids: list[str],
) -> dict[str, dict]:
    """Batch check saved status for job IDs via the HireMeTech REST API.

    Args:
        request_context: Playwright APIRequestContext instance.
        job_ids: List of job ID strings to query.

    Returns:
        dict[str, dict]: Mapping of job_id to saved status details.

    Raises:
        RuntimeError: If the API returns a non-200 status code.
    """
    if not job_ids:
        return {}

    clean_ids = [str(jid).strip() for jid in job_ids if str(jid).strip()]
    if not clean_ids:
        return {}

    ids_param = ",".join(clean_ids)
    url = f"{BASE_URL}/api/saved-jobs/check-batch?job_ids={ids_param}"
    logger.info("Batch checking %d saved jobs via API: %s", len(clean_ids), url)
    resp = await request_context.get(url)
    if resp.status != 200:
        error_msg = f"HireMe saved-jobs batch check returned status {resp.status} for {url}"
        logger.warning(error_msg)
        raise RuntimeError(error_msg)

    data = await resp.json()
    if isinstance(data, dict):
        return data.get("jobs", data)
    return {}


async def fetch_user_resume_profile(
    request_context: Any,
) -> dict:
    """Fetch the authenticated user's resume profile via the HireMeTech REST API.

    Args:
        request_context: Playwright APIRequestContext instance.

    Returns:
        dict: User resume profile dictionary.

    Raises:
        RuntimeError: If the API returns a non-200 status code.
    """
    url = f"{BASE_URL}/api/resume/profile"
    logger.info("Fetching user resume profile from API: %s", url)
    resp = await request_context.get(url)
    if resp.status != 200:
        error_msg = f"HireMe resume profile fetch returned status {resp.status} for {url}"
        logger.warning(error_msg)
        raise RuntimeError(error_msg)

    data = await resp.json()
    if isinstance(data, dict):
        return data
    return {}
