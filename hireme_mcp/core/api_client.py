"""Caching, CV keyword extraction, and job filtering logic for HireMeTech MCP server."""

from __future__ import annotations

import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode
from hireme_mcp.utils.logger import get_logger

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
