"""Caching, CV keyword extraction, and direct API client logic for Tech Job  MCP server."""

from __future__ import annotations

import asyncio
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from job_mcp.core.auth import BASE_URL
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences, WorkMode
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

# Curated tech dictionary for CV and listing parsing
CURATED_TECH_KEYWORDS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Go", "Golang", "Rust", "Java", "Kotlin",
    "Scala", "C++", "C#", "C", ".NET", "Ruby", "PHP", "Swift", "Objective-C",
    "SQL", "HTML", "CSS", "Bash", "Shell", "R", "Dart", "Elixir", "Clojure",
    # Web3 & Blockchain
    "Solidity", "Noir", "Web3", "Smart Contracts", "Smart Contract Development",
    "EVM", "Hardhat", "Foundry", "Ethers.js", "Viem", "IPFS", "ZK", "MPT",
    "Keccak256", "Merkle Patricia Trie",
    # Frameworks & Libraries
    "React", "Next.js", "Vue", "Vue.js", "Nuxt", "Angular", "Svelte",
    "Node.js", "Node", "Express", "FastAPI", "Django", "Flask", "Spring", "Spring Boot",
    "Ruby on Rails", "Rails", "Laravel", "ASP.NET", "GraphQL", "gRPC", "REST", "RESTful",
    "Redux", "Zustand", "TailwindCSS", "Bootstrap", "Prisma", "SQLAlchemy", "Pydantic",
    "Supabase", "TRPC", "AsyncIO", "Vite", "UltraHonk", "bb.js",
    # Databases & Storage
    "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "SQLite", "MariaDB", "Neo4j", "Kafka", "RabbitMQ", "Celery",
    "Azure Cosmos DB", "Cosmos DB", "Gremlin API", "Blob Storage", "Azure Blob Storage",
    # DevOps, Cloud & Infrastructure
    "Docker", "Kubernetes", "K8s", "AWS", "Amazon Web Services", "GCP", "Google Cloud",
    "Azure", "Azure Functions", "Azure Durable Functions", "Container Apps", "Azure Container Apps",
    "Terraform", "Ansible", "Helm", "CI/CD", "GitHub Actions", "GitLab CI",
    "Jenkins", "CircleCI", "Linux", "Git", "Nginx", "Prometheus", "Grafana",
    # AI / LLM / Agentic / ML / Data
    "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Pandas", "NumPy",
    "OpenAI", "LLM", "LangChain", "LlamaIndex", "Hugging Face", "NLP", "OCR",
    "GraphRAG", "LangGraph", "RAG", "Agentic", "Vector DB", "ChromaDB", "Chroma",
    "Pinecone", "Qdrant", "Weaviate", "CrewAI", "Autogen", "vLLM", "Ollama",
    "LangSmith", "Semantic Kernel", "Transformers", "Fine-Tuning", "Embeddings",
    "FastMCP", "Playwright", "Selenium", "Airflow", "Spark", "Hadoop",
    "Azure AI Search", "Azure Document Intelligence", "Azure AI Document Intelligence",
    "Document Intelligence", "ClinicalBERT", "NetworkX", "Cytoscape", "Leiden",
    "Machine Learning", "Operating Systems", "Data Structures",
]

RESUME_STOPWORDS: set[str] = {
    "THE", "AND", "FOR", "WITH", "FROM", "PRESENT", "MARCH", "FEBRUARY", "JANUARY", "APRIL",
    "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    "SUMMER", "WINTER", "FALL", "SPRING", "COMPUTER", "SCIENCE", "EXPERIENCE", "PROJECTS",
    "EDUCATION", "CLIENT", "PROJECT", "SYSTEMS", "SUMMARY", "EXPECTED", "COURSES",
    "DEVELOPER", "ARCHITECT", "ENGINEER", "LEGAL", "HEBREW", "MAINNET", "KEY", "ALL",
    "NEW", "OUR", "WHO", "HOW", "WHY", "LINKEDIN", "GITHUB", "EMAIL", "PHONE", "TEL",
    "AVIV", "ISRAEL", "UNIVERSITY", "COLLEGE", "DEGREE", "BSC", "MSC", "PHD", "HIT",
    "IDF", "PROFESSIONAL", "WORK", "STRONG", "BACKGROUND", "SCRIPT", "LEVEL",
}

MULTI_TOKEN_TECH_PHRASES: list[str] = [
    "Azure AI Search", "Azure Cosmos DB", "Azure Functions", "Azure Durable Functions",
    "Azure AI Document Intelligence", "Azure Document Intelligence", "Document Intelligence",
    "Azure Container Apps", "Container Apps", "Azure Blob Storage", "Blob Storage",
    "Smart Contract Development", "Smart Contracts", "Gremlin API",
    "Solidity", "Noir", "Foundry", "Hardhat", "Viem", "Ollama", "Pandas", "Python", "Rust",
    "React", "Docker", "Linux", "Git", "C#", "C++", ".NET", "ClinicalBERT", "NetworkX",
    "TailwindCSS", "Vite", "UltraHonk", "Axiom", "Axiom V3", "Cytoscape", "Leiden",
    "Merkle Patricia Trie", "Hugging Face", "Scikit-Learn", "Operating Systems", "Data Structures",
    "Machine Learning", "Ethers.js", "Node.js", "bb.js", "IPFS", "Keccak256",
]

MONTHS_AND_NOISE: set[str] = {
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
    "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER", "PRESENT", "DEVELOPER",
    "ENGINEER", "ARCHITECT", "HOLON", "TELAVIV", "ISRAEL", "SUMMER", "WINTER",
    "FALL", "SPRING", "SUMMARY", "EXPERIENCE", "EDUCATION", "CLASSIFICATION",
    "PLATFORM",
}

CORE_LANGUAGES_FRAMEWORKS: set[str] = {
    "python", "typescript", "javascript", "go", "golang", "rust", "java", "kotlin",
    "scala", "c++", "c#", "c", ".net", "ruby", "php", "swift", "sql",
    "react", "next.js", "vue", "vue.js", "angular", "svelte", "node.js", "node",
    "fastapi", "django", "flask", "spring", "spring boot", "express", "graphql", "rest", "grpc",
    "html", "css", "tailwindcss", "redux", "vite",
}

SPECIALIZED_COMPETENCIES: set[str] = {
    "ai", "langgraph", "graphrag", "rag", "langchain", "llamaindex", "pytorch",
    "tensorflow", "scikit-learn", "pandas", "numpy", "ollama", "vllm", "vector db",
    "chromadb", "chroma", "pinecone", "qdrant", "weaviate", "crewai", "autogen",
    "langsmith", "semantic kernel", "transformers", "fine-tuning", "embeddings", "llm", "nlp",
    "docker", "kubernetes", "k8s", "aws", "amazon web services", "gcp", "google cloud",
    "azure", "azure ai search", "azure container apps", "container apps", "terraform",
    "ci/cd", "postgresql", "postgres", "redis", "mongodb", "solidity", "noir", "web3",
    "smart contracts", "smart contract development", "foundry", "hardhat", "ethers.js", "viem",
    "zk", "evm", "ansible", "prometheus", "grafana", "helm", "linux", "git",
}


class JobCache:
    """In-memory cache for job listings with TTL expiration."""

    def __init__(self, ttl_minutes: Optional[int] = None) -> None:
        """Initialize cache with TTL in minutes.

        Args:
            ttl_minutes: Optional cache time-to-live in minutes. Defaults to CACHE_TTL_MINUTES env or 120.
        """
        if ttl_minutes is None:
            env_ttl = os.getenv("CACHE_TTL_MINUTES", "120").strip()
            try:
                ttl_minutes = int(env_ttl)
            except ValueError:
                ttl_minutes = 120

        self.ttl_seconds: int = ttl_minutes * 60
        self._jobs: list[Job] = []
        self._last_updated: float = 0.0
        self.dismissed_ids: set[str] = set()
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_stale(self) -> bool:
        """Check if the cache has expired or contains no data."""
        if not self._jobs or self._last_updated <= 0:
            return True
        return (time.time() - self._last_updated) > self.ttl_seconds

    def dismiss(self, job_id: str) -> None:
        """Dismiss a job by ID, adding to dismissed_ids and removing from cached jobs.

        Args:
            job_id: ID of the job to dismiss.
        """
        str_id = str(job_id)
        self.dismissed_ids.add(str_id)
        self._jobs = [j for j in self._jobs if j.job_id != str_id]
        logger.info("Job %s dismissed and removed from JobCache.", str_id)

    def update(self, jobs: list[Job]) -> None:
        """Update cached jobs and reset expiration timestamp.

        Args:
            jobs: New list of Job instances to cache (dismissed jobs are filtered out).
        """
        self._jobs = [j for j in jobs if j.job_id not in self.dismissed_ids]
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


def _extract_text_from_file(path: Path) -> str:
    """Extract text from a file (.pdf, .docx, or plain text)."""
    if not path.is_file():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(path)
    elif suffix == ".docx":
        return _extract_text_from_docx(path)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as exc:
            logger.warning("Failed reading text file '%s': %s", path, exc)
            return ""


def _extract_text_from_source(cv_source: Optional[Union[str, Path]]) -> str:
    """Extract text from a CV path, Path object, or raw string content."""
    if cv_source is None:
        resolved = resolve_cv_path(None)
        if resolved and resolved.is_file():
            return _extract_text_from_file(resolved)
        return ""

    if isinstance(cv_source, Path):
        try:
            if cv_source.is_file():
                return _extract_text_from_file(cv_source.resolve())
            resolved = resolve_cv_path(str(cv_source))
            if resolved and resolved.is_file():
                return _extract_text_from_file(resolved)
        except Exception:
            pass
        return ""

    if isinstance(cv_source, str):
        cleaned = cv_source.strip()
        if not cleaned:
            return ""

        # If multiline, treat directly as raw text content
        if "\n" in cleaned:
            return cleaned

        # Check if it points directly to an existing file
        try:
            p = Path(cleaned).expanduser()
            if p.is_file():
                return _extract_text_from_file(p.resolve())
        except Exception:
            pass

        # Determine if string looks like a file path
        looks_like_file_path = (
            cleaned.lower().endswith((".pdf", ".docx", ".txt", ".rtf", ".md", ".doc"))
            or (len(cleaned.split()) <= 4 and ("/" in cleaned or "\\" in cleaned or "." in cleaned))
        )

        if looks_like_file_path:
            try:
                resolved = resolve_cv_path(cleaned)
                if resolved and resolved.is_file():
                    return _extract_text_from_file(resolved)
            except Exception:
                pass
            return ""

        # Otherwise, treat as raw text content
        return cleaned

    return ""


def resolve_cv_path(cv_path: Optional[str] = None) -> Optional[Path]:
    """Resolve the CV file path based on explicit parameter, environment variable, or fallback discovery.

    Resolution candidate order:
    1. If cv_path is provided and exists directly on disk, return Path(cv_path).expanduser().resolve().
    2. If cv_path is provided but does not exist directly, search fallback candidate order:
       a. DEFAULT_CV_PATH environment variable if set.
       b. Candidate files matching basename or standard filenames:
          - /app/<basename>, /app/cv.pdf, /app/resume.pdf, /app/lior_zvieli_cv.pdf
          - cwd / <basename>, cwd / "cv.pdf", cwd / "resume.pdf", cwd / "lior_zvieli_cv.pdf"
       c. Glob matches in cwd (*cv*.pdf, *.pdf) and /app (*.pdf).
       If a candidate exists, log an info message and return candidate.resolve().
    3. If cv_path is None or empty:
       Search fallback candidates (DEFAULT_CV_PATH, standard container/workspace paths, glob matches)
       and return the first existing resolved candidate.

    Returns:
        Optional[Path]: The first candidate that is an existing file, resolved. None if none exist.
    """
    raw_path = str(cv_path).strip() if cv_path is not None else ""

    # 1. Explicit cv_path if provided and exists directly
    if raw_path:
        p = Path(raw_path).expanduser()
        try:
            if p.is_file():
                return p.resolve()
        except (OSError, PermissionError):
            pass

    # Extract basename if path provided (handling both / and \ path separators)
    clean_name = re.split(r"[/\\]+", raw_path)[-1] if raw_path else None
    base_name = clean_name if clean_name and clean_name not in (".", "..") else None

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(cand: Path) -> None:
        try:
            resolved_cand = cand.resolve()
            if resolved_cand not in seen:
                seen.add(resolved_cand)
                candidates.append(cand)
        except Exception:
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)

    # 2. DEFAULT_CV_PATH environment variable if set
    env_cv = os.getenv("DEFAULT_CV_PATH")
    if env_cv and env_cv.strip():
        add_candidate(Path(env_cv.strip()).expanduser())

    # 3. Container paths (/app)
    app_dir = Path("/app")
    if base_name:
        add_candidate(app_dir / base_name)
    add_candidate(app_dir / "cv.pdf")
    add_candidate(app_dir / "resume.pdf")
    add_candidate(app_dir / "lior_zvieli_cv.pdf")

    # 4. Local workspace paths (cwd)
    cwd = Path.cwd()
    if base_name:
        add_candidate(cwd / base_name)
    add_candidate(cwd / "cv.pdf")
    add_candidate(cwd / "resume.pdf")
    add_candidate(cwd / "lior_zvieli_cv.pdf")

    # 5. Any .pdf in working directory matching *cv*.pdf or *.pdf
    try:
        cv_glob = sorted(cwd.glob("*cv*.pdf"))
        all_glob = sorted(cwd.glob("*.pdf"))
        for p in cv_glob + all_glob:
            add_candidate(p)
    except Exception:
        pass

    # 6. Any .pdf in /app if /app directory exists
    if app_dir.is_dir():
        try:
            for p in sorted(app_dir.glob("*.pdf")):
                add_candidate(p)
        except Exception:
            pass

    for candidate in candidates:
        try:
            if candidate.is_file():
                resolved = candidate.resolve()
                if raw_path:
                    logger.info(
                        "CV path '%s' not found directly; falling back to '%s'",
                        raw_path,
                        resolved,
                    )
                return resolved
        except (OSError, PermissionError):
            continue

    return None


def _clean_and_canonicalize_token(tok: str, canonical_map: dict[str, str]) -> Optional[str]:
    """Clean and canonicalize a potential skill token, filtering out noise and stopwords.

    Args:
        tok: Raw token candidate.
        canonical_map: Dictionary mapping lowercase terms to canonical display strings.

    Returns:
        Optional[str]: Canonicalized skill name or None if filtered as noise/stopword.
    """
    tok = tok.strip()
    # Strip leading/trailing bullets, quotes, dashes, brackets, parens
    tok = re.sub(r"^[•*—\-\s\"'()\[\]{}]+|[•*—\-\s\"'()\[\]{}]+$", "", tok).strip()
    if not tok:
        return None
    if tok.upper() in RESUME_STOPWORDS:
        return None
    # Remove single character items (except valid language names like C, R)
    if len(tok) == 1 and tok.upper() not in ("C", "R"):
        return None
    # Filter pure numbers or floating numbers
    if re.match(r"^\d+(\.\d+)?$", tok):
        return None

    tok_lower = tok.lower()
    if tok_lower in canonical_map:
        return canonical_map[tok_lower]

    # Filter glued PascalCase noise (e.g. OracleMarch, ClassificationDeveloper, ScienceHolon)
    if re.match(r"^[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*$", tok):
        subwords = re.findall(r"[A-Z]+[a-z0-9]*", tok)
        if any(sw.upper() in MONTHS_AND_NOISE for sw in subwords):
            return None

    # Retain original casing if mixed/upper, or Title Case if all lower
    if tok.islower():
        return tok.title()
    return tok


def extract_dynamic_cv_skills(text_content: str) -> list[str]:
    """Discovers skills, technologies, frameworks, tools, libraries, and languages dynamically from text.

    Args:
        text_content: Raw text content from CV or resume.

    Returns:
        list[str]: Sorted, deduplicated list of dynamic skills.
    """
    if not text_content:
        return []

    # Build canonical casing map
    canonical_map: dict[str, str] = {kw.lower(): kw for kw in CURATED_TECH_KEYWORDS}
    for phrase in MULTI_TOKEN_TECH_PHRASES:
        canonical_map[phrase.lower()] = phrase

    canonical_map.update({
        "zk": "ZK", "rag": "RAG", "nlp": "NLP", "ocr": "OCR", "llm": "LLM", "llms": "LLM",
        "mpt": "MPT", "evm": "EVM", "ipfs": "IPFS", "gpt": "GPT", "gpt-4o": "GPT-4o",
        "json": "JSON", "bb.js": "bb.js", "ethers.js": "Ethers.js", "node.js": "Node.js",
        "vue.js": "Vue.js", "next.js": "Next.js", "vite": "Vite", "noir": "Noir",
        "viem": "Viem", "foundry": "Foundry", "hardhat": "Hardhat", "ollama": "Ollama",
        "ultrahonk": "UltraHonk", "clinicalbert": "ClinicalBERT", "networkx": "NetworkX",
        "cytoscape": "Cytoscape", "leiden": "Leiden", "scikit-learn": "Scikit-Learn",
        "tailwindcss": "TailwindCSS", "c#": "C#", "c++": "C++", ".net": ".NET",
        "c": "C", "r": "R", "sql": "SQL", "html": "HTML", "css": "CSS", "bash": "Bash",
        "api": "API", "sdk": "SDK", "rest": "REST", "grpc": "gRPC", "k8s": "K8s",
        "ci/cd": "CI/CD", "aws": "AWS", "gcp": "GCP", "ui": "UI", "ux": "UX",
    })

    discovered: set[str] = set()

    # a) Section-based parsing
    lines = text_content.splitlines()
    in_skills_section = False

    section_start_re = re.compile(
        r"^\s*(?:[#*•\-\s]*)(?:Technical\s+Skills|Skills|Technologies|Tech\s+Stack|Core\s+Competencies|Technical\s+Competencies)\s*[:\-]?\s*(.*)$",
        re.IGNORECASE,
    )
    section_end_re = re.compile(
        r"^\s*(?:[#*•\-\s]*)(?:Experience|Work\s+Experience|Projects|Education|Certifications|Professional\s+Summary|Summary|Publications|Awards)\s*[:\-]?\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m_start = section_start_re.match(stripped)
        if m_start:
            in_skills_section = True
            after = m_start.group(1).strip()
            if after:
                cleaned_line = after.replace("(", ",").replace(")", ",")
                tokens = re.split(r"[,;•\n\r|]+", cleaned_line)
                for tok in tokens:
                    c = _clean_and_canonicalize_token(tok, canonical_map)
                    if c:
                        discovered.add(c)
            continue
        elif in_skills_section and section_end_re.match(stripped) and not stripped.lower().startswith("languages:"):
            in_skills_section = False
            continue

        if in_skills_section:
            cleaned_line = re.sub(r"^[•*—\-\s]*[A-Za-z0-9\s&/\\+,\-]+?:\s*", "", stripped)
            cleaned_line = cleaned_line.replace("(", ",").replace(")", ",")
            tokens = re.split(r"[,;•\n\r|]+", cleaned_line)
            for tok in tokens:
                c = _clean_and_canonicalize_token(tok, canonical_map)
                if c:
                    discovered.add(c)

    # b) Pattern-based entity discovery across the entire text
    # 1. PascalCase / CamelCase frameworks & libraries
    for m in re.finditer(r"\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*\b", text_content):
        c = _clean_and_canonicalize_token(m.group(0), canonical_map)
        if c:
            discovered.add(c)

    # 2. Package / runtime notations
    for m in re.finditer(r"\b[A-Za-z0-9]+\.(?:js|ts|py|rs|io)\b", text_content, re.IGNORECASE):
        c = _clean_and_canonicalize_token(m.group(0), canonical_map)
        if c:
            discovered.add(c)

    # 3. Acronyms & technical terms (including plurals like LLMs, APIs, SDKs)
    for m in re.finditer(r"\b([A-Z]{2,6})(?:s)?\b", text_content):
        c = _clean_and_canonicalize_token(m.group(1), canonical_map)
        if c:
            discovered.add(c)
        c_full = _clean_and_canonicalize_token(m.group(0), canonical_map)
        if c_full:
            discovered.add(c_full)

    # 4. Multi-token tech phrases & key tech keywords
    all_phrases = sorted(list(set(MULTI_TOKEN_TECH_PHRASES + CURATED_TECH_KEYWORDS)), key=lambda s: len(s), reverse=True)
    text_lower = text_content.lower()
    for phrase in all_phrases:
        if phrase in (".NET", "C#", "C++", "C", "R"):
            pattern = r"(?<![a-zA-Z0-9_])" + re.escape(phrase) + r"(?![a-zA-Z0-9_])"
            if re.search(pattern, text_content, re.IGNORECASE):
                c = _clean_and_canonicalize_token(phrase, canonical_map)
                if c:
                    discovered.add(c)
        else:
            pattern = r"(?<![a-zA-Z0-9_])" + re.escape(phrase.lower()) + r"(?![a-zA-Z0-9_])"
            if re.search(pattern, text_lower):
                c = _clean_and_canonicalize_token(phrase, canonical_map)
                if c:
                    discovered.add(c)

    # e) Return sorted, deduplicated list of dynamic skills
    seen: dict[str, str] = {}
    for skill in discovered:
        low = skill.lower()
        if low not in seen:
            seen[low] = skill

    return sorted(list(seen.values()), key=lambda s: s.lower())


def extract_cv_keywords(cv_path: Optional[str] = None) -> list[str]:
    """Read a CV/resume file (text, pdf, docx) and extract dynamic technology keywords.

    Args:
        cv_path: Optional path to the resume/CV file. If not provided, resolved automatically.

    Returns:
        list[str]: Sorted, deduplicated list of detected technology keywords.
    """
    path = resolve_cv_path(cv_path)
    if path is None:
        logger.warning("CV file path '%s' does not exist.", cv_path or os.getenv("DEFAULT_CV_PATH"))
        return []

    logger.info("Extracting keywords from CV: %s", path)
    text_content = _extract_text_from_file(path)
    if not text_content:
        logger.warning("No readable text could be extracted from '%s'.", path)
        return []

    result = extract_dynamic_cv_skills(text_content)
    logger.info("Extracted %d dynamic skills from CV '%s': %s", len(result), path.name, result)
    return result


def _detect_cv_seniority(text: str) -> Optional[str]:
    """Detect candidate seniority level from CV text."""
    if not text:
        return None

    header_chunk = text[:1500]

    # Explicit summary/title markers
    if re.search(r"\b(student)\b", header_chunk, re.IGNORECASE):
        if not re.search(r"\b(senior\s+(?:developer|engineer|architect)|lead\s+developer|tech\s+lead)\b", header_chunk, re.IGNORECASE):
            return "Student"
    if re.search(r"\b(intern|internship)\b", header_chunk, re.IGNORECASE):
        return "Intern"
    if re.search(r"\b(junior|entry[\s-]level|graduate)\b", header_chunk, re.IGNORECASE):
        return "Junior"
    if re.search(r"\b(principal|distinguished)\b", header_chunk, re.IGNORECASE):
        return "Principal"
    if re.search(r"\b(tech\s+lead|team\s+lead|lead\s+developer|lead\s+engineer|head\s+of|director|vp)\b", header_chunk, re.IGNORECASE):
        return "Lead"
    if re.search(r"\b(senior|staff|architect)\b|\bsr\.?\b", header_chunk, re.IGNORECASE):
        return "Senior"
    if re.search(r"\b(mid[\s-]level|intermediate)\b", header_chunk, re.IGNORECASE):
        return "Mid"

    # Check years of experience in full text
    m_years = re.search(r"\b(\d+)\+?\s*years(?:\s+of)?\s*(?:experience|working)?\b", text, re.IGNORECASE)
    if m_years:
        years = int(m_years.group(1))
        if years >= 10:
            return "Principal" if re.search(r"\b(architect|principal|staff)\b", text, re.IGNORECASE) else "Senior"
        elif years >= 5:
            return "Senior"
        elif years >= 2:
            return "Mid"
        elif years <= 1:
            return "Junior"

    # Fallback to full text keyword scan
    if re.search(r"\b(principal|distinguished)\b", text, re.IGNORECASE):
        return "Principal"
    if re.search(r"\b(tech\s+lead|team\s+lead|lead\s+developer|lead\s+engineer)\b", text, re.IGNORECASE):
        return "Lead"
    if re.search(r"\b(senior|software\s+architect|solutions\s+architect|staff\s+engineer)\b", text, re.IGNORECASE):
        return "Senior"
    if re.search(r"\b(junior|entry[\s-]level)\b", text, re.IGNORECASE):
        return "Junior"
    if re.search(r"\b(student)\b", text, re.IGNORECASE):
        return "Student"
    if re.search(r"\b(intern|internship)\b", text, re.IGNORECASE):
        return "Intern"
    if re.search(r"\b(mid[\s-]level|intermediate)\b", text, re.IGNORECASE):
        return "Mid"

    return None


def _compute_top_skills(skills: list[str], text: str) -> list[str]:
    """Compute top 8-12 primary skills ensuring strong representation of core languages/frameworks and specialized competencies."""
    if not skills:
        return []
    if len(skills) <= 8:
        return list(skills)

    text_lower = text.lower()
    first_quarter = text_lower[: max(500, len(text_lower) // 4)]

    skills_section = ""
    m_sec = re.search(
        r"(?:technical\s+skills|skills|technologies|tech\s+stack|core\s+competencies)[\s:\-]+(.*?)(?:\n\s*(?:experience|work|projects|education)|\Z)",
        text_lower,
        re.DOTALL | re.IGNORECASE,
    )
    if m_sec:
        skills_section = m_sec.group(1)

    scored: list[tuple[float, str]] = []
    for skill in skills:
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9_])"
        occurrences = len(re.findall(pattern, text_lower))
        score = occurrences * 2.0

        if pattern and re.search(pattern, first_quarter):
            score += 4.0
        if skills_section and pattern and re.search(pattern, skills_section):
            score += 5.0

        low = skill.lower()
        if low in CORE_LANGUAGES_FRAMEWORKS:
            score += 3.0
        if low in SPECIALIZED_COMPETENCIES:
            score += 3.0

        scored.append((score, skill))

    # Sort descending by score, then ascending by name
    scored.sort(key=lambda item: (-item[0], item[1].lower()))

    target_count = min(len(skills), max(8, min(12, len(skills))))

    selected: list[str] = []
    seen: set[str] = set()

    # Pass 1: Ensure representation of top core languages & frameworks (up to 3)
    for _, skill in scored:
        if len(selected) >= 3:
            break
        if skill.lower() in CORE_LANGUAGES_FRAMEWORKS and skill not in seen:
            selected.append(skill)
            seen.add(skill)

    # Pass 2: Ensure representation of top specialized competencies (up to 3)
    for _, skill in scored:
        if len(selected) >= 6:
            break
        if skill.lower() in SPECIALIZED_COMPETENCIES and skill not in seen:
            selected.append(skill)
            seen.add(skill)

    # Pass 3: Fill remaining slots with highest scoring skills up to target_count
    for _, skill in scored:
        if len(selected) >= target_count:
            break
        if skill not in seen:
            selected.append(skill)
            seen.add(skill)

    return selected


def _compute_primary_stack(skills: list[str], top_skills: list[str], text: str = "") -> list[str]:
    """Compute candidate's primary tech stack (6-10 core skills) balancing core languages/frameworks and domain abilities."""
    if not top_skills and not skills:
        return []
    source = top_skills if top_skills else skills
    if len(source) <= 6:
        return list(source)

    target_count = min(len(source), max(6, min(10, len(source))))

    selected: list[str] = []
    seen: set[str] = set()

    # 1. Take top core languages/frameworks
    for skill in source:
        if len(selected) >= 3:
            break
        if skill.lower() in CORE_LANGUAGES_FRAMEWORKS and skill not in seen:
            selected.append(skill)
            seen.add(skill)

    # 2. Take top specialized competencies
    for skill in source:
        if len(selected) >= 6:
            break
        if skill.lower() in SPECIALIZED_COMPETENCIES and skill not in seen:
            selected.append(skill)
            seen.add(skill)

    # 3. Fill up to target_count (6-10) preserving order from source
    for skill in source:
        if len(selected) >= target_count:
            break
        if skill not in seen:
            selected.append(skill)
            seen.add(skill)

    return selected


def _derive_target_roles(skills: list[str], text: str, seniority: Optional[str] = None) -> list[str]:
    """Derive suggested job titles from extracted skills and CV text."""
    skills_lower = {s.lower() for s in skills}
    text_lower = text.lower()
    roles: list[str] = []

    # AI / ML / Data
    ai_indicators = {
        "graphrag", "langgraph", "langchain", "llamaindex", "pytorch", "tensorflow",
        "hugging face", "llm", "rag", "nlp", "machine learning", "ollama", "vllm",
        "pinecone", "qdrant", "chromadb", "chroma", "vector db", "clinicalbert",
        "pandas", "scikit-learn", "numpy", "transformers", "fine-tuning", "embeddings",
        "agentic", "semantic kernel",
    }
    matched_ai = skills_lower & ai_indicators
    if len(matched_ai) >= 2 or "ai " in text_lower or "artificial intelligence" in text_lower or "machine learning" in text_lower or "llm" in text_lower:
        roles.append("AI Engineer")
        if any(s in skills_lower for s in ["pytorch", "tensorflow", "scikit-learn", "machine learning", "clinicalbert"]):
            roles.append("Machine Learning Engineer")

    # Web3 / Blockchain
    web3_indicators = {
        "solidity", "noir", "web3", "smart contracts", "smart contract development",
        "foundry", "hardhat", "evm", "ethers.js", "viem", "zk", "ultrahonk", "bb.js", "ipfs",
        "keccak256", "merkle patricia trie",
    }
    matched_web3 = skills_lower & web3_indicators
    if len(matched_web3) >= 1 or "web3" in text_lower or "smart contract" in text_lower:
        roles.append("Web3 Developer")
        roles.append("Smart Contract Engineer")
        roles.append("Blockchain Developer")

    # Backend / Python / Languages
    backend_indicators = {
        "fastapi", "django", "flask", "sqlalchemy", "asyncio", "postgresql", "postgres",
        "redis", "rest", "grpc", "fastmcp", "pydantic",
    }
    if "python" in skills_lower and (skills_lower & backend_indicators or "backend" in text_lower or "developer" in text_lower or "engineer" in text_lower):
        roles.append("Python Developer")
        roles.append("Backend Engineer")
    elif "python" in skills_lower:
        roles.append("Python Developer")

    if "go" in skills_lower or "golang" in skills_lower:
        roles.append("Go Developer")
        if "Backend Engineer" not in roles:
            roles.append("Backend Engineer")

    if "rust" in skills_lower:
        roles.append("Rust Developer")
        if "Systems Engineer" not in roles:
            roles.append("Systems Engineer")

    if "c++" in skills_lower or "c" in skills_lower or "operating systems" in skills_lower:
        if "Systems Engineer" not in roles:
            roles.append("Systems Engineer")

    # Frontend / Full Stack
    frontend_indicators = {
        "react", "next.js", "vue", "vue.js", "nuxt", "angular", "svelte",
        "typescript", "javascript", "tailwindcss", "vite", "redux", "html", "css", "zustand",
    }
    matched_frontend = skills_lower & frontend_indicators
    has_backend_skill = any(s in skills_lower for s in [
        "python", "node.js", "node", "go", "golang", "java", "c#", "rust", "fastapi",
        "django", "postgresql", "postgres", "sql", "redis", "mongodb",
    ])

    if len(matched_frontend) >= 2 or "frontend" in text_lower:
        if has_backend_skill and "Full Stack Engineer" not in roles:
            roles.append("Full Stack Engineer")
        if "Frontend Engineer" not in roles:
            roles.append("Frontend Engineer")
    elif has_backend_skill and len(matched_frontend) >= 1:
        if "Full Stack Engineer" not in roles:
            roles.append("Full Stack Engineer")

    # DevOps / Cloud
    devops_indicators = {
        "kubernetes", "k8s", "docker", "terraform", "ansible", "helm", "aws",
        "amazon web services", "gcp", "google cloud", "azure", "ci/cd", "github actions",
        "gitlab ci", "jenkins", "prometheus", "grafana",
    }
    matched_devops = skills_lower & devops_indicators
    if len(matched_devops) >= 2 or "devops" in text_lower or "cloud" in text_lower:
        roles.append("DevOps Engineer")
        roles.append("Cloud Engineer")

    # Java / .NET
    if "java" in skills_lower or "spring" in skills_lower or "spring boot" in skills_lower:
        roles.append("Java Developer")
        if "Backend Engineer" not in roles:
            roles.append("Backend Engineer")
    if "c#" in skills_lower or ".net" in skills_lower:
        roles.append(".NET Developer")
        if "Backend Engineer" not in roles:
            roles.append("Backend Engineer")

    # Default fallback
    if not roles and skills:
        roles.append("Software Engineer")

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped_roles: list[str] = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            deduped_roles.append(r)

    return deduped_roles


def _derive_search_queries(top_skills: list[str], target_roles: list[str]) -> list[str]:
    """Derive concise 1-2 word tech queries for job search platforms."""
    queries: list[str] = []
    seen: set[str] = set()

    for role in target_roles[:2]:
        if role.lower() not in seen:
            seen.add(role.lower())
            queries.append(role)

    for skill in top_skills[:4]:
        if skill.lower() not in seen:
            seen.add(skill.lower())
            queries.append(skill)

    for role in target_roles[2:4]:
        if len(queries) >= 6:
            break
        if role.lower() not in seen:
            seen.add(role.lower())
            queries.append(role)

    for skill in top_skills[4:]:
        if len(queries) >= 6:
            break
        if skill.lower() not in seen:
            seen.add(skill.lower())
            queries.append(skill)

    return queries


def _derive_suggested_exclusions(seniority: Optional[str]) -> list[str]:
    """Derive adaptive exclusions based on candidate seniority level."""
    if not seniority:
        return []
    sen_upper = seniority.upper()
    if sen_upper in ("STUDENT", "INTERN", "JUNIOR"):
        return ["Senior", "Lead", "Principal", "Staff", "Director", "VP", "Head", "7+ years", "10+ years"]
    elif sen_upper == "MID":
        return ["Principal", "Staff", "Director", "VP", "Head", "10+ years"]
    elif sen_upper in ("SENIOR", "LEAD", "PRINCIPAL"):
        return ["Student", "Intern", "Junior", "Entry Level", "Graduate"]
    return []


def extract_candidate_profile(cv_source: Optional[Union[str, Path]] = None) -> CandidateProfile:
    """Extract a structured candidate profile from a CV file path, Path object, or raw text.

    Args:
        cv_source: Optional path to CV file or raw string content. If None, default CV path is resolved.

    Returns:
        CandidateProfile: Populated candidate profile data model.
    """
    text_content = _extract_text_from_source(cv_source)
    if not text_content or not text_content.strip():
        logger.warning("No readable text content found for candidate profile extraction.")
        return CandidateProfile()

    skills = extract_dynamic_cv_skills(text_content)
    seniority_level = _detect_cv_seniority(text_content)
    top_skills = _compute_top_skills(skills, text_content)
    primary_stack = _compute_primary_stack(skills, top_skills, text_content)
    target_roles = _derive_target_roles(skills, text_content, seniority_level)
    search_queries = _derive_search_queries(top_skills, target_roles)
    suggested_exclusions = _derive_suggested_exclusions(seniority_level)

    logger.info(
        "Candidate profile extracted: %d skills, %d top skills, %d primary stack, seniority: %s, %d target roles",
        len(skills),
        len(top_skills),
        len(primary_stack),
        seniority_level,
        len(target_roles),
    )

    return CandidateProfile(
        skills=skills,
        top_skills=top_skills,
        primary_stack=primary_stack,
        seniority_level=seniority_level,
        target_roles=target_roles,
        search_queries=search_queries,
        suggested_exclusions=suggested_exclusions,
    )


# Precompile curated tech regexes for sub-millisecond scoring
_CURATED_TECH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (tech, re.compile(r"(?<![a-zA-Z0-9_])" + re.escape(tech.lower()) + r"(?![a-zA-Z0-9_])"))
    for tech in CURATED_TECH_KEYWORDS
]


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
    for tech, pattern in _CURATED_TECH_PATTERNS:
        if pattern.search(text_lower):
            found.append(tech)
    return found


SENIORITY_EXCLUDE_TERMS = {
    "senior", "sr", "sr.", "lead", "principal", "staff", "architect", "director", "vp", "head"
}


def detect_seniority_level(title: str, text: str = "") -> Optional[str]:
    """Detect seniority level from job title and description.

    Args:
        title: Job title string.
        text: Optional job description or body text.

    Returns:
        Optional[str]: Detected seniority level ('Student', 'Intern', 'Junior', 'Mid', 'Senior', 'Lead') or None.
    """
    title_clean = (title or "").lower()

    # 1. Check title first (highest precision)
    if re.search(r"\b(student)\b", title_clean, re.IGNORECASE):
        return "Student"
    if re.search(r"\b(intern|internship)\b", title_clean, re.IGNORECASE):
        return "Intern"
    if re.search(r"\b(junior|entry|graduate)\b", title_clean, re.IGNORECASE):
        return "Junior"
    if re.search(r"\b(lead|director|vp)\b|\bhead(\s+of)?\b", title_clean, re.IGNORECASE):
        return "Lead"
    if re.search(r"\b(senior|principal|staff|architect)\b|\bsr\b|\bsr\.", title_clean, re.IGNORECASE):
        return "Senior"
    if re.search(r"\b(mid|intermediate)\b", title_clean, re.IGNORECASE):
        return "Mid"

    # 2. Check text fallback if title has no clear indication
    text_clean = (text or "").lower()
    if text_clean:
        if re.search(r"\b(tech\s+lead|team\s+lead|lead\s+developer|lead\s+engineer|head\s+of|director|vp)\b", text_clean, re.IGNORECASE):
            return "Lead"
        if re.search(r"\b(senior\s+developer|senior\s+engineer|senior\s+fullstack|senior\s+backend|senior\s+frontend|principal\s+engineer|staff\s+engineer|software\s+architect)\b", text_clean, re.IGNORECASE):
            return "Senior"
        if re.search(r"\b(junior\s+developer|junior\s+engineer|junior\s+position|entry[\s-]level|graduate\s+program)\b", text_clean, re.IGNORECASE):
            return "Junior"
        if re.search(r"\b(student\s+position|student\s+developer|internship|intern\s+position)\b", text_clean, re.IGNORECASE):
            return "Intern" if "intern" in text_clean else "Student"
        if re.search(r"\b(mid[\s-]level|intermediate)\b", text_clean, re.IGNORECASE):
            return "Mid"
        if re.search(r"\b(junior|entry|graduate)\b", text_clean, re.IGNORECASE):
            return "Junior"
        if re.search(r"\b(intern|internship)\b", text_clean, re.IGNORECASE):
            return "Intern"
        if re.search(r"\b(student)\b", text_clean, re.IGNORECASE):
            return "Student"
        if re.search(r"\b(senior|lead|principal|staff|architect)\b", text_clean, re.IGNORECASE):
            return "Senior"

    return None


def generate_description_summary(text: str, max_chars: int = 250) -> Optional[str]:
    """Generate a clean 2-3 sentence summary of the job description (up to ~250 chars).

    Args:
        text: Full job description.
        max_chars: Target max character limit.

    Returns:
        Optional[str]: Concise description summary.
    """
    if not text:
        return None
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:- ") + "..."

    selected: list[str] = []
    current_len = 0
    for s in sentences[:3]:
        if selected and (current_len + len(s) + 1 > max_chars):
            break
        selected.append(s)
        current_len += len(s) + 1

    summary = " ".join(selected)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:- ") + "..."
    return summary


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


def _match_terms_in_job(terms: set[str], text_lower: str, tech_tokens: set[str]) -> set[str]:
    """Check which search terms match in job tech stack or text using word boundaries."""
    if not terms:
        return set()
    matched: set[str] = set()
    for t in terms:
        if t in tech_tokens:
            matched.add(t)
        else:
            pattern = r"(?<![a-zA-Z0-9_])" + re.escape(t) + r"(?![a-zA-Z0-9_])"
            if re.search(pattern, text_lower):
                matched.add(t)
    return matched


def calculate_match_score(
    job: Job,
    prefs: JobPreferences,
    profile: Optional[CandidateProfile] = None,
    display_map: Optional[dict[str, str]] = None,
) -> float:
    """Calculate dynamic requirement-based fit score (0.0 - 100.0) and populate explainability fields on Job.

    Scoring formula components:
    1. Job Requirement Coverage (C_req): Ratio of candidate skills matching job required skills.
    2. Primary Skill Affinity (A_top): Bonus for matching candidate's top primary competencies.
    3. Target Role Match (B_role): Bonus for matching candidate's target job titles or explicit keywords in job title.
    4. User Preference Weighting: Explicit prefs.tech_stack is integrated and weighted alongside CV requirements.

    Args:
        job: Job instance to score and update with explainability fields.
        prefs: User JobPreferences configuration.
        profile: Optional pre-extracted CandidateProfile. If None, resolved dynamically.
        display_map: Optional dictionary mapping lowercase terms to canonical display strings.

    Returns:
        float: Calculated match score between 0.0 and 100.0 rounded to 1 decimal place.
    """
    if profile is None:
        if prefs.cv_path:
            profile = extract_candidate_profile(prefs.cv_path)
        else:
            profile = CandidateProfile(
                skills=prefs.tech_stack,
                top_skills=prefs.tech_stack,
                primary_stack=prefs.tech_stack,
                search_queries=prefs.tech_stack,
            )

    if display_map is None:
        display_map = {k.lower(): k for k in CURATED_TECH_KEYWORDS}
        for phrase in MULTI_TOKEN_TECH_PHRASES:
            display_map[phrase.lower()] = phrase
        for s in profile.skills:
            display_map[s.lower()] = s
        for s in profile.primary_stack:
            display_map[s.lower()] = s
        for s in profile.top_skills:
            display_map[s.lower()] = s
        for t in prefs.tech_stack:
            if t.strip():
                display_map[t.strip().lower()] = t.strip()
        for k in prefs.keywords:
            if k.strip():
                display_map[k.strip().lower()] = k.strip()

    job_full_text = f"{job.title} {job.company} {job.location} {job.description} {' '.join(job.tech_stack)}".lower()

    # Job's identified tech stack/skills
    job_tech_tokens = {t.strip().lower() for t in job.tech_stack if t.strip()}
    extracted_job_skills = {s.lower() for s in _extract_text_tech_keywords(f"{job.title} {job.description} {' '.join(job.tech_stack)}")}
    job_skills = job_tech_tokens | extracted_job_skills

    # Candidate profile skills and explicit preferences
    profile_skills = {s.strip().lower() for s in profile.skills if s.strip()}
    primary_skills = [s.strip().lower() for s in (profile.primary_stack or profile.top_skills or profile.skills) if s.strip()]
    top_skills = [s.strip().lower() for s in (profile.top_skills or profile.primary_stack or profile.skills) if s.strip()]
    target_roles = [r.strip().lower() for r in profile.target_roles if r.strip()]
    explicit_tech = {t.strip().lower() for t in prefs.tech_stack if t.strip()}
    explicit_keywords = {k.strip().lower() for k in prefs.keywords if k.strip()}

    if not top_skills and explicit_tech:
        top_skills = list(explicit_tech)
    if not primary_skills and explicit_tech:
        primary_skills = list(explicit_tech)

    all_candidate_skills = profile_skills | set(primary_skills) | set(top_skills) | explicit_tech
    all_desired_tokens = all_candidate_skills | explicit_keywords

    # Matched skills
    matched_job_skills = all_candidate_skills & job_skills
    matched_in_text = _match_terms_in_job(all_candidate_skills | explicit_keywords, job_full_text, job_tech_tokens)
    all_matched_tokens = matched_job_skills | matched_in_text

    # Explainability: matched & missing skills
    job.matched_skills = sorted([display_map.get(s, s.title()) for s in all_matched_tokens], key=lambda s: s.lower())

    has_cv_profile = bool(prefs.cv_path or (profile and profile.skills and profile.skills != prefs.tech_stack))
    if has_cv_profile:
        if job_skills:
            missing_from_job = job_skills - all_candidate_skills
            missing_desired = explicit_tech - all_matched_tokens
            missing_tokens = missing_from_job | missing_desired
        else:
            missing_tokens = all_desired_tokens - all_matched_tokens
    else:
        missing_tokens = all_desired_tokens - all_matched_tokens
    job.missing_skills = sorted([display_map.get(s, s.title()) for s in missing_tokens], key=lambda s: s.lower())

    if not job.seniority_level:
        job.seniority_level = detect_seniority_level(job.title, job.description)

    if not job.description_summary and job.description:
        job.description_summary = generate_description_summary(job.description)

    # 1. Job Requirement Coverage (C_req)
    if job_skills:
        c_req = len(matched_job_skills) / len(job_skills)
    elif all_matched_tokens:
        c_req = min(1.0, len(all_matched_tokens) / 3.0)
    else:
        c_req = 0.0

    # 2. Candidate Primary Skill Affinity (A_top)
    affinity_targets = primary_skills if primary_skills else top_skills
    top_matches = [
        s for s in affinity_targets
        if s in all_matched_tokens or s in job_skills or re.search(r"(?<![a-zA-Z0-9_])" + re.escape(s) + r"(?![a-zA-Z0-9_])", job_full_text)
    ]
    if len(top_matches) >= 3:
        a_top = 35.0
    elif len(top_matches) == 2:
        a_top = 22.0
    elif len(top_matches) == 1:
        a_top = 10.0
    else:
        a_top = 0.0

    # 3. Role Title Match Bonus (B_role)
    b_role = 0.0
    matched_target_role = None
    title_lower = job.title.lower()
    
    is_admin_title = any(x in title_lower for x in ["מנתח מערכות", "מנהל פרויקטים", "data annotator", "qa", "project manager", "system analyst", "scrum master", "product manager", "help desk", "support"])

    if not is_admin_title:
        for role in target_roles:
            pattern = r"(?<![a-zA-Z0-9_])" + re.escape(role) + r"(?![a-zA-Z0-9_])"
            if re.search(pattern, title_lower):
                b_role = 15.0
                matched_target_role = role
                break

        if b_role == 0.0 and explicit_keywords:
            for kw in explicit_keywords:
                pattern = r"(?<![a-zA-Z0-9_])" + re.escape(kw) + r"(?![a-zA-Z0-9_])"
                if re.search(pattern, title_lower):
                    b_role = 12.0
                    break

    # 4. Explicit Tech Stack Alignment
    if explicit_tech:
        tech_matched = explicit_tech & all_matched_tokens
        tech_coverage = len(tech_matched) / len(explicit_tech)
    else:
        tech_coverage = 1.0

    # 5. Base Score calculation
    if not all_desired_tokens:
        score = 100.0
    elif not all_matched_tokens and b_role == 0.0:
        score = 0.0
    elif not all_matched_tokens and b_role > 0.0:
        score = min(25.0, b_role)
    else:
        if has_cv_profile:
            kw_matched = _match_terms_in_job(explicit_keywords, job_full_text, job_tech_tokens)
            kw_cov = len(kw_matched) / len(explicit_keywords) if explicit_keywords else 1.0
            
            num_matched = len(all_matched_tokens)
            if num_matched >= 4:
                volume_score = 30.0
            elif num_matched == 3:
                volume_score = 22.0
            elif num_matched == 2:
                volume_score = 15.0
            elif num_matched == 1:
                volume_score = 8.0
            else:
                volume_score = 0.0

            if explicit_tech and explicit_keywords:
                coverage_score = ((c_req * 0.5) + (tech_coverage * 0.3) + (kw_cov * 0.2)) * 20.0
            elif explicit_tech:
                coverage_score = ((c_req * 0.6) + (tech_coverage * 0.4)) * 20.0
            else:
                coverage_score = c_req * 20.0

            raw_score = volume_score + coverage_score + a_top + b_role
            if (c_req >= 0.85 or (tech_coverage == 1.0 and len(all_matched_tokens) >= 3 and c_req >= 0.65)) and len(all_matched_tokens) >= 3:
                raw_score = max(raw_score, 88.0)
            elif c_req >= 0.65 and len(all_matched_tokens) >= 3:
                raw_score = max(raw_score, 75.0)
        else:
            # Pure explicit search (no CV)
            kw_matched = _match_terms_in_job(explicit_keywords, job_full_text, job_tech_tokens)
            kw_coverage = len(kw_matched) / len(explicit_keywords) if explicit_keywords else 1.0
            tech_matched = explicit_tech & all_matched_tokens
            if (not explicit_tech or tech_coverage == 1.0) and (not explicit_keywords or kw_coverage == 1.0):
                raw_score = 100.0
            else:
                if explicit_keywords and explicit_tech:
                    base_score = (tech_coverage * 55.0) + (kw_coverage * 20.0)
                elif explicit_tech:
                    base_score = tech_coverage * 70.0
                else:
                    base_score = kw_coverage * 70.0
                volume_bonus = min(10.0, len(tech_matched) * 2.5) if explicit_tech else 0.0
                raw_score = base_score + volume_bonus + (5.0 if b_role > 0 else 0.0)
                if tech_coverage >= 0.75 and len(tech_matched) >= 3:
                    raw_score = max(raw_score, 75.0)

        # Title match bonus for exact title relevance
        if b_role == 0.0 and (explicit_tech | profile_skills | explicit_keywords):
            for t in explicit_tech | profile_skills | explicit_keywords:
                pattern = r"(?<![a-zA-Z0-9_])" + re.escape(t) + r"(?![a-zA-Z0-9_])"
                if re.search(pattern, title_lower):
                    if not is_admin_title and has_cv_profile:
                        raw_score += 5.0
                    break
                    
        is_dev_role = any(x in title_lower for x in ["developer", "engineer", "software", "ai", "backend", "frontend", "full stack", "data engineer", "mlops"]) or matched_target_role
        has_primary_match = any(s in all_matched_tokens for s in primary_skills)
        if is_dev_role and has_primary_match:
            raw_score = max(raw_score, 75.0)
        
        if is_admin_title:
            raw_score = min(raw_score, 65.0)

        if job.work_mode == WorkMode.ONSITE and "eilat" in job.location.lower():
            if not prefs.location or "israel" in prefs.location.lower():
                raw_score = min(raw_score, 60.0)

        score = round(max(0.0, min(100.0, raw_score)), 1)

    # 6. Generate explainability reasons
    reasons: list[str] = []
    if all_matched_tokens:
        if has_cv_profile and matched_job_skills:
            cv_names = sorted([display_map.get(s, s.title()) for s in matched_job_skills], key=lambda s: s.lower())
            cv_fit_pct = round((len(matched_job_skills) / len(job_skills)) * 100.0) if job_skills else 100.0
            reasons.append(f"CV matched {len(matched_job_skills)} core skills: {', '.join(cv_names)} ({int(cv_fit_pct)}% tech requirements match)")
        elif has_cv_profile and all_matched_tokens:
            cv_names = sorted([display_map.get(s, s.title()) for s in all_matched_tokens], key=lambda s: s.lower())
            reasons.append(f"CV matched {len(all_matched_tokens)} core skills: {', '.join(cv_names)}")
        else:
            cv_names = sorted([display_map.get(s, s.title()) for s in all_matched_tokens], key=lambda s: s.lower())
            reasons.append(f"CV matched skills: {', '.join(cv_names)}")

    if explicit_tech:
        matched_tech = explicit_tech & all_matched_tokens
        if matched_tech:
            tech_names = sorted([display_map.get(s, s.title()) for s in matched_tech], key=lambda s: s.lower())
            reasons.append(f"Target stack matched: {', '.join(tech_names)}")

    if explicit_keywords:
        matched_kw = _match_terms_in_job(explicit_keywords, job_full_text, job_tech_tokens)
        if matched_kw:
            kw_names = sorted([display_map.get(s, s.title()) for s in matched_kw], key=lambda s: s.lower())
            reasons.append(f"Keywords matched: {', '.join(kw_names)}")

    if matched_target_role:
        reasons.append(f"Target role matched: {matched_target_role.title()}")

    if prefs.work_mode:
        pref_wm = prefs.work_mode.value if isinstance(prefs.work_mode, WorkMode) else str(prefs.work_mode)
        reasons.append(f"{pref_wm.capitalize()} work mode aligned with preference")
    elif job.work_mode == WorkMode.REMOTE:
        reasons.append("Remote work mode aligned with preference")

    if job.seniority_level:
        reasons.append(f"{job.seniority_level} level position")

    if prefs.location and (prefs.location.lower() in job.location.lower() or prefs.location.lower() in job_full_text):
        reasons.append(f"Location aligned with preference ({prefs.location})")

    if prefs.min_salary and job.salary_range:
        reasons.append(f"Salary range meets requirement ({job.salary_range})")

    job.match_reasons = reasons
    job.match_score = score
    return score


def filter_jobs(
    jobs: list[Job],
    prefs: JobPreferences,
    profile: Optional[CandidateProfile] = None,
) -> list[Job]:
    """Filter and score job listings according to user preferences and candidate profile.

    Args:
        jobs: List of Job instances.
        prefs: JobPreferences configuration.
        profile: Optional CandidateProfile instance. If omitted, resolved dynamically from prefs.cv_path or preferences.

    Returns:
        list[Job]: Filtered and ranked list of Job instances sorted by match_score descending.
    """
    has_explicit_profile = profile is not None
    if profile is None:
        if prefs.cv_path:
            profile = extract_candidate_profile(prefs.cv_path)
            has_explicit_profile = True
        else:
            profile = CandidateProfile(
                skills=prefs.tech_stack,
                top_skills=prefs.tech_stack,
                primary_stack=prefs.tech_stack,
                search_queries=prefs.tech_stack,
            )

    filtered: list[Job] = []

    # Prepare display mapping
    display_map: dict[str, str] = {k.lower(): k for k in CURATED_TECH_KEYWORDS}
    for phrase in MULTI_TOKEN_TECH_PHRASES:
        display_map[phrase.lower()] = phrase
    for s in profile.skills:
        display_map[s.lower()] = s
    for s in profile.primary_stack:
        display_map[s.lower()] = s
    for s in profile.top_skills:
        display_map[s.lower()] = s
    for t in prefs.tech_stack:
        if t.strip():
            display_map[t.strip().lower()] = t.strip()
    for k in prefs.keywords:
        if k.strip():
            display_map[k.strip().lower()] = k.strip()

    # Collect exclusion keywords
    exclusions = [k.strip() for k in prefs.exclude_keywords if k.strip()]
    if has_explicit_profile and profile and profile.suggested_exclusions:
        for exc in profile.suggested_exclusions:
            if exc not in exclusions:
                exclusions.append(exc)

    for job in jobs:
        job_full_text = f"{job.title} {job.company} {job.location} {job.description} {' '.join(job.tech_stack)}".lower()

        # 1. Exclude keywords check
        is_excluded = False
        for exc in exclusions:
            exc_lower = exc.lower()
            if exc_lower in SENIORITY_EXCLUDE_TERMS or re.search(
                r"^(senior|sr\.?|principal|lead|staff|architect|director|vp|head|student|intern|junior|entry[\s-]level|graduate)$",
                exc_lower,
            ):
                # Seniority exclusion: check job title with word boundaries
                if exc_lower in ("senior", "sr", "sr."):
                    term_pattern = r"\b(senior|sr)\b|\bsr\."
                else:
                    term_pattern = r"\b" + re.escape(exc_lower) + r"\b"
                if re.search(term_pattern, job.title, re.IGNORECASE):
                    is_excluded = True
                    break
            elif re.search(r"(\d+)\+?\s*years", exc_lower):
                m_req = re.search(r"(\d+)\+?\s*years", exc_lower)
                exc_years = int(m_req.group(1)) if m_req else 7
                m_job_years = re.search(r"\b(\d+)\+?\s*years(?:\s+of)?\s*(?:experience|working)?\b", job_full_text, re.IGNORECASE)
                if m_job_years and int(m_job_years.group(1)) >= exc_years:
                    is_excluded = True
                    break
            else:
                # Custom exclusion: check full text with word boundary matching
                term_pattern = r"(?<![a-zA-Z0-9_])" + re.escape(exc_lower) + r"(?![a-zA-Z0-9_])"
                if re.search(term_pattern, job_full_text, re.IGNORECASE):
                    is_excluded = True
                    break

        if is_excluded:
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
            is_remote_job = job.work_mode == WorkMode.REMOTE or "remote" in job_loc
            
            loc_matched = loc_pref in job_loc or loc_pref in job_full_text
            if loc_pref == "israel" or "israel" in loc_pref:
                israel_aliases = [", il", " il", "(il)", "- il", "il,", "isr", "ישראל", "tel aviv", "herzliya", "haifa", "petah tikva", "jerusalem", "rehovot", "netanya", "ra'anana", "ramat gan", "holon", "rishon lezion", "yokneam", "beer sheva", "karmiel", "azor"]
                if any(alias in job_loc or alias in job_full_text for alias in israel_aliases):
                    loc_matched = True

            if not loc_matched:
                if not (is_remote_job and "remote" in loc_pref):
                    continue

        # 4. Minimum salary filter
        if prefs.min_salary and job.salary_range:
            parsed_salary = _parse_salary_number(job.salary_range)
            if parsed_salary is not None and parsed_salary < prefs.min_salary:
                continue

        # 5. Calculate match score and enrich job fields
        calculate_match_score(job, prefs, profile=profile, display_map=display_map)
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

    seniority_level = detect_seniority_level(title, combined_desc)
    description_summary = generate_description_summary(combined_desc)

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
        seniority_level=seniority_level,
        description_summary=description_summary,
    )


DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


async def fetch_jobs_via_api(
    request_context: Any = None,
    page: int = 1,
    size: int = 50,
    sort_by: str = "posted_date",
    sort_order: str = "desc",
    client: Optional[httpx.AsyncClient] = None,
) -> list[Job]:
    """Fetch jobs directly via the HireMeTech REST API using httpx or Playwright APIRequestContext.

    Args:
        request_context: Optional Playwright APIRequestContext or httpx.AsyncClient instance.
        page: Page number (1-indexed).
        size: Number of jobs per page.
        sort_by: Field to sort by ('posted_date', 'relevance', etc.).
        sort_order: Sort direction ('desc' or 'asc').
        client: Optional httpx.AsyncClient instance.

    Returns:
        list[Job]: Parsed list of Job objects.

    Raises:
        RuntimeError: If the API returns a non-200 status code.
    """
    url = f"{BASE_URL}/api/jobs/search?page={page}&size={size}&sort_by={sort_by}&sort_order={sort_order}&country=Israel&israeli=true"
    logger.info("Fetching jobs from API: %s", url)

    if client is not None or isinstance(request_context, httpx.AsyncClient):
        http_client = client if isinstance(client, httpx.AsyncClient) else request_context
        t0 = time.perf_counter()
        resp = await http_client.get(url, headers=DEFAULT_HTTP_HEADERS)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        status_code = resp.status_code
        if status_code != 200:
            error_msg = f"HireMe API returned status {status_code} for {url}"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)
        data = resp.json()
    elif request_context is None:
        async with httpx.AsyncClient(timeout=6.0, headers=DEFAULT_HTTP_HEADERS) as http_client:
            t0 = time.perf_counter()
            resp = await http_client.get(url)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            status_code = resp.status_code
            if status_code != 200:
                error_msg = f"HireMe API returned status {status_code} for {url}"
                logger.warning(error_msg)
                raise RuntimeError(error_msg)
            data = resp.json()
    else:
        # Playwright APIRequestContext or test mock
        t0 = time.perf_counter()
        resp = await request_context.get(url)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        status_code = getattr(resp, "status", getattr(resp, "status_code", 200))
        if status_code != 200:
            error_msg = f"HireMe API returned status {status_code} for {url}"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)

        json_method = getattr(resp, "json", None)
        if callable(json_method):
            json_res = json_method()
            if asyncio.iscoroutine(json_res):
                data = await json_res
            else:
                data = json_res
        elif hasattr(resp, "json"):
            data = resp.json
        else:
            data = {}

    raw_jobs = data.get("jobs", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if isinstance(data, dict) and not raw_jobs and "data" in data:
        raw_jobs = data["data"]

    jobs = [parse_api_job_dict(item) for item in raw_jobs if isinstance(item, dict)]
    logger.info(
        "HTTP API request completed",
        url=url,
        status=status_code,
        duration_ms=round(duration_ms, 2),
        jobs_count=len(jobs),
        source="hiremetech",
    )
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
