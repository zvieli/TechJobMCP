"""Public job sources subpackage."""

from __future__ import annotations

from job_mcp.sources.public.alljobs import (
    ALLJOBS_BASE_URL,
    ALLJOBS_HEADERS,
    DEFAULT_TECH_CATEGORIES,
    AllJobsSource,
    parse_alljobs_position,
)
from job_mcp.sources.public.comeet import (
    DEFAULT_COMEET_COMPANIES,
    ComeetCompany,
    ComeetSource,
    parse_comeet_position,
)
from job_mcp.sources.public.eightfold import (
    DEFAULT_EIGHTFOLD_COMPANIES,
    EIGHTFOLD_COMPANIES,
    EightfoldAISource,
    EightfoldCompany,
    parse_eightfold_position,
)
from job_mcp.sources.public.greenhouse import (
    GREENHOUSE_COMPANIES,
    GreenhouseCompany,
    GreenhouseSource,
    parse_greenhouse_job,
)
from job_mcp.sources.public.jobify import (
    DEFAULT_JOBIFY_SEED_URLS,
    DEFAULT_SEED_URLS,
    JOBIFY_BASE_URL,
    JOBIFY_HEADERS,
    JobifySource,
    extract_jsonld_job_postings,
    extract_related_job_urls,
    parse_jobify_position,
)

__all__ = [
    # AllJobs
    "ALLJOBS_BASE_URL",
    "ALLJOBS_HEADERS",
    "DEFAULT_TECH_CATEGORIES",
    "AllJobsSource",
    "parse_alljobs_position",
    # Comeet
    "ComeetCompany",
    "ComeetSource",
    "DEFAULT_COMEET_COMPANIES",
    "parse_comeet_position",
    # Eightfold
    "DEFAULT_EIGHTFOLD_COMPANIES",
    "EIGHTFOLD_COMPANIES",
    "EightfoldAISource",
    "EightfoldCompany",
    "parse_eightfold_position",
    # Greenhouse
    "GREENHOUSE_COMPANIES",
    "GreenhouseCompany",
    "GreenhouseSource",
    "parse_greenhouse_job",
    # Jobify
    "DEFAULT_JOBIFY_SEED_URLS",
    "DEFAULT_SEED_URLS",
    "JOBIFY_BASE_URL",
    "JOBIFY_HEADERS",
    "JobifySource",
    "extract_jsonld_job_postings",
    "extract_related_job_urls",
    "parse_jobify_position",
]
