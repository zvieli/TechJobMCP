"""Authenticated job sources subpackage."""

from __future__ import annotations

from job_mcp.sources.authenticated.hiremetech import HireMeTechSource
from job_mcp.sources.authenticated.linkedin import (
    LINKEDIN_DEFAULT_LOCATION,
    LINKEDIN_DEFAULT_MAX_RETRIES,
    LINKEDIN_DEFAULT_RATE_LIMIT_DELAY,
    LINKEDIN_HEADERS,
    LINKEDIN_HEALTH_TIMEOUT,
    LINKEDIN_JOB_DETAIL_URL,
    LINKEDIN_REQUEST_TIMEOUT,
    LINKEDIN_SEARCH_API_URL,
    LinkedInSource,
    clean_html_text,
    clean_linkedin_url,
    extract_job_id_from_text,
    parse_linkedin_job_card,
    parse_linkedin_job_details,
    parse_linkedin_search_results,
    search_linkedin_jobs_api,
)

__all__ = [
    # HireMeTech
    "HireMeTechSource",
    # LinkedIn
    "LINKEDIN_DEFAULT_LOCATION",
    "LINKEDIN_DEFAULT_MAX_RETRIES",
    "LINKEDIN_DEFAULT_RATE_LIMIT_DELAY",
    "LINKEDIN_HEADERS",
    "LINKEDIN_HEALTH_TIMEOUT",
    "LINKEDIN_JOB_DETAIL_URL",
    "LINKEDIN_REQUEST_TIMEOUT",
    "LINKEDIN_SEARCH_API_URL",
    "LinkedInSource",
    "clean_html_text",
    "clean_linkedin_url",
    "extract_job_id_from_text",
    "parse_linkedin_job_card",
    "parse_linkedin_job_details",
    "parse_linkedin_search_results",
    "search_linkedin_jobs_api",
]
