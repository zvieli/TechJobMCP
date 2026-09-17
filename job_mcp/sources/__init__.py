"""Sources module for multi-source job fetching, registry, and deduplication."""

from __future__ import annotations

import os
from typing import Any, Optional

from job_mcp.sources.base import (
    BaseAuthenticatedSource,
    BaseEnterpriseSource,
    BaseJobSource,
    BasePublicSource,
    SourceMetadata,
)
from job_mcp.sources.contracts import (
    IAuthenticatedSource,
    IBookmarkable,
    IConfigurableSource,
    IJobSource,
    SourceCategory,
)
from job_mcp.sources.dedup import (
    compute_dedup_key,
    deduplicate_jobs,
    merge_job_entities,
    normalize_company,
    normalize_title,
)
from job_mcp.sources.alljobs import (
    ALLJOBS_BASE_URL,
    ALLJOBS_HEADERS,
    DEFAULT_TECH_CATEGORIES,
    AllJobsSource,
    parse_alljobs_position,
)
from job_mcp.sources.comeet import (
    DEFAULT_COMEET_COMPANIES,
    ComeetCompany,
    ComeetSource,
    parse_comeet_position,
)
from job_mcp.sources.workday import (
    DEFAULT_WORKDAY_COMPANIES,
    WORKDAY_COMPANIES,
    WorkdayCompany,
    WorkdaySource,
    parse_workday_position,
)
from job_mcp.sources.eightfold import (
    DEFAULT_EIGHTFOLD_COMPANIES,
    EIGHTFOLD_COMPANIES,
    EightfoldAISource,
    EightfoldCompany,
    parse_eightfold_position,
)
from job_mcp.sources.direct_tech import (
    DEFAULT_DIRECT_TECH_COMPANIES,
    DIRECT_TECH_COMPANIES,
    DirectTechCompany,
    DirectTechSource,
    parse_amazon_position,
    parse_amazon_positions,
    parse_apple_position,
    parse_apple_positions,
    parse_google_job,
    parse_google_positions,
    parse_ibm_position,
    parse_ibm_positions,
)
from job_mcp.sources.linkedin import (
    LINKEDIN_HEADERS,
    LINKEDIN_JOB_DETAIL_URL,
    LINKEDIN_SEARCH_API_URL,
    LinkedInSource,
    parse_linkedin_job_card,
    parse_linkedin_job_details,
    parse_linkedin_search_results,
    search_linkedin_jobs_api,
)
from job_mcp.sources.hiremetech import HireMeTechSource
from job_mcp.sources.jobify import (
    DEFAULT_JOBIFY_SEED_URLS,
    DEFAULT_SEED_URLS,
    JOBIFY_BASE_URL,
    JOBIFY_HEADERS,
    JobifySource,
    extract_jsonld_job_postings,
    extract_related_job_urls,
    parse_jobify_position,
)
from job_mcp.sources.aggregator import DEFAULT_SOURCE_TIMEOUT, JobAggregator


from job_mcp.sources.registry import (
    SourceProvider,
    SourceRegistry,
    clear_providers,
    create_default_registry,
    get_registered_providers,
    register_provider,
    register_source_provider,
    registry,
    reset_builtin_providers,
    unregister_provider,
)

__all__ = [
    # Metadata & Base & Contracts
    "SourceMetadata",
    "BaseJobSource",
    "BasePublicSource",
    "BaseEnterpriseSource",
    "BaseAuthenticatedSource",
    "SourceCategory",
    "IJobSource",
    "IBookmarkable",
    "IAuthenticatedSource",
    "IConfigurableSource",
    # Implementations
    "HireMeTechSource",
    "ComeetSource",
    "ComeetCompany",
    "DEFAULT_COMEET_COMPANIES",
    "parse_comeet_position",
    "WorkdaySource",
    "WorkdayCompany",
    "WORKDAY_COMPANIES",
    "DEFAULT_WORKDAY_COMPANIES",
    "parse_workday_position",
    "EightfoldAISource",
    "EightfoldCompany",
    "EIGHTFOLD_COMPANIES",
    "DEFAULT_EIGHTFOLD_COMPANIES",
    "parse_eightfold_position",
    "AllJobsSource",
    "ALLJOBS_BASE_URL",
    "ALLJOBS_HEADERS",
    "DEFAULT_TECH_CATEGORIES",
    "parse_alljobs_position",
    "DirectTechSource",
    "DirectTechCompany",
    "DIRECT_TECH_COMPANIES",
    "DEFAULT_DIRECT_TECH_COMPANIES",
    "parse_google_job",
    "parse_google_positions",
    "parse_amazon_position",
    "parse_amazon_positions",
    "parse_apple_position",
    "parse_apple_positions",
    "parse_ibm_position",
    "parse_ibm_positions",
    "LinkedInSource",
    "parse_linkedin_job_card",
    "parse_linkedin_search_results",
    "parse_linkedin_job_details",
    "search_linkedin_jobs_api",
    "LINKEDIN_SEARCH_API_URL",
    "LINKEDIN_JOB_DETAIL_URL",
    "LINKEDIN_HEADERS",
    "JobifySource",
    "DEFAULT_SEED_URLS",
    "DEFAULT_JOBIFY_SEED_URLS",
    "JOBIFY_BASE_URL",
    "JOBIFY_HEADERS",
    "parse_jobify_position",
    "extract_jsonld_job_postings",
    "extract_related_job_urls",
    # Registry & Aggregator
    "SourceRegistry",
    "SourceProvider",
    "register_provider",
    "unregister_provider",
    "get_registered_providers",
    "clear_providers",
    "register_source_provider",
    "reset_builtin_providers",
    "create_default_registry",
    "registry",
    "JobAggregator",
    "DEFAULT_SOURCE_TIMEOUT",
    # Deduplication & Merging
    "normalize_title",
    "normalize_company",
    "compute_dedup_key",
    "merge_job_entities",
    "deduplicate_jobs",
]
