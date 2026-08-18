"""Sources module for multi-source job fetching, registry, and deduplication."""

from __future__ import annotations

import os
from typing import Any, Optional

from job_mcp.sources.base import BaseJobSource, SourceMetadata
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
from job_mcp.sources.aggregator import DEFAULT_SOURCE_TIMEOUT, JobAggregator


class SourceRegistry:
    """Registry holding registered BaseJobSource instances keyed by source_id."""

    def __init__(self) -> None:
        """Initialize an empty SourceRegistry."""
        self._sources: dict[str, BaseJobSource] = {}

    def register(self, source: BaseJobSource) -> None:
        """Register a job source instance.

        Args:
            source: BaseJobSource instance to register.

        Raises:
            TypeError: If source is not an instance of BaseJobSource.
            ValueError: If source has an empty source_id.
        """
        if not isinstance(source, BaseJobSource):
            raise TypeError(f"Expected BaseJobSource instance, got {type(source)}")
        if not source.source_id or not source.source_id.strip():
            raise ValueError("Source must have a non-empty source_id")
        self._sources[source.source_id] = source

    def unregister(self, source_id: str) -> Optional[BaseJobSource]:
        """Unregister a job source by ID.

        Args:
            source_id: Unique string identifier of the source.

        Returns:
            Optional[BaseJobSource]: Removed source instance if found, else None.
        """
        return self._sources.pop(source_id, None)

    def get(self, source_id: str) -> Optional[BaseJobSource]:
        """Get a registered job source by ID.

        Args:
            source_id: Unique string identifier of the source.

        Returns:
            Optional[BaseJobSource]: Found source instance or None.
        """
        return self._sources.get(source_id)

    def list_sources(self) -> list[SourceMetadata]:
        """Return metadata descriptors for all registered sources.

        Returns:
            list[SourceMetadata]: List of metadata objects for registered sources.
        """
        return [source.get_metadata() for source in self._sources.values()]

    def get_all(self) -> list[BaseJobSource]:
        """Return all registered job source instances.

        Returns:
            list[BaseJobSource]: All registered sources.
        """
        return list(self._sources.values())

    def get_active(self, source_ids: Optional[list[str]] = None) -> list[BaseJobSource]:
        """Return active sources, optionally filtered by source_ids.

        Args:
            source_ids: Optional list of source ID strings. If None, returns all registered sources.

        Returns:
            list[BaseJobSource]: Filtered list of registered sources.
        """
        if source_ids is None:
            return list(self._sources.values())
        return [self._sources[sid] for sid in source_ids if sid in self._sources]

    def clear(self) -> None:
        """Clear all registered sources."""
        self._sources.clear()

    def __contains__(self, source_id: str) -> bool:
        """Check if source_id is in registry."""
        return source_id in self._sources

    def __len__(self) -> int:
        """Return count of registered sources."""
        return len(self._sources)


def create_default_registry(session_manager: Optional[Any] = None) -> SourceRegistry:
    """Create and return a SourceRegistry pre-populated with standard job sources."""
    reg = SourceRegistry()
    reg.register(HireMeTechSource(session_manager=session_manager))
    reg.register(ComeetSource())

    enable_alljobs = os.getenv("ENABLE_ALLJOBS", "false").strip().lower() in ("true", "1", "yes")
    if enable_alljobs:
        reg.register(AllJobsSource())

    enable_workday = os.getenv("ENABLE_WORKDAY", "false").strip().lower() in ("true", "1", "yes")
    if enable_workday:
        reg.register(WorkdaySource())

    enable_eightfold = os.getenv("ENABLE_EIGHTFOLD", "false").strip().lower() in ("true", "1", "yes")
    if enable_eightfold:
        reg.register(EightfoldAISource())

    enable_direct_tech = os.getenv("ENABLE_DIRECT_TECH", "false").strip().lower() in ("true", "1", "yes")
    if enable_direct_tech:
        reg.register(DirectTechSource())

    enable_linkedin = os.getenv("ENABLE_LINKEDIN", "false").strip().lower() in ("true", "1", "yes")
    if enable_linkedin:
        reg.register(LinkedInSource(session_manager=session_manager))
    return reg


# Global default registry with standard sources
registry = create_default_registry()

__all__ = [
    # Metadata & Base
    "SourceMetadata",
    "BaseJobSource",
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
    # Registry & Aggregator
    "SourceRegistry",
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
