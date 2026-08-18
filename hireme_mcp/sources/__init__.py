"""Sources module for multi-source job fetching, registry, and deduplication."""

from __future__ import annotations

from typing import Optional

from hireme_mcp.sources.base import BaseJobSource, SourceMetadata
from hireme_mcp.sources.dedup import (
    compute_dedup_key,
    deduplicate_jobs,
    merge_job_entities,
    normalize_company,
    normalize_title,
)
from hireme_mcp.sources.comeet import (
    DEFAULT_COMEET_COMPANIES,
    ComeetCompany,
    ComeetSource,
    parse_comeet_position,
)
from hireme_mcp.sources.hiremetech import HireMeTechSource


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


# Global default registry
registry = SourceRegistry()

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
    # Registry
    "SourceRegistry",
    "registry",
    # Deduplication & Merging
    "normalize_title",
    "normalize_company",
    "compute_dedup_key",
    "merge_job_entities",
    "deduplicate_jobs",
]
