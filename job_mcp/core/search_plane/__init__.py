"""Unified Search Plane domain contracts and retrieval plane for TechJobMCP (Milestone 6)."""

from __future__ import annotations

from job_mcp.core.search_plane.adapter import (
    SOURCE_CAPABILITY_MAP,
    SearchPlaneAdapter,
    get_source_capabilities,
)
from job_mcp.core.search_plane.discovery import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
    discover_companies,
    discover_company,
    to_registry_config,
)
from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
    JobSearchRequest,
    JobSearchResultItem,
    JobSearchResultSet,
    SourceCapabilities,
)

__all__ = [
    "SOURCE_CAPABILITY_MAP",
    "DiscoveredPortal",
    "DiscoveryEvidence",
    "DiscoveryResult",
    "DiscoveryStatus",
    "DiscoveryTarget",
    "FetchResult",
    "FetchStatus",
    "JobRef",
    "JobSearchRequest",
    "JobSearchResultItem",
    "JobSearchResultSet",
    "SearchPlaneAdapter",
    "SourceCapabilities",
    "discover_companies",
    "discover_company",
    "get_source_capabilities",
    "to_registry_config",
]
