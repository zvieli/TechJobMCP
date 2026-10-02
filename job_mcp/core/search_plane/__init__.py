"""Unified Search Plane domain contracts and retrieval plane for TechJobMCP (Milestone 6)."""

from __future__ import annotations

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
    "FetchResult",
    "FetchStatus",
    "JobRef",
    "JobSearchRequest",
    "JobSearchResultItem",
    "JobSearchResultSet",
    "SourceCapabilities",
]
