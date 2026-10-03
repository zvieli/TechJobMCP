"""Discovery plane for TechJobMCP (Milestone 6-C).

Exports domain models, deterministic discovery service, SSRF protection,
and M5 CompanyRegistry configuration projection.
"""

from __future__ import annotations

from job_mcp.core.search_plane.discovery.config import to_registry_config
from job_mcp.core.search_plane.discovery.fingerprints import (
    classify_direct_url,
    inspect_html_body,
)
from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)
from job_mcp.core.search_plane.discovery.security import validate_url_safety
from job_mcp.core.search_plane.discovery.service import (
    DEFAULT_DISCOVERY_BUDGET_SECONDS,
    MAX_BODY_BYTES,
    MAX_REDIRECTS,
    MAX_SLUG_VARIANTS,
    discover_companies,
    discover_company,
)

__all__ = [
    "DEFAULT_DISCOVERY_BUDGET_SECONDS",
    "MAX_BODY_BYTES",
    "MAX_REDIRECTS",
    "MAX_SLUG_VARIANTS",
    "DiscoveredPortal",
    "DiscoveryEvidence",
    "DiscoveryResult",
    "DiscoveryStatus",
    "DiscoveryTarget",
    "classify_direct_url",
    "discover_companies",
    "discover_company",
    "inspect_html_body",
    "to_registry_config",
    "validate_url_safety",
]
