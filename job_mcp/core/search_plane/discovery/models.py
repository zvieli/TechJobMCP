"""Domain models for Milestone 6-C ATS and career board discovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiscoveryStatus(str, Enum):
    """Authoritative outward status for a discovery operation."""

    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_FOUND = "NOT_FOUND"
    MALFORMED = "MALFORMED"


@dataclass(frozen=True)
class DiscoveryTarget:
    """Input specification for an ATS/career board discovery request.

    Requires at least one of company_name or career_url to be non-empty.
    """

    company_name: str | None = None
    career_url: str | None = None

    def validate_input(self) -> tuple[bool, str | None]:
        """Check if target contains at least one usable, non-whitespace input."""
        name_clean = self.company_name.strip() if self.company_name is not None else ""
        url_clean = self.career_url.strip() if self.career_url is not None else ""
        if not name_clean and not url_clean:
            return False, "Target must specify at least one non-empty company_name or career_url"
        # Check for control characters in inputs
        for val, label in ((name_clean, "company_name"), (url_clean, "career_url")):
            if val and any(ord(c) < 32 for c in val):
                return False, f"Input {label} contains invalid control characters"
        return True, None


@dataclass(frozen=True)
class DiscoveryEvidence:
    """A factual evidence item supporting a discovery conclusion."""

    kind: str
    value: str
    source_url: str | None = None


@dataclass(frozen=True)
class DiscoveredPortal:
    """A resolved provider candidate with associated confidence evidence."""

    source_family: str
    account: str
    confidence_basis: tuple[DiscoveryEvidence, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of a discovery process against a specific target."""

    status: DiscoveryStatus
    input: DiscoveryTarget
    candidates: tuple[DiscoveredPortal, ...] = ()
    evidence: tuple[DiscoveryEvidence, ...] = ()
    diagnostic: str | None = None

    def to_registry_config(
        self,
        *,
        company_id: str | None = None,
        display_name: str | None = None,
    ) -> str:
        """Convert a CONFIRMED result into a valid M5 portals.yml YAML document."""
        from job_mcp.core.search_plane.discovery.config import to_registry_config

        return to_registry_config(self, company_id=company_id, display_name=display_name)
