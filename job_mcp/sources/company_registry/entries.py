"""Typed company entry descriptors for configuration-driven provider families.

These dataclasses are the normalized currency exchanged between the company
registry layer and provider constructors. Providers receive instances of these
types; they never parse YAML or JSON themselves.

The classes live here, rather than in provider modules, so the registry can
build them without importing provider code (which would create an import cycle,
because providers resolve their catalog through the registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AshbyCompany:
    """Descriptor for a company using Ashby ATS."""

    name: str
    board_name: str
    enabled: bool = True


@dataclass
class GreenhouseCompany:
    """Descriptor for a company using Greenhouse ATS."""

    name: str
    board_token: str
    enabled: bool = True


@dataclass
class LeverCompany:
    """Descriptor for a company using Lever ATS."""

    name: str
    slug: str
    enabled: bool = True


@dataclass
class EightfoldCompany:
    """Descriptor for a company using Eightfold AI ATS."""

    name: str
    hostname: str
    domain: str
    locations: list[str] = field(default_factory=list)
    filter_distance: str | None = "16"
    enabled: bool = True

    def get_search_url(self) -> str:
        """Return the PCSX search endpoint URL."""
        return f"https://{self.hostname.rstrip('/')}/api/pcsx/search"

    def get_job_url(self, position_url: str) -> str:
        """Return the public apply / details URL for a job posting."""
        if not position_url:
            return f"https://{self.hostname.rstrip('/')}"
        if position_url.startswith(("http://", "https://")):
            return position_url
        if not position_url.startswith("/"):
            position_url = f"/{position_url}"
        return f"https://{self.hostname.rstrip('/')}{position_url}"


# The narrowing query applied when a caller supplies no keywords. Declared once
# here and reused as the configuration schema default, so the entry type and the
# registry cannot drift to different values.
DIRECT_TECH_DEFAULT_QUERY: str = "student"


@dataclass
class DirectTechCompany:
    """Descriptor for a direct tech company career endpoint."""

    provider_id: str
    name: str
    search_url: str
    default_query: str = DIRECT_TECH_DEFAULT_QUERY
    default_location: str = ""
    locations: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class WorkdayCompany:
    """Descriptor for a company using Workday ATS."""

    name: str
    wd_company: str
    wd_version: int = 1
    wd_suffix: str = "External"
    wd_locations: list[str] = field(default_factory=list)
    base_url: str | None = None
    enabled: bool = True

    def get_base_url(self) -> str:
        """Return the base URL for the company's Workday portal."""
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"https://{self.wd_company}.wd{self.wd_version}.myworkdayjobs.com"

    def get_cxs_url(self) -> str:
        """Return the CXS search endpoint URL."""
        base = self.get_base_url()
        return f"{base}/wday/cxs/{self.wd_company}/{self.wd_suffix}/jobs"

    def get_job_url(self, external_path: str) -> str:
        """Return the public apply / details URL for a job posting."""
        base = self.get_base_url()
        if not external_path:
            return f"{base}/en-US/{self.wd_suffix}"
        if external_path.startswith(("http://", "https://")):
            return external_path
        if not external_path.startswith("/"):
            external_path = f"/{external_path}"
        return f"{base}/en-US/{self.wd_suffix}{external_path}"


__all__ = [
    "DIRECT_TECH_DEFAULT_QUERY",
    "AshbyCompany",
    "DirectTechCompany",
    "EightfoldCompany",
    "GreenhouseCompany",
    "LeverCompany",
    "WorkdayCompany",
]
