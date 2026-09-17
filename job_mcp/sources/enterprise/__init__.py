"""Enterprise job sources subpackage."""

from __future__ import annotations

from job_mcp.sources.enterprise.direct_tech import (
    DEFAULT_DIRECT_TECH_COMPANIES,
    DIRECT_TECH_COMPANIES,
    DirectTechCompany,
    DirectTechSource,
    USER_AGENT,
    parse_amazon_position,
    parse_amazon_positions,
    parse_apple_position,
    parse_apple_positions,
    parse_google_job,
    parse_google_positions,
    parse_ibm_position,
    parse_ibm_positions,
)
from job_mcp.sources.enterprise.workday import (
    DEFAULT_WORKDAY_COMPANIES,
    WORKDAY_COMPANIES,
    WorkdayCompany,
    WorkdaySource,
    parse_workday_position,
)

__all__ = [
    # Workday
    "DEFAULT_WORKDAY_COMPANIES",
    "WORKDAY_COMPANIES",
    "WorkdayCompany",
    "WorkdaySource",
    "parse_workday_position",
    # Direct Tech
    "DEFAULT_DIRECT_TECH_COMPANIES",
    "DIRECT_TECH_COMPANIES",
    "DirectTechCompany",
    "DirectTechSource",
    "USER_AGENT",
    "parse_amazon_position",
    "parse_amazon_positions",
    "parse_apple_position",
    "parse_apple_positions",
    "parse_google_job",
    "parse_google_positions",
    "parse_ibm_position",
    "parse_ibm_positions",
]
