"""Validated, configuration-driven company registry (Milestone 5).

Public surface. Providers import :func:`catalog` / :func:`enabled_catalog` to
resolve their company set; nothing in this package performs network I/O.
"""

from __future__ import annotations

from job_mcp.sources.company_registry.core import (
    CONFIG_PATH_ENV_VAR,
    DEFAULT_CONFIG_FILENAMES,
    CompanyRegistry,
    CompanyRegistryError,
    catalog,
    configure,
    discover_config_paths,
    effective_catalog,
    enabled_catalog,
    get_registry,
    load_config,
    reset,
    resolve,
    supported_providers,
    validate_document,
)
from job_mcp.sources.company_registry.entries import (
    AshbyCompany,
    DirectTechCompany,
    EightfoldCompany,
    GreenhouseCompany,
    LeverCompany,
    SmartRecruitersCompany,
    WorkdayCompany,
)
from job_mcp.sources.company_registry.schema import RegistryConfig

__all__ = [
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG_FILENAMES",
    "AshbyCompany",
    "CompanyRegistry",
    "CompanyRegistryError",
    "DirectTechCompany",
    "EightfoldCompany",
    "GreenhouseCompany",
    "LeverCompany",
    "RegistryConfig",
    "SmartRecruitersCompany",
    "WorkdayCompany",
    "catalog",
    "configure",
    "discover_config_paths",
    "effective_catalog",
    "enabled_catalog",
    "get_registry",
    "load_config",
    "reset",
    "resolve",
    "supported_providers",
    "validate_document",
]
