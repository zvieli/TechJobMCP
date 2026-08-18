"""Core package for HireMeTech MCP server."""

from hireme_mcp.core.api_client import (
    JobCache,
    extract_cv_keywords,
    fetch_jobs_via_api,
    fetch_saved_jobs_batch,
    fetch_user_resume_profile,
    filter_jobs,
    parse_api_job_dict,
)
from hireme_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    DEFAULT_PROFILE_DIR,
    LOGIN_PATH,
    SessionManager,
)
from hireme_mcp.core.browser import (
    SELECTORS,
    _resolve_selector,
    bookmark_job,
    delete_job,
    dynamic_registry,
    execute_application,
    extract_jobs,
    preview_application,
)
from hireme_mcp.core.discovery import (
    DynamicSelectorRegistry,
    calibrate_all_selectors,
    discover_card_selector,
    discover_child_selector,
)

__all__ = [
    "BASE_URL",
    "DASHBOARD_PATH",
    "LOGIN_PATH",
    "DEFAULT_PROFILE_DIR",
    "SessionManager",
    "SELECTORS",
    "dynamic_registry",
    "DynamicSelectorRegistry",
    "discover_card_selector",
    "discover_child_selector",
    "calibrate_all_selectors",
    "_resolve_selector",
    "extract_jobs",
    "bookmark_job",
    "delete_job",
    "preview_application",
    "execute_application",
    "JobCache",
    "extract_cv_keywords",
    "filter_jobs",
    "parse_api_job_dict",
    "fetch_jobs_via_api",
    "fetch_saved_jobs_batch",
    "fetch_user_resume_profile",
]
