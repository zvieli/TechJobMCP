"""Core package for HireMeTech MCP server."""

from hireme_mcp.core.api_client import (
    JobCache,
    extract_cv_keywords,
    filter_jobs,
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
    execute_application,
    extract_jobs,
    preview_application,
)

__all__ = [
    "BASE_URL",
    "DASHBOARD_PATH",
    "LOGIN_PATH",
    "DEFAULT_PROFILE_DIR",
    "SessionManager",
    "SELECTORS",
    "_resolve_selector",
    "extract_jobs",
    "bookmark_job",
    "delete_job",
    "preview_application",
    "execute_application",
    "JobCache",
    "extract_cv_keywords",
    "filter_jobs",
]
