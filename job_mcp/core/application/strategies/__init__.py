"""Application strategy implementations for ATS submission routing."""

from job_mcp.core.application.strategies.api import ApiPostStrategy
from job_mcp.core.application.strategies.browser import BrowserPlaywrightStrategy
from job_mcp.core.application.strategies.easy_apply import EasyApplyStrategy

__all__ = [
    "ApiPostStrategy",
    "EasyApplyStrategy",
    "BrowserPlaywrightStrategy",
]
