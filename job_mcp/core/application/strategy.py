"""Base strategy class and registry for job application routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional, Type

from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job

logger = logging.getLogger(__name__)


class ApplicationStrategy(ABC):
    """Abstract Base Class for all ATS application submitter strategies."""

    method: ApplicationMethod | str = ApplicationMethod.API

    @abstractmethod
    async def preview(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Inspect and preview the application fields, method, and warnings without submitting.

        Args:
            job: The target Job listing.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional path to CV document.

        Returns:
            ApplicationPreview: Form fields, submission method, and validation warnings.
        """
        pass

    @abstractmethod
    async def apply(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute submission of job application.

        Args:
            job: The target Job listing.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional path to CV document.

        Returns:
            dict[str, Any]: Submission outcome including success status, tracking ID, and response details.
        """
        pass


# Dynamic strategy registry mapping source identifiers/prefixes to Strategy classes
_STRATEGY_REGISTRY: dict[str, Type[ApplicationStrategy]] = {}


def register_application_strategy(
    source_key: str,
    strategy_cls: Type[ApplicationStrategy],
) -> None:
    """Register a custom strategy class for a specific source key or prefix.

    Args:
        source_key: Key or prefix identifying the job source (e.g., 'comeet', 'linkedin').
        strategy_cls: Concrete ApplicationStrategy subclass.
    """
    key = str(source_key).lower().strip()
    _STRATEGY_REGISTRY[key] = strategy_cls
    logger.debug("Registered strategy '%s' for source key '%s'", strategy_cls.__name__, key)


def get_application_strategy(
    source: str,
    session_manager: Optional[Any] = None,
) -> ApplicationStrategy:
    """Resolve and instantiate the appropriate ApplicationStrategy for a job source.

    Strategy routing:
    1. Direct API POST (`ApiPostStrategy`):
       - Comeet, HireMeTech, Greenhouse, Lever, Direct Tech API, or explicit 'api'/'api_direct'.
    2. Easy Apply (`EasyApplyStrategy`):
       - LinkedIn, Easy Apply quick submissions, or explicit 'easy_apply'/'easyapply'.
    3. Browser Playwright (`BrowserPlaywrightStrategy`):
       - Workday, AllJobs, Eightfold, generic browser automation, or fallback.

    Args:
        source: Source name or identifier from Job.source or Job.job_id.
        session_manager: Optional browser session manager or active browser context.

    Returns:
        ApplicationStrategy: Concrete strategy instance configured for the source.
    """
    from job_mcp.core.application.strategies.api import ApiPostStrategy
    from job_mcp.core.application.strategies.browser import BrowserPlaywrightStrategy
    from job_mcp.core.application.strategies.easy_apply import EasyApplyStrategy

    src = str(source or "").lower().strip()

    # Check dynamic registry first
    if src in _STRATEGY_REGISTRY:
        strategy_cls = _STRATEGY_REGISTRY[src]
        if issubclass(strategy_cls, (BrowserPlaywrightStrategy, EasyApplyStrategy)):
            return strategy_cls(session_manager=session_manager)
        return strategy_cls()

    for reg_key, reg_cls in _STRATEGY_REGISTRY.items():
        if src.startswith(reg_key) or reg_key in src:
            if issubclass(reg_cls, (BrowserPlaywrightStrategy, EasyApplyStrategy)):
                return reg_cls(session_manager=session_manager)
            return reg_cls()

    # Built-in source routing rules
    # 1. API POST Sources
    api_sources = ("comeet", "hiremetech", "greenhouse", "lever", "api", "api_direct", "direct_tech")
    if any(s in src for s in api_sources):
        return ApiPostStrategy()

    # 2. Easy Apply Sources
    easy_apply_sources = ("linkedin", "easy_apply", "easyapply", "quick_apply")
    if any(s in src for s in easy_apply_sources):
        return EasyApplyStrategy(session_manager=session_manager)

    # 3. Browser / DOM Fallback Sources
    browser_sources = ("workday", "alljobs", "eightfold", "browser", "playwright")
    if any(s in src for s in browser_sources):
        return BrowserPlaywrightStrategy(session_manager=session_manager)

    # Default fallback: Browser Playwright strategy
    return BrowserPlaywrightStrategy(session_manager=session_manager)
