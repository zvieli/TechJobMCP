"""Base classes and metadata models for job sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from job_mcp.models.schemas import Job, JobPreferences
from job_mcp.sources.contracts import (
    SourceCategory,
)


class SourceMetadata(BaseModel):
    """Metadata describing a job source and its capabilities."""

    source_id: str
    display_name: str
    description: str = ""
    category: SourceCategory = SourceCategory.PUBLIC
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False


class BaseJobSource(ABC):
    """Abstract base class for all job search and aggregation sources."""

    source_id: str = ""
    display_name: str = ""
    description: str = ""
    category: SourceCategory = SourceCategory.PUBLIC
    is_authenticated: bool = False
    supports_auth: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def get_metadata(self) -> SourceMetadata:
        """Return the metadata descriptor for this source."""
        return SourceMetadata(
            source_id=self.source_id,
            display_name=self.display_name,
            description=self.description,
            category=self.category,
            is_authenticated=self.is_authenticated,
            supports_bookmarks=self.supports_bookmarks,
            supports_auto_apply=self.supports_auto_apply,
        )

    @abstractmethod
    async def fetch_jobs(
        self,
        preferences: Optional[JobPreferences] = None,
        limit: int = 50,
    ) -> list[Job]:
        """Fetch job listings matching optional preferences up to limit.

        Args:
            preferences: Optional JobPreferences filter.
            limit: Maximum number of jobs to retrieve.

        Returns:
            list[Job]: List of standardized Job objects.
        """
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check the operational health and readiness of this source.

        Returns:
            bool: True if source is healthy and accessible, False otherwise.
        """
        pass

    async def bookmark_job(self, job_id: str) -> bool:
        """Bookmark/favorite a job listing by ID.

        Args:
            job_id: ID of the job listing.

        Returns:
            bool: True if bookmarked successfully.

        Raises:
            NotImplementedError: If source supports bookmarks but has not implemented this method.
        """
        if not self.supports_bookmarks:
            return False
        raise NotImplementedError(f"bookmark_job not implemented for source '{self.source_id}'")

    bookmark_job._is_base_job_source_default = True  # type: ignore[attr-defined]


class BasePublicSource(BaseJobSource):
    """Base class for public job boards that do not require authentication."""

    category: SourceCategory = SourceCategory.PUBLIC
    is_authenticated: bool = False
    supports_auth: bool = False


class BaseEnterpriseSource(BaseJobSource):
    """Base class for enterprise ATS and career platform sources."""

    category: SourceCategory = SourceCategory.ENTERPRISE
    supports_auth: bool = False


class BaseAuthenticatedSource(BaseJobSource):
    """Base class for authenticated job sources requiring session management."""

    category: SourceCategory = SourceCategory.AUTHENTICATED
    supports_auth: bool = True

    def __init__(self, session_manager: Any = None) -> None:
        """Initialize authenticated source with optional session manager."""
        self.session_manager: Any = session_manager
        self._is_authenticated: Optional[bool] = None

    @property
    def is_authenticated(self) -> bool:
        """Check if source is currently authenticated."""
        if self._is_authenticated is not None:
            return self._is_authenticated
        if self.session_manager is not None:
            if hasattr(self.session_manager, "is_authenticated"):
                return bool(self.session_manager.is_authenticated)
            if hasattr(self.session_manager, "is_running"):
                return bool(self.session_manager.is_running)
            return True
        return False

    @is_authenticated.setter
    def is_authenticated(self, value: bool) -> None:
        """Set authentication state."""
        self._is_authenticated = value

    async def ensure_authenticated(self) -> bool:
        """Ensure session or credentials are valid and active."""
        if self.session_manager is not None:
            if hasattr(self.session_manager, "ensure_ready"):
                try:
                    await self.session_manager.ensure_ready()
                    self._is_authenticated = True
                    return True
                except Exception:
                    self._is_authenticated = False
                    return False
            if hasattr(self.session_manager, "check_session_health"):
                try:
                    res = await self.session_manager.check_session_health()
                    self._is_authenticated = bool(res)
                    return bool(res)
                except Exception:
                    self._is_authenticated = False
                    return False
            self._is_authenticated = True
            return True
        return self.is_authenticated


__all__ = [
    "SourceMetadata",
    "BaseJobSource",
    "BasePublicSource",
    "BaseEnterpriseSource",
    "BaseAuthenticatedSource",
]
