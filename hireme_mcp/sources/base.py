"""Base classes and metadata models for job sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from hireme_mcp.models.schemas import Job, JobPreferences


class SourceMetadata(BaseModel):
    """Metadata describing a job source and its capabilities."""

    source_id: str
    display_name: str
    description: str = ""
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False


class BaseJobSource(ABC):
    """Abstract base class for all job search and aggregation sources."""

    source_id: str = ""
    display_name: str = ""
    description: str = ""
    is_authenticated: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False

    def get_metadata(self) -> SourceMetadata:
        """Return the metadata descriptor for this source."""
        return SourceMetadata(
            source_id=self.source_id,
            display_name=self.display_name,
            description=self.description,
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
