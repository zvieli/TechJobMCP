"""Domain contracts and protocols for job sources."""

from __future__ import annotations

import typing
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from job_mcp.models.schemas import Job, UserPreferences

if TYPE_CHECKING:
    from job_mcp.sources.base import SourceMetadata


class SourceCategory(str, Enum):
    """Categorization of job sources for discovery and execution strategies."""

    PUBLIC = "public"
    ENTERPRISE = "enterprise"
    AUTHENTICATED = "authenticated"


@runtime_checkable
class IJobSource(Protocol):
    """Core contract that every job source must satisfy."""

    source_id: str
    display_name: str
    description: str
    category: SourceCategory

    async def fetch_jobs(
        self, preferences: UserPreferences | None = None, limit: int = 20
    ) -> list[Job]:
        """Fetch job listings matching optional preferences up to limit."""
        ...

    async def check_health(self) -> bool:
        """Check operational readiness and accessibility of the source."""
        ...

    def get_metadata(self) -> SourceMetadata:
        """Return standardized metadata describing source capabilities."""
        ...


class _BookmarkableMeta(typing._ProtocolMeta):
    """Metaclass ensuring IBookmarkable cleanly checks actual bookmark support."""

    def __instancecheck__(cls, instance: Any) -> bool:
        if not super().__instancecheck__(instance):
            return False
        if getattr(instance, "supports_bookmarks", True) is False:
            return False
        method = getattr(type(instance), "bookmark_job", None)
        if getattr(method, "_is_base_job_source_default", False):
            return False
        return True


@runtime_checkable
class IBookmarkable(Protocol, metaclass=_BookmarkableMeta):
    """Protocol for sources supporting job bookmarking / saving."""

    async def bookmark_job(self, job_id: str) -> bool:
        """Bookmark/favorite a job listing by ID."""
        ...


@runtime_checkable
class IAuthenticatedSource(Protocol):
    """Protocol for sources requiring session management or authentication."""

    session_manager: Any
    is_authenticated: bool

    async def ensure_authenticated(self) -> bool:
        """Ensure the session or credentials are valid and active."""
        ...


@runtime_checkable
class IConfigurableSource(Protocol):
    """Protocol for sources accepting dynamic runtime configuration."""

    def configure(self, **kwargs: Any) -> None:
        """Configure source credentials, endpoints, or preferences."""
        ...
