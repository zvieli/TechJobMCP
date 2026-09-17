"""Tests for job source protocols and contracts."""

import pytest
from typing import Any
from pydantic import BaseModel

from job_mcp.models.schemas import Job, UserPreferences
from job_mcp.sources.base import SourceMetadata
from job_mcp.sources.contracts import (
    IAuthenticatedSource,
    IBookmarkable,
    IConfigurableSource,
    IJobSource,
    SourceCategory,
)


class DummyFullJobSource:
    source_id: str = "dummy_full"
    display_name: str = "Dummy Full Source"
    description: str = "A full mock source"
    category: SourceCategory = SourceCategory.PUBLIC

    async def fetch_jobs(
        self, preferences: UserPreferences | None = None, limit: int = 20
    ) -> list[Job]:
        return []

    async def check_health(self) -> bool:
        return True

    def get_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            display_name=self.display_name,
            description=self.description,
        )


class DummyIncompleteSource:
    source_id: str = "dummy_incomplete"
    # Missing fetch_jobs, check_health, etc.


class DummyBookmarkableSource:
    async def bookmark_job(self, job_id: str) -> bool:
        return True


class DummyAuthenticatedSource:
    session_manager: Any = None
    is_authenticated: bool = False

    async def ensure_authenticated(self) -> bool:
        return True


class DummyConfigurableSource:
    def configure(self, **kwargs: Any) -> None:
        pass


class DummyHybridEnterpriseSource:
    source_id: str = "hybrid_enterprise"
    display_name: str = "Enterprise ATS"
    description: str = "Enterprise integration with auth and bookmarking"
    category: SourceCategory = SourceCategory.ENTERPRISE
    session_manager: Any = "mock_session"
    is_authenticated: bool = True

    async def fetch_jobs(
        self, preferences: UserPreferences | None = None, limit: int = 20
    ) -> list[Job]:
        return []

    async def check_health(self) -> bool:
        return True

    def get_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            display_name=self.display_name,
            description=self.description,
            is_authenticated=True,
            supports_bookmarks=True,
        )

    async def bookmark_job(self, job_id: str) -> bool:
        return True

    async def ensure_authenticated(self) -> bool:
        return True

    def configure(self, **kwargs: Any) -> None:
        pass


def test_source_category_enum_values():
    assert SourceCategory.PUBLIC == "public"
    assert SourceCategory.ENTERPRISE == "enterprise"
    assert SourceCategory.AUTHENTICATED == "authenticated"
    assert isinstance(SourceCategory.PUBLIC, str)
    assert set(SourceCategory) == {
        SourceCategory.PUBLIC,
        SourceCategory.ENTERPRISE,
        SourceCategory.AUTHENTICATED,
    }


def test_ijobsource_runtime_checkable():
    src = DummyFullJobSource()
    assert isinstance(src, IJobSource)

    incomplete = DummyIncompleteSource()
    assert not isinstance(incomplete, IJobSource)


def test_ibookmarkable_runtime_checkable():
    src = DummyBookmarkableSource()
    assert isinstance(src, IBookmarkable)

    non_bookmarkable = DummyFullJobSource()
    assert not isinstance(non_bookmarkable, IBookmarkable)


def test_iauthenticated_source_runtime_checkable():
    src = DummyAuthenticatedSource()
    assert isinstance(src, IAuthenticatedSource)

    unauth = DummyFullJobSource()
    assert not isinstance(unauth, IAuthenticatedSource)


def test_iconfigurable_source_runtime_checkable():
    src = DummyConfigurableSource()
    assert isinstance(src, IConfigurableSource)

    non_conf = DummyFullJobSource()
    assert not isinstance(non_conf, IConfigurableSource)


def test_hybrid_source_satisfies_multiple_protocols():
    hybrid = DummyHybridEnterpriseSource()
    assert isinstance(hybrid, IJobSource)
    assert isinstance(hybrid, IBookmarkable)
    assert isinstance(hybrid, IAuthenticatedSource)
    assert isinstance(hybrid, IConfigurableSource)
    assert hybrid.category == SourceCategory.ENTERPRISE


def test_package_exports_contracts():
    import job_mcp.sources as sources_pkg

    assert hasattr(sources_pkg, "SourceCategory")
    assert hasattr(sources_pkg, "IJobSource")
    assert hasattr(sources_pkg, "IBookmarkable")
    assert hasattr(sources_pkg, "IAuthenticatedSource")
    assert hasattr(sources_pkg, "IConfigurableSource")


@pytest.mark.asyncio
async def test_ijobsource_method_invocation():
    src: IJobSource = DummyFullJobSource()
    jobs = await src.fetch_jobs(limit=10)
    assert jobs == []

    healthy = await src.check_health()
    assert healthy is True

    meta = src.get_metadata()
    assert meta.source_id == "dummy_full"

