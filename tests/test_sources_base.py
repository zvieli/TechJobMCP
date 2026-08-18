"""Unit tests for BaseJobSource, SourceMetadata, SourceRegistry, and HireMeTechSource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode
from hireme_mcp.sources import (
    BaseJobSource,
    HireMeTechSource,
    SourceMetadata,
    SourceRegistry,
)


# Concrete dummy source for testing base class
class DummyJobSource(BaseJobSource):
    source_id = "dummy_source"
    display_name = "Dummy Source"
    description = "A dummy source for testing"
    supports_bookmarks = True
    supports_auto_apply = False

    def __init__(self, is_authenticated: bool = True, jobs: list[Job] | None = None) -> None:
        self._auth = is_authenticated
        self._jobs = jobs or []

    @property
    def is_authenticated(self) -> bool:
        return self._auth

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        return self._jobs[:limit]

    async def check_health(self) -> bool:
        return self._auth


class UnimplementedBookmarkSource(BaseJobSource):
    source_id = "no_bm_impl"
    display_name = "No BM Impl"
    supports_bookmarks = True

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        return []

    async def check_health(self) -> bool:
        return True


class NoBookmarkSource(BaseJobSource):
    source_id = "no_bm"
    display_name = "No BM"
    supports_bookmarks = False

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        return []

    async def check_health(self) -> bool:
        return True


class TestSourceMetadata:
    """Tests for SourceMetadata Pydantic model."""

    def test_source_metadata_defaults(self) -> None:
        meta = SourceMetadata(
            source_id="test_src",
            display_name="Test Source",
        )
        assert meta.source_id == "test_src"
        assert meta.display_name == "Test Source"
        assert meta.description == ""
        assert meta.is_authenticated is False
        assert meta.supports_bookmarks is False
        assert meta.supports_auto_apply is False

    def test_source_metadata_full(self) -> None:
        meta = SourceMetadata(
            source_id="hiremetech",
            display_name="HireMeTech",
            description="AI matching platform",
            is_authenticated=True,
            supports_bookmarks=True,
            supports_auto_apply=True,
        )
        assert meta.source_id == "hiremetech"
        assert meta.is_authenticated is True
        assert meta.supports_bookmarks is True
        assert meta.supports_auto_apply is True
        data = meta.model_dump()
        assert data["source_id"] == "hiremetech"
        assert data["supports_bookmarks"] is True


class TestBaseJobSource:
    """Tests for BaseJobSource abstract class."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        with pytest.raises(TypeError):
            BaseJobSource()  # type: ignore[abstract]

    def test_concrete_subclass_get_metadata(self) -> None:
        dummy = DummyJobSource(is_authenticated=True)
        meta = dummy.get_metadata()
        assert isinstance(meta, SourceMetadata)
        assert meta.source_id == "dummy_source"
        assert meta.display_name == "Dummy Source"
        assert meta.description == "A dummy source for testing"
        assert meta.is_authenticated is True
        assert meta.supports_bookmarks is True
        assert meta.supports_auto_apply is False

    @pytest.mark.asyncio
    async def test_bookmark_job_when_unsupported_returns_false(self) -> None:
        source = NoBookmarkSource()
        res = await source.bookmark_job("job-123")
        assert res is False

    @pytest.mark.asyncio
    async def test_bookmark_job_when_supported_but_not_implemented_raises(self) -> None:
        source = UnimplementedBookmarkSource()
        with pytest.raises(NotImplementedError):
            await source.bookmark_job("job-123")


class TestSourceRegistry:
    """Tests for SourceRegistry."""

    def test_register_and_get(self) -> None:
        registry = SourceRegistry()
        src1 = DummyJobSource()
        registry.register(src1)

        assert registry.get("dummy_source") is src1
        assert registry.get("non_existent") is None
        assert "dummy_source" in registry
        assert len(registry) == 1

    def test_register_invalid_type_raises(self) -> None:
        registry = SourceRegistry()
        with pytest.raises(TypeError):
            registry.register("not_a_source")  # type: ignore[arg-type]

    def test_register_empty_id_raises(self) -> None:
        registry = SourceRegistry()

        class EmptyIdSource(BaseJobSource):
            source_id = ""
            display_name = "Empty"

            async def fetch_jobs(self, preferences=None, limit=50):
                return []

            async def check_health(self):
                return True

        with pytest.raises(ValueError):
            registry.register(EmptyIdSource())

    def test_list_sources(self) -> None:
        registry = SourceRegistry()
        src1 = DummyJobSource(is_authenticated=True)
        src2 = NoBookmarkSource()
        registry.register(src1)
        registry.register(src2)

        metas = registry.list_sources()
        assert len(metas) == 2
        ids = [m.source_id for m in metas]
        assert "dummy_source" in ids
        assert "no_bm" in ids

    def test_get_all(self) -> None:
        registry = SourceRegistry()
        src1 = DummyJobSource()
        src2 = NoBookmarkSource()
        registry.register(src1)
        registry.register(src2)

        all_sources = registry.get_all()
        assert len(all_sources) == 2
        assert src1 in all_sources
        assert src2 in all_sources

    def test_get_active_filtered(self) -> None:
        registry = SourceRegistry()
        src1 = DummyJobSource()
        src2 = NoBookmarkSource()
        registry.register(src1)
        registry.register(src2)

        active = registry.get_active(["dummy_source", "non_existent"])
        assert active == [src1]

        all_active = registry.get_active(None)
        assert len(all_active) == 2

    def test_unregister_and_clear(self) -> None:
        registry = SourceRegistry()
        src1 = DummyJobSource()
        registry.register(src1)
        assert len(registry) == 1

        unreg = registry.unregister("dummy_source")
        assert unreg is src1
        assert len(registry) == 0

        registry.register(src1)
        registry.clear()
        assert len(registry) == 0


class TestHireMeTechSource:
    """Tests for HireMeTechSource implementation."""

    def test_source_attributes(self) -> None:
        source = HireMeTechSource()
        assert source.source_id == "hiremetech"
        assert source.display_name == "HireMeTech"
        assert source.supports_bookmarks is True
        assert source.supports_auto_apply is True
        meta = source.get_metadata()
        assert meta.source_id == "hiremetech"
        assert meta.supports_bookmarks is True
        assert meta.supports_auto_apply is True

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        mock_sm = MagicMock()
        mock_sm.check_session_health = AsyncMock(return_value=True)
        source = HireMeTechSource(session_manager=mock_sm)

        is_healthy = await source.check_health()
        assert is_healthy is True
        assert source.is_authenticated is True
        mock_sm.check_session_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_health_unhealthy_or_exception(self) -> None:
        mock_sm = MagicMock()
        mock_sm.check_session_health = AsyncMock(return_value=False)
        source = HireMeTechSource(session_manager=mock_sm)

        assert await source.check_health() is False
        assert source.is_authenticated is False

        mock_sm.check_session_health = AsyncMock(side_effect=RuntimeError("Connection lost"))
        assert await source.check_health() is False
        assert source.is_authenticated is False

    @pytest.mark.asyncio
    async def test_bookmark_job_delegation(self) -> None:
        mock_sm = MagicMock()
        mock_page = MagicMock()
        mock_sm.get_page = AsyncMock(return_value=mock_page)
        source = HireMeTechSource(session_manager=mock_sm)

        with patch("hireme_mcp.sources.hiremetech.browser_bookmark_job", new_callable=AsyncMock) as mock_bm:
            mock_bm.return_value = True
            result = await source.bookmark_job("job-abc")
            assert result is True
            mock_bm.assert_awaited_once_with(mock_page, "job-abc")

    @pytest.mark.asyncio
    async def test_fetch_jobs_via_api_primary(self) -> None:
        mock_sm = MagicMock()
        mock_page = MagicMock()
        mock_sm.get_page = AsyncMock(return_value=mock_page)
        source = HireMeTechSource(session_manager=mock_sm)

        raw_jobs = [
            Job(
                job_id="job-1",
                title="Backend Developer",
                company="Wix",
                tech_stack=["Python", "FastAPI"],
                work_mode=WorkMode.REMOTE,
            ),
            Job(
                job_id="job-2",
                title="Frontend Developer",
                company="Monday",
                tech_stack=["React", "TypeScript"],
                work_mode=WorkMode.HYBRID,
            ),
        ]

        with patch("hireme_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = raw_jobs
            jobs = await source.fetch_jobs(limit=10)
            assert len(jobs) == 2
            assert jobs[0].job_id == "job-1"
            assert jobs[0].source == "hiremetech"
            assert jobs[0].sources == ["hiremetech"]
            assert jobs[1].job_id == "job-2"
            mock_api.assert_awaited_once_with(mock_page.request, size=10)

    @pytest.mark.asyncio
    async def test_fetch_jobs_fallback_to_dom_extraction(self) -> None:
        mock_sm = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://hiremetech.com/other"
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_sm.get_page = AsyncMock(return_value=mock_page)
        source = HireMeTechSource(session_manager=mock_sm)

        dom_jobs = [
            Job(
                job_id="job-dom-1",
                title="Full Stack Engineer",
                company="Fiverr",
                tech_stack=["Python", "React"],
            ),
        ]

        with patch("hireme_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api, \
             patch("hireme_mcp.sources.hiremetech.browser_extract_jobs", new_callable=AsyncMock) as mock_dom:
            mock_api.side_effect = RuntimeError("API unavailable")
            mock_dom.return_value = dom_jobs

            jobs = await source.fetch_jobs(limit=5)
            assert len(jobs) == 1
            assert jobs[0].job_id == "job-dom-1"
            assert jobs[0].source == "hiremetech"
            assert jobs[0].sources == ["hiremetech"]
            mock_page.goto.assert_awaited_once()
            mock_dom.assert_awaited_once_with(mock_page)

    @pytest.mark.asyncio
    async def test_fetch_jobs_with_preferences_filtering(self) -> None:
        mock_sm = MagicMock()
        mock_page = MagicMock()
        mock_sm.get_page = AsyncMock(return_value=mock_page)
        source = HireMeTechSource(session_manager=mock_sm)

        raw_jobs = [
            Job(
                job_id="job-py",
                title="Python Engineer",
                company="Acme",
                tech_stack=["Python"],
                work_mode=WorkMode.REMOTE,
            ),
            Job(
                job_id="job-java",
                title="Java Developer",
                company="Beta",
                tech_stack=["Java"],
                work_mode=WorkMode.ONSITE,
            ),
        ]

        with patch("hireme_mcp.sources.hiremetech.fetch_jobs_via_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = raw_jobs
            prefs = JobPreferences(tech_stack=["Python"], work_mode=WorkMode.REMOTE)
            filtered = await source.fetch_jobs(preferences=prefs, limit=10)
            assert len(filtered) == 1
            assert filtered[0].job_id == "job-py"
