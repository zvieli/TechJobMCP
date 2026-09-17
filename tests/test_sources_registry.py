"""Tests for upgraded SourceRegistry and open/closed provider registration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from job_mcp.models.schemas import Job, JobPreferences
from job_mcp.sources.base import (
    BaseAuthenticatedSource,
    BaseEnterpriseSource,
    BaseJobSource,
    BasePublicSource,
    SourceMetadata,
)
from job_mcp.sources.contracts import (
    IAuthenticatedSource,
    IBookmarkable,
    IJobSource,
    SourceCategory,
)
from job_mcp.sources.registry import (
    SourceProvider,
    SourceRegistry,
    clear_providers,
    create_default_registry,
    get_registered_providers,
    register_provider,
    register_source_provider,
    reset_builtin_providers,
    unregister_provider,
)


class MockPublicSource(BasePublicSource):
    source_id = "mock_public"
    display_name = "Mock Public Source"

    async def fetch_jobs(self, preferences=None, limit=50):
        return []

    async def check_health(self):
        return True


class MockEnterpriseSource(BaseEnterpriseSource):
    source_id = "mock_enterprise"
    display_name = "Mock Enterprise Source"

    async def fetch_jobs(self, preferences=None, limit=50):
        return []

    async def check_health(self):
        return True


class MockAuthenticatedSource(BaseAuthenticatedSource):
    source_id = "mock_authenticated"
    display_name = "Mock Authenticated Source"

    async def fetch_jobs(self, preferences=None, limit=50):
        return []

    async def check_health(self):
        return True


class MockBookmarkableSource(BaseJobSource):
    source_id = "mock_bookmarkable"
    display_name = "Mock Bookmarkable Source"
    category = SourceCategory.PUBLIC
    supports_bookmarks = True

    async def fetch_jobs(self, preferences=None, limit=50):
        return []

    async def check_health(self):
        return True

    async def bookmark_job(self, job_id: str) -> bool:
        return True


class TestSourceRegistryBasic:
    """Test standard registry operations."""

    def test_register_and_get(self) -> None:
        reg = SourceRegistry()
        src = MockPublicSource()
        reg.register(src)

        assert reg.get("mock_public") is src
        assert reg.get("non_existent") is None
        assert "mock_public" in reg
        assert src in reg
        assert len(reg) == 1

    def test_register_invalid_type_raises(self) -> None:
        reg = SourceRegistry()
        with pytest.raises(TypeError):
            reg.register("invalid")  # type: ignore[arg-type]

    def test_register_empty_id_raises(self) -> None:
        reg = SourceRegistry()

        class EmptyIdSource(BaseJobSource):
            source_id = ""
            display_name = "Empty"

            async def fetch_jobs(self, preferences=None, limit=50):
                return []

            async def check_health(self):
                return True

        with pytest.raises(ValueError):
            reg.register(EmptyIdSource())

    def test_unregister_and_clear(self) -> None:
        reg = SourceRegistry()
        src = MockPublicSource()
        reg.register(src)
        assert len(reg) == 1

        unreg = reg.unregister("mock_public")
        assert unreg is src
        assert len(reg) == 0
        assert reg.unregister("mock_public") is None

        reg.register(src)
        reg.clear()
        assert len(reg) == 0

    def test_list_and_get_all(self) -> None:
        reg = SourceRegistry()
        src1 = MockPublicSource()
        src2 = MockEnterpriseSource()
        reg.register(src1)
        reg.register(src2)

        metas = reg.list_sources()
        assert len(metas) == 2
        ids = {m.source_id for m in metas}
        assert ids == {"mock_public", "mock_enterprise"}

        all_sources = reg.get_all()
        assert set(all_sources) == {src1, src2}

        list_sources = reg.list_all()
        assert set(list_sources) == {src1, src2}

    def test_iteration(self) -> None:
        reg = SourceRegistry()
        src1 = MockPublicSource()
        src2 = MockEnterpriseSource()
        reg.register(src1)
        reg.register(src2)

        yielded = list(reg)
        assert set(yielded) == {src1, src2}

    def test_get_active(self) -> None:
        reg = SourceRegistry()
        src1 = MockPublicSource()
        src2 = MockEnterpriseSource()
        reg.register(src1)
        reg.register(src2)

        active = reg.get_active(["mock_public", "unknown"])
        assert active == [src1]

        all_active = reg.get_active(None)
        assert len(all_active) == 2


class TestSourceRegistryEnablement:
    """Test enablement toggles in registry."""

    def test_default_enabled(self) -> None:
        reg = SourceRegistry()
        src = MockPublicSource()
        reg.register(src)

        assert reg.is_enabled("mock_public") is True
        assert reg.get_enabled() == [src]

    def test_enable_disable(self) -> None:
        reg = SourceRegistry()
        src1 = MockPublicSource()
        src2 = MockEnterpriseSource()
        reg.register(src1)
        reg.register(src2)

        reg.disable("mock_public")
        assert reg.is_enabled("mock_public") is False
        assert reg.is_enabled("mock_enterprise") is True
        assert reg.get_enabled() == [src2]

        reg.enable("mock_public")
        assert reg.is_enabled("mock_public") is True
        assert set(reg.get_enabled()) == {src1, src2}

    def test_enable_disable_unknown_raises(self) -> None:
        reg = SourceRegistry()
        with pytest.raises(KeyError):
            reg.enable("unknown")
        with pytest.raises(KeyError):
            reg.disable("unknown")
        assert reg.is_enabled("unknown") is False


class TestCategoryAndCapabilityQueries:
    """Test querying by category and protocols."""

    def test_category_queries(self) -> None:
        reg = SourceRegistry()
        pub = MockPublicSource()
        ent = MockEnterpriseSource()
        auth = MockAuthenticatedSource()

        reg.register(pub)
        reg.register(ent)
        reg.register(auth)

        # By category enum
        assert reg.get_by_category(SourceCategory.PUBLIC) == [pub]
        assert reg.get_by_category(SourceCategory.ENTERPRISE) == [ent]
        assert reg.get_by_category(SourceCategory.AUTHENTICATED) == [auth]

        # By string
        assert reg.get_by_category("public") == [pub]
        assert reg.get_by_category("enterprise") == [ent]
        assert reg.get_by_category("authenticated") == [auth]

        # Helper methods
        assert reg.get_public_sources() == [pub]
        assert reg.get_enterprise_sources() == [ent]
        assert reg.get_authenticated_sources() == [auth]

    def test_category_queries_with_enabled_only(self) -> None:
        reg = SourceRegistry()
        pub1 = MockPublicSource()
        reg.register(pub1)

        assert reg.get_public_sources(enabled_only=True) == [pub1]
        reg.disable("mock_public")
        assert reg.get_public_sources(enabled_only=True) == []
        assert reg.get_public_sources(enabled_only=False) == [pub1]

    def test_bookmarkable_sources_query(self) -> None:
        reg = SourceRegistry()
        pub = MockPublicSource()
        bm = MockBookmarkableSource()

        reg.register(pub)
        reg.register(bm)

        bm_sources = reg.get_bookmarkable_sources()
        assert bm_sources == [bm]

        # Disabled filtering
        reg.disable("mock_bookmarkable")
        assert reg.get_bookmarkable_sources(enabled_only=True) == []
        assert reg.get_bookmarkable_sources(enabled_only=False) == [bm]

    def test_re_registration_updates_category_index(self) -> None:
        reg = SourceRegistry()
        src = MockPublicSource()
        reg.register(src)
        assert reg.get_public_sources() == [src]

        # Overwrite with same id but different category
        class SwappedSource(BaseEnterpriseSource):
            source_id = "mock_public"
            display_name = "Swapped"

            async def fetch_jobs(self, preferences=None, limit=50):
                return []

            async def check_health(self):
                return True

        swapped = SwappedSource()
        reg.register(swapped)
        assert reg.get_public_sources() == []
        assert reg.get_enterprise_sources() == [swapped]

        # Unregistering removes from category index
        reg.unregister("mock_public")
        assert reg.get_enterprise_sources() == []


class TestProviderRegistration:
    """Test open/closed provider registry mechanism."""

    @pytest.fixture(autouse=True)
    def clean_providers(self):
        # Backup and restore providers around each test
        yield
        reset_builtin_providers()

    def test_register_provider_imperative(self) -> None:
        provider = register_provider(
            name="custom_source",
            factory_or_cls=MockPublicSource,
            default_enabled=True,
            env_var="ENABLE_CUSTOM_SOURCE",
            category=SourceCategory.PUBLIC,
        )
        assert isinstance(provider, SourceProvider)
        assert provider.name == "custom_source"
        providers = get_registered_providers()
        assert "custom_source" in providers

        # Cleanup
        unregister_provider("custom_source")
        assert "custom_source" not in get_registered_providers()

    def test_register_source_provider_decorator(self) -> None:
        @register_source_provider(
            name="decorated_src",
            default_enabled=False,
            env_var="ENABLE_DECORATED",
            category=SourceCategory.ENTERPRISE,
        )
        class DecoratedSource(BaseEnterpriseSource):
            source_id = "decorated_src"
            display_name = "Decorated"

            async def fetch_jobs(self, preferences=None, limit=50):
                return []

            async def check_health(self):
                return True

        providers = get_registered_providers()
        assert "decorated_src" in providers
        p = providers["decorated_src"]
        assert p.default_enabled is False
        assert p.env_var == "ENABLE_DECORATED"
        assert p.category == SourceCategory.ENTERPRISE

        unregister_provider("decorated_src")

    def test_populate_registry_from_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reg = SourceRegistry()
        # Default providers are registered
        reg.populate_from_providers()
        # 8 default enabled sources
        assert len(reg) == 8
        assert "alljobs" not in reg
        assert "hiremetech" in reg
        assert "greenhouse" in reg

        # Enable alljobs via env
        monkeypatch.setenv("ENABLE_ALLJOBS", "true")
        reg2 = SourceRegistry()
        reg2.populate_from_providers()
        assert len(reg2) == 9
        assert "alljobs" in reg2
        assert "greenhouse" in reg2

    def test_provider_with_session_manager(self) -> None:
        mock_sm = MagicMock()
        reg = SourceRegistry()
        reg.populate_from_providers(session_manager=mock_sm)

        hmt = reg.get("hiremetech")
        assert hmt is not None
        assert getattr(hmt, "session_manager", None) is mock_sm

        linkedin = reg.get("linkedin")
        assert linkedin is not None
        assert getattr(linkedin, "session_manager", None) is mock_sm


class TestBackwardsCompatibility:
    """Ensure existing create_default_registry behavior is 100% preserved."""

    def test_create_default_registry_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENABLE_ALLJOBS", raising=False)
        reg = create_default_registry()
        assert len(reg) == 8
        assert "hiremetech" in reg
        assert "comeet" in reg
        assert "workday" in reg
        assert "eightfold" in reg
        assert "direct_tech" in reg
        assert "linkedin" in reg
        assert "jobify" in reg
        assert "greenhouse" in reg
        assert "alljobs" not in reg

    def test_create_default_registry_explicit_args(self) -> None:
        reg = create_default_registry(
            enable_alljobs=True,
            enable_workday=False,
            enable_eightfold=False,
            enable_direct_tech=False,
            enable_linkedin=False,
            enable_jobify=False,
            enable_greenhouse=False,
            enable_hiremetech=False,
            enable_comeet=False,
        )
        assert len(reg) == 1
        assert "alljobs" in reg

    def test_package_exports_registry_symbols(self) -> None:
        import job_mcp.sources as sources_pkg

        assert hasattr(sources_pkg, "SourceRegistry")
        assert hasattr(sources_pkg, "create_default_registry")
        assert hasattr(sources_pkg, "SourceProvider")
        assert hasattr(sources_pkg, "register_provider")
        assert hasattr(sources_pkg, "register_source_provider")
