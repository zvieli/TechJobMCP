"""Modular SourceRegistry and open/closed provider registration."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from typing import Any, Callable, Iterator, Optional

from job_mcp.sources.base import BaseJobSource, SourceMetadata
from job_mcp.sources.contracts import (
    IAuthenticatedSource,
    IBookmarkable,
    IJobSource,
    SourceCategory,
)


@dataclass
class SourceProvider:
    """Descriptor representing a registered job source factory or class."""

    name: str
    factory: Callable[..., IJobSource]
    default_enabled: bool = True
    env_var: Optional[str] = None
    category: Optional[SourceCategory] = None


_SOURCE_PROVIDERS: dict[str, SourceProvider] = {}


def _is_enabled(flag_name: Optional[str], explicit: Optional[bool], default_enabled: bool = True) -> bool:
    """Evaluate whether a provider should be enabled based on explicit arg or env var."""
    if explicit is not None:
        return bool(explicit)
    if not flag_name:
        return default_enabled
    default_str = "true" if default_enabled else "false"
    return os.getenv(flag_name, default_str).strip().lower() in ("true", "1", "yes")


def _instantiate_provider(
    provider: SourceProvider,
    session_manager: Optional[Any] = None,
    **kwargs: Any,
) -> IJobSource:
    """Instantiate a source provider passing relevant kwargs."""
    factory = provider.factory
    sig = inspect.signature(factory)
    call_kwargs: dict[str, Any] = {}
    if "session_manager" in sig.parameters:
        call_kwargs["session_manager"] = session_manager
    for k, v in kwargs.items():
        if k in sig.parameters:
            call_kwargs[k] = v
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        if session_manager is not None:
            call_kwargs["session_manager"] = session_manager
        call_kwargs.update(kwargs)
    return factory(**call_kwargs)


def register_provider(
    name: Any = None,
    factory_or_cls: Optional[Callable[..., IJobSource]] = None,
    default_enabled: bool = True,
    env_var: Optional[str] = None,
    category: Optional[SourceCategory] = None,
    *,
    name_or_cls: Any = None,
) -> SourceProvider:
    """Register a source provider imperatively.

    Supports:
        register_provider("name", FactoryClass, default_enabled=True, ...)
        register_provider(name="name", factory_or_cls=FactoryClass, ...)
        register_provider(SourceClass, default_enabled=True, ...)
    """
    target_name = name if name is not None else name_or_cls
    if isinstance(target_name, str) and factory_or_cls is not None:
        p_name = target_name
        factory = factory_or_cls
    elif callable(target_name) and factory_or_cls is None:
        factory = target_name
        p_name = getattr(factory, "source_id", getattr(factory, "__name__", "unknown"))
    elif isinstance(target_name, str) and factory_or_cls is None:
        raise ValueError("Must provide factory_or_cls when name is string")
    elif target_name is not None:
        p_name = getattr(target_name, "source_id", getattr(target_name, "__name__", str(target_name)))
        factory = factory_or_cls or target_name
    else:
        raise ValueError("Must provide name or provider class/factory")

    provider = SourceProvider(
        name=p_name,
        factory=factory,
        default_enabled=default_enabled,
        env_var=env_var,
        category=category,
    )
    _SOURCE_PROVIDERS[p_name] = provider
    return provider


def unregister_provider(name: str) -> Optional[SourceProvider]:
    """Remove a registered source provider by name."""
    return _SOURCE_PROVIDERS.pop(name, None)


def get_registered_providers() -> dict[str, SourceProvider]:
    """Get a copy of all registered source providers."""
    return dict(_SOURCE_PROVIDERS)


def clear_providers() -> None:
    """Clear all registered source providers."""
    _SOURCE_PROVIDERS.clear()


def register_source_provider(
    name: Optional[Any] = None,
    default_enabled: bool = True,
    env_var: Optional[str] = None,
    category: Optional[SourceCategory] = None,
    *,
    name_or_cls: Optional[Any] = None,
) -> Any:
    """Decorator to register a source provider class or factory function."""
    target = name if name is not None else name_or_cls
    if callable(target) and not isinstance(target, str):
        cls_or_fn = target
        p_name = getattr(cls_or_fn, "source_id", cls_or_fn.__name__)
        register_provider(
            name=p_name,
            factory_or_cls=cls_or_fn,
            default_enabled=default_enabled,
            env_var=env_var,
            category=category,
        )
        return cls_or_fn

    def decorator(cls_or_fn: Callable[..., IJobSource]) -> Callable[..., IJobSource]:
        p_name = target or getattr(cls_or_fn, "source_id", cls_or_fn.__name__)
        register_provider(
            name=p_name,
            factory_or_cls=cls_or_fn,
            default_enabled=default_enabled,
            env_var=env_var,
            category=category,
        )
        return cls_or_fn

    return decorator


class SourceRegistry:
    """Registry holding registered IJobSource/BaseJobSource instances keyed by source_id."""

    def __init__(self) -> None:
        """Initialize an empty SourceRegistry."""
        self._sources: dict[str, IJobSource] = {}
        self._by_category: dict[SourceCategory, list[str]] = {cat: [] for cat in SourceCategory}
        self._enabled: dict[str, bool] = {}

    def register(self, source: IJobSource) -> None:
        """Register a job source instance.

        Args:
            source: IJobSource or BaseJobSource instance to register.

        Raises:
            TypeError: If source is not an instance of IJobSource or BaseJobSource.
            ValueError: If source has an empty source_id.
        """
        if not isinstance(source, (IJobSource, BaseJobSource)):
            raise TypeError(f"Expected IJobSource or BaseJobSource instance, got {type(source)}")
        if not getattr(source, "source_id", None) or not str(source.source_id).strip():
            raise ValueError("Source must have a non-empty source_id")

        sid = str(source.source_id)

        # If already registered, remove old category mapping
        if sid in self._sources:
            for cat_list in self._by_category.values():
                if sid in cat_list:
                    cat_list.remove(sid)

        self._sources[sid] = source

        # Index by category
        raw_cat = getattr(source, "category", SourceCategory.PUBLIC)
        if isinstance(raw_cat, str):
            try:
                cat_enum = SourceCategory(raw_cat)
            except ValueError:
                cat_enum = SourceCategory.PUBLIC
        else:
            cat_enum = raw_cat

        if cat_enum not in self._by_category:
            self._by_category[cat_enum] = []
        if sid not in self._by_category[cat_enum]:
            self._by_category[cat_enum].append(sid)

        if sid not in self._enabled:
            self._enabled[sid] = True

    def unregister(self, source_id: str) -> Optional[IJobSource]:
        """Unregister a job source by ID.

        Args:
            source_id: Unique string identifier of the source.

        Returns:
            Optional[IJobSource]: Removed source instance if found, else None.
        """
        source = self._sources.pop(source_id, None)
        if source is not None:
            for cat_list in self._by_category.values():
                if source_id in cat_list:
                    cat_list.remove(source_id)
            self._enabled.pop(source_id, None)
        return source

    def get(self, source_id: str) -> Optional[IJobSource]:
        """Get a registered job source by ID."""
        return self._sources.get(source_id)

    def list_sources(self) -> list[SourceMetadata]:
        """Return metadata descriptors for all registered sources."""
        return [source.get_metadata() for source in self._sources.values()]

    def get_all(self) -> list[IJobSource]:
        """Return all registered job source instances."""
        return list(self._sources.values())

    def list_all(self) -> list[IJobSource]:
        """Alias for get_all() returning all registered sources."""
        return list(self._sources.values())

    def get_active(self, source_ids: Optional[list[str]] = None) -> list[IJobSource]:
        """Return active sources, optionally filtered by source_ids."""
        if source_ids is None:
            return list(self._sources.values())
        return [self._sources[sid] for sid in source_ids if sid in self._sources]

    def clear(self) -> None:
        """Clear all registered sources and indexes."""
        self._sources.clear()
        self._enabled.clear()
        self._by_category = {cat: [] for cat in SourceCategory}

    def enable(self, source_id: str) -> None:
        """Mark a registered source as enabled."""
        if source_id not in self._sources:
            raise KeyError(f"Source '{source_id}' is not registered")
        self._enabled[source_id] = True

    def disable(self, source_id: str) -> None:
        """Mark a registered source as disabled."""
        if source_id not in self._sources:
            raise KeyError(f"Source '{source_id}' is not registered")
        self._enabled[source_id] = False

    def is_enabled(self, source_id: str) -> bool:
        """Check whether a registered source is enabled."""
        if source_id not in self._sources:
            return False
        return self._enabled.get(source_id, True)

    def get_enabled(self) -> list[IJobSource]:
        """Return all enabled job source instances."""
        return [s for s in self._sources.values() if self.is_enabled(str(s.source_id))]

    def get_by_category(
        self,
        category: SourceCategory | str,
        enabled_only: bool = False,
    ) -> list[IJobSource]:
        """Get sources belonging to a specific SourceCategory."""
        if isinstance(category, str):
            try:
                cat_enum = SourceCategory(category)
            except ValueError:
                return []
        else:
            cat_enum = category

        source_ids = self._by_category.get(cat_enum, [])
        sources = [self._sources[sid] for sid in source_ids if sid in self._sources]
        if enabled_only:
            sources = [s for s in sources if self.is_enabled(str(s.source_id))]
        return sources

    def get_public_sources(self, enabled_only: bool = False) -> list[IJobSource]:
        """Get all registered public job sources."""
        return self.get_by_category(SourceCategory.PUBLIC, enabled_only=enabled_only)

    def get_enterprise_sources(self, enabled_only: bool = False) -> list[IJobSource]:
        """Get all registered enterprise job sources."""
        return self.get_by_category(SourceCategory.ENTERPRISE, enabled_only=enabled_only)

    def get_authenticated_sources(self, enabled_only: bool = False) -> list[IJobSource]:
        """Get all registered authenticated job sources."""
        sources = [
            s
            for s in self._sources.values()
            if getattr(s, "category", None) == SourceCategory.AUTHENTICATED
            or isinstance(s, IAuthenticatedSource)
        ]
        if enabled_only:
            sources = [s for s in sources if self.is_enabled(str(s.source_id))]
        return sources

    def get_bookmarkable_sources(self, enabled_only: bool = False) -> list[IJobSource]:
        """Get all sources that implement the IBookmarkable interface."""
        sources = [s for s in self._sources.values() if isinstance(s, IBookmarkable)]
        if enabled_only:
            sources = [s for s in sources if self.is_enabled(str(s.source_id))]
        return sources

    def populate_from_providers(
        self,
        session_manager: Optional[Any] = None,
        overrides: Optional[dict[str, Optional[bool]]] = None,
        **kwargs: Any,
    ) -> None:
        """Populate registry dynamically using registered source providers."""
        overrides = overrides or {}
        for name, provider in _SOURCE_PROVIDERS.items():
            explicit = overrides.get(name)
            if explicit is None:
                explicit = kwargs.get(f"enable_{name}")
            if explicit is None:
                explicit = kwargs.get(f"enable_{name.lower()}")
            if explicit is None:
                explicit = kwargs.get(name)

            if _is_enabled(provider.env_var, explicit, default_enabled=provider.default_enabled):
                source = _instantiate_provider(
                    provider,
                    session_manager=session_manager,
                    **kwargs,
                )
                self.register(source)

    def __contains__(self, item: Any) -> bool:
        """Check if source_id or source instance is in registry."""
        if isinstance(item, str):
            return item in self._sources
        return item in self._sources.values()

    def __iter__(self) -> Iterator[IJobSource]:
        """Iterate over registered source instances."""
        return iter(self._sources.values())

    def __len__(self) -> int:
        """Return count of registered sources."""
        return len(self._sources)


def reset_builtin_providers() -> None:
    """Reset provider registry and re-register standard built-in providers."""
    clear_providers()

    from job_mcp.sources.authenticated.hiremetech import HireMeTechSource
    from job_mcp.sources.authenticated.linkedin import LinkedInSource
    from job_mcp.sources.enterprise.direct_tech import DirectTechSource
    from job_mcp.sources.enterprise.workday import WorkdaySource
    from job_mcp.sources.public.alljobs import AllJobsSource
    from job_mcp.sources.public.comeet import ComeetSource
    from job_mcp.sources.public.eightfold import EightfoldAISource
    from job_mcp.sources.public.greenhouse import GreenhouseSource
    from job_mcp.sources.public.jobify import JobifySource

    register_provider(
        name="hiremetech",
        factory_or_cls=HireMeTechSource,
        default_enabled=True,
        env_var="ENABLE_HIREMETECH",
        category=SourceCategory.AUTHENTICATED,
    )
    register_provider(
        name="comeet",
        factory_or_cls=ComeetSource,
        default_enabled=True,
        env_var="ENABLE_COMEET",
        category=SourceCategory.PUBLIC,
    )
    register_provider(
        name="alljobs",
        factory_or_cls=AllJobsSource,
        default_enabled=False,
        env_var="ENABLE_ALLJOBS",
        category=SourceCategory.PUBLIC,
    )
    register_provider(
        name="workday",
        factory_or_cls=WorkdaySource,
        default_enabled=True,
        env_var="ENABLE_WORKDAY",
        category=SourceCategory.ENTERPRISE,
    )
    register_provider(
        name="eightfold",
        factory_or_cls=EightfoldAISource,
        default_enabled=True,
        env_var="ENABLE_EIGHTFOLD",
        category=SourceCategory.PUBLIC,
    )
    register_provider(
        name="direct_tech",
        factory_or_cls=DirectTechSource,
        default_enabled=True,
        env_var="ENABLE_DIRECT_TECH",
        category=SourceCategory.ENTERPRISE,
    )
    register_provider(
        name="linkedin",
        factory_or_cls=LinkedInSource,
        default_enabled=True,
        env_var="ENABLE_LINKEDIN",
        category=SourceCategory.AUTHENTICATED,
    )
    register_provider(
        name="jobify",
        factory_or_cls=JobifySource,
        default_enabled=True,
        env_var="ENABLE_JOBIFY",
        category=SourceCategory.PUBLIC,
    )
    register_provider(
        name="greenhouse",
        factory_or_cls=GreenhouseSource,
        default_enabled=True,
        env_var="ENABLE_GREENHOUSE",
        category=SourceCategory.PUBLIC,
    )


# Register built-ins on module load
reset_builtin_providers()


def create_default_registry(
    session_manager: Optional[Any] = None,
    enable_alljobs: Optional[bool] = None,
    enable_workday: Optional[bool] = None,
    enable_eightfold: Optional[bool] = None,
    enable_direct_tech: Optional[bool] = None,
    enable_linkedin: Optional[bool] = None,
    enable_jobify: Optional[bool] = None,
    enable_greenhouse: Optional[bool] = None,
    enable_hiremetech: Optional[bool] = None,
    enable_comeet: Optional[bool] = None,
    **kwargs: Any,
) -> SourceRegistry:
    """Create and return a SourceRegistry pre-populated with standard job sources.

    By default, all enterprise and authenticated sources (HireMeTech, Comeet, Workday,
    Eightfold AI, Direct Tech, LinkedIn, Jobify, and Greenhouse) are enabled out-of-the-box.
    AllJobs is disabled by default and can be enabled via ENABLE_ALLJOBS=true.
    """
    reg = SourceRegistry()
    explicit_map: dict[str, Optional[bool]] = {
        "hiremetech": enable_hiremetech,
        "comeet": enable_comeet,
        "alljobs": enable_alljobs,
        "workday": enable_workday,
        "eightfold": enable_eightfold,
        "direct_tech": enable_direct_tech,
        "linkedin": enable_linkedin,
        "jobify": enable_jobify,
        "greenhouse": enable_greenhouse,
    }
    reg.populate_from_providers(
        session_manager=session_manager,
        overrides=explicit_map,
        **kwargs,
    )
    return reg


registry: SourceRegistry = create_default_registry()

__all__ = [
    "SourceProvider",
    "SourceRegistry",
    "register_provider",
    "unregister_provider",
    "get_registered_providers",
    "clear_providers",
    "register_source_provider",
    "reset_builtin_providers",
    "create_default_registry",
    "registry",
]
