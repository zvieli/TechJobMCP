"""Configuration-driven company registry.

Milestone 5 converts the hardcoded per-provider company catalogs into a
validated, typed registry. Two distinct axes exist in this package tree and must
not be confused:

* ``job_mcp.sources.registry`` — provider *family* registry (source_id to
  instance) with ``ENABLE_*`` environment flags. Unchanged by Milestone 5.
* this package — the *company* registry (which companies a family queries).

Contract:

* Built-in curated defaults work with zero user configuration.
* YAML is parsed and validated **here and in** ``schema`` **only**; providers
  never read config files and receive normalized typed entry objects.
* Merge precedence is deterministic: built-in defaults, then configuration
  overlays applied in load order, producing one effective catalog per family.
* Reloading is explicit (:func:`configure`, :func:`reset`); implicit filesystem
  hot-reload is a non-goal.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from job_mcp.sources.company_registry import defaults as _defaults
from job_mcp.sources.company_registry.entries import (
    DirectTechCompany,
    EightfoldCompany,
    GreenhouseCompany,
    LeverCompany,
    WorkdayCompany,
)
from job_mcp.sources.company_registry.schema import (
    ENTRY_MODELS,
    MANAGED_PROVIDERS,
    OVERRIDE_MODELS,
    RegistryConfig,
)

CONFIG_PATH_ENV_VAR = "COMPANY_REGISTRY_PATH"
# Setting this to a truthy value forces the built-in curated catalogs and skips
# project/env discovery entirely. It exists so a test session can stay hermetic:
# `job_mcp.sources.registry` constructs providers at import time, so an invalid
# personal portals.yml would otherwise abort pytest collection.
BUILTINS_ONLY_ENV_VAR = "COMPANY_REGISTRY_BUILTINS_ONLY"
_TRUTHY = ("1", "true", "yes", "on")
DEFAULT_CONFIG_FILENAMES = ("portals.yml", "portals.yaml")

_BUILTIN_CATALOGS: dict[str, dict[str, Any]] = {
    "greenhouse": _defaults.GREENHOUSE_COMPANIES,
    "lever": _defaults.LEVER_COMPANIES,
    "eightfold": _defaults.EIGHTFOLD_COMPANIES,
    "direct_tech": _defaults.DIRECT_TECH_COMPANIES,
    "workday": _defaults.WORKDAY_COMPANIES,
}

_ENTRY_TYPES: dict[str, type] = {
    "greenhouse": GreenhouseCompany,
    "lever": LeverCompany,
    "eightfold": EightfoldCompany,
    "direct_tech": DirectTechCompany,
    "workday": WorkdayCompany,
}


class CompanyRegistryError(ValueError):
    """Raised for invalid company registry configuration."""

    def __init__(self, message: str, *, source: str | None = None) -> None:
        """Store an actionable message plus the offending source path."""
        self.source = source
        location = f" in {source}" if source else ""
        super().__init__(f"company registry{location}: {message}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def supported_providers() -> tuple[str, ...]:
    """Return the provider families this registry can configure."""
    return MANAGED_PROVIDERS


def discover_config_paths(explicit: str | None = None) -> list[Path]:
    """Return config files to load, in ascending precedence order.

    Resolution is single-location and deterministic:

    1. ``explicit`` (used by :func:`configure`) replaces discovery entirely;
    2. ``$COMPANY_REGISTRY_PATH`` selects exactly one file and must exist --
       pointing at a missing file is a configuration error, not a silent fall
       back to defaults;
    3. otherwise one project file in the current working directory, preferring
       ``portals.yml`` over ``portals.yaml``.

    Only one project file is selected on purpose. Loading ``portals.yaml`` in
    addition to ``portals.yml`` would merge two files that claim to describe the
    same project, and scanning the installed package's parent directories as
    well would let a less specific location be applied last and silently outrank
    the working directory.
    """
    if explicit is not None:
        return [Path(explicit)]

    if os.environ.get(BUILTINS_ONLY_ENV_VAR, "").strip().lower() in _TRUTHY:
        return []

    env_value = os.environ.get(CONFIG_PATH_ENV_VAR, "").strip()
    if env_value:
        path = Path(env_value).expanduser()
        if not path.is_file():
            raise CompanyRegistryError(
                f"{CONFIG_PATH_ENV_VAR} points at a missing file {str(path)!r}",
                source=str(path),
            )
        return [path]

    cwd = Path.cwd()
    for filename in DEFAULT_CONFIG_FILENAMES:
        candidate = cwd / filename
        if candidate.is_file():
            return [candidate]
    return []


def _parse_document(path: Path) -> dict[str, Any]:
    """Read and YAML-decode one configuration file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompanyRegistryError(f"cannot read file: {exc}", source=str(path)) from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CompanyRegistryError(f"invalid YAML: {exc}", source=str(path)) from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise CompanyRegistryError(
            f"top-level document must be a mapping, got {type(document).__name__}",
            source=str(path),
        )
    return document


def _format_validation_error(exc: ValidationError) -> str:
    """Render pydantic errors as one actionable, ordered message."""
    lines: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "<document>"
        lines.append(f"{location}: {error['msg']}")
    count = len(lines)
    plural = "s" if count != 1 else ""
    return f"configuration rejected ({count} problem{plural})\n  - " + "\n  - ".join(lines)


def validate_document(document: dict[str, Any], *, source: str | None = None) -> RegistryConfig:
    """Validate a raw mapping into a typed :class:`RegistryConfig`.

    Raises:
        CompanyRegistryError: On unknown fields, malformed identifiers, invalid
            URLs, duplicate conflicting entries, or wrong types.
    """
    try:
        return RegistryConfig.model_validate(document)
    except ValidationError as exc:
        raise CompanyRegistryError(_format_validation_error(exc), source=source) from exc


def load_config(paths: list[Path] | None = None) -> list[RegistryConfig]:
    """Load and validate every configuration document, in precedence order."""
    resolved_paths = discover_config_paths() if paths is None else [Path(p) for p in paths]
    configs: list[RegistryConfig] = []
    for path in resolved_paths:
        document = _parse_document(path)
        configs.append(validate_document(document, source=str(path)))
    return configs


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _field_dict(entry: Any, provider: str) -> dict[str, Any]:
    """Return the configurable field map of a built-in entry.

    ``id`` is a registry-side key, not a provider entry attribute, so it is
    excluded here. List fields are copied so a rebuilt entry never aliases the
    list of the entry it replaced.
    """
    fields: dict[str, Any] = {}
    for name in OVERRIDE_MODELS[provider].model_fields:
        if name == "id":
            continue
        value = getattr(entry, name)
        fields[name] = list(value) if isinstance(value, list) else value
    return fields


def _rebuild(provider: str, company_id: str, fields: dict[str, Any]) -> Any:
    """Rebuild a typed entry from a field map."""
    entry_type = _ENTRY_TYPES[provider]
    if provider == "direct_tech":
        return entry_type(provider_id=company_id, **fields)
    return entry_type(**fields)


def effective_catalog(
    configs: list[RegistryConfig],
    *,
    builtin: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge configuration onto built-in defaults deterministically.

    Rules, applied per provider family:

    * an entry whose ``id`` matches a built-in company **overrides** it; only the
      fields present in the configuration change, the rest are inherited;
    * an entry whose ``id`` is new is **appended** after the built-in block,
      preserving configuration order;
    * ``enabled: false`` disables an entry, including a built-in one, without
      deleting it, so a later overlay can re-enable it;
    * identical duplicates collapse; conflicting duplicates are rejected by
      schema validation before this point;
    * a later configuration document wins over an earlier one.

    Built-in catalogs are never mutated or aliased: every entry in the result is
    a deep copy, so mutating a resolved entry (including a nested ``locations``
    list) cannot reach ``defaults.py`` or another resolution.
    """
    source_catalogs = builtin if builtin is not None else _BUILTIN_CATALOGS
    merged: dict[str, dict[str, Any]] = {
        provider: {
            company_id: copy.deepcopy(entry)
            for company_id, entry in source_catalogs.get(provider, {}).items()
        }
        for provider in MANAGED_PROVIDERS
    }

    for config in configs:
        for provider in MANAGED_PROVIDERS:
            block = config.providers.get(provider)
            if block is None:
                continue
            override_fields = OVERRIDE_MODELS[provider].model_fields
            for company in block.companies:
                # Entries arrive typed and validated; no parsing happens here.
                # The YAML path already types each entry against its own family,
                # so this only constrains hand-built RegistryConfig objects: a
                # Lever entry placed under "greenhouse" would otherwise inject a
                # wrong-typed object into the Greenhouse catalog.
                allowed = (ENTRY_MODELS[provider], OVERRIDE_MODELS[provider])
                if not isinstance(company, allowed):
                    raise CompanyRegistryError(
                        f"{provider}: company entry {type(company).__name__} belongs to a "
                        f"different provider family; expected "
                        f"{ENTRY_MODELS[provider].__name__} or "
                        f"{OVERRIDE_MODELS[provider].__name__}"
                    )
                # exclude_unset (not exclude_none) keeps "field omitted, inherit
                # the built-in" distinct from "field explicitly null".
                payload = company.model_dump(exclude_unset=True)
                company_id = str(payload.pop("id"))
                target = merged[provider]
                if company_id in target:
                    fields = _field_dict(target[company_id], provider)
                    fields.update({k: v for k, v in payload.items() if k in override_fields})
                    target[company_id] = _rebuild(provider, company_id, fields)
                else:
                    to_entry = getattr(company, "to_entry", None)
                    if to_entry is None:
                        # Reachable only by building ProviderConfig directly with
                        # an override-shaped dict for a company that is not a
                        # built-in; the YAML path always types entries through
                        # schema.parse_company. Reported as configuration rather
                        # than an AttributeError deep inside the merge.
                        raise CompanyRegistryError(
                            f"{provider}: new company {company_id!r} must supply every "
                            f"required field for that family, not a partial override"
                        )
                    target[company_id] = to_entry()
    return merged


# ---------------------------------------------------------------------------
# Resolution cache (explicit invalidation only)
# ---------------------------------------------------------------------------


class CompanyRegistry:
    """An immutable, resolved view of effective company catalogs.

    Reads return deep copies, so a caller that mutates a returned entry (or one
    of its list fields) cannot alter this snapshot, a later provider built from
    it, or the built-in defaults. Providers treat entries as read-only data; this
    makes that contract enforced instead of assumed.
    """

    def __init__(self, catalogs: dict[str, dict[str, Any]], sources: list[Path]) -> None:
        """Store resolved per-provider catalogs and the config files applied.

        The catalogs are deep-copied so the registry owns an isolated snapshot:
        a caller that retains a reference to the original mapping (or to the
        entry objects it contains) cannot mutate this registry's state.
        """
        self._catalogs = copy.deepcopy(catalogs)
        self._sources = list(sources)

    @property
    def config_sources(self) -> list[Path]:
        """Return the configuration files that were applied."""
        return list(self._sources)

    def catalog(self, provider: str) -> dict[str, Any]:
        """Return ``{company_id: typed entry}`` for one provider family.

        The mapping and every entry are copies, so mutating the result cannot
        reach this registry snapshot or the built-in catalogs.
        """
        if provider not in self._catalogs:
            raise CompanyRegistryError(
                f"unsupported provider {provider!r}; supported: "
                f"{', '.join(MANAGED_PROVIDERS)}"
            )
        return {
            company_id: copy.deepcopy(entry) for company_id, entry in self._catalogs[provider].items()
        }

    def enabled_catalog(self, provider: str) -> dict[str, Any]:
        """Return only the entries that are not disabled."""
        return {
            company_id: entry
            for company_id, entry in self.catalog(provider).items()
            if getattr(entry, "enabled", True)
        }

    def company_ids(self, provider: str, *, enabled_only: bool = False) -> list[str]:
        """Return company identifiers in deterministic order."""
        catalog = self.enabled_catalog(provider) if enabled_only else self.catalog(provider)
        return list(catalog)

    def is_managed(self, provider: str) -> bool:
        """Return whether this registry manages the given provider family."""
        return provider in self._catalogs

    def __repr__(self) -> str:
        """Describe the resolved registry."""
        counts = {provider: len(entries) for provider, entries in self._catalogs.items()}
        return (
            f"CompanyRegistry(catalogs={counts}, "
            f"sources={[str(path) for path in self._sources]})"
        )


_ACTIVE: CompanyRegistry | None = None


def resolve(explicit_paths: list[Any] | None = None) -> CompanyRegistry:
    """Build the effective registry from defaults plus configuration.

    Always re-reads and re-validates, so it is directly usable in tests. Use
    :func:`get_registry` for the cached production path.
    """
    paths = discover_config_paths() if explicit_paths is None else [Path(p) for p in explicit_paths]
    return CompanyRegistry(effective_catalog(load_config(paths)), paths)


def get_registry() -> CompanyRegistry:
    """Return the active registry, resolving it once and caching the result."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = resolve()
    return _ACTIVE


def configure(paths: list[Any] | None = None) -> CompanyRegistry:
    """Explicitly (re)load configuration and install it as the active registry.

    This is the only supported reload path; there is no filesystem watching.

    Args:
        paths: Explicit config files, or ``None`` to re-run discovery. An empty
            list installs the built-in defaults only.
    """
    global _ACTIVE
    _ACTIVE = resolve(None if paths is None else list(paths))
    return _ACTIVE


def reset() -> None:
    """Drop the cached registry so the next access resolves from scratch."""
    global _ACTIVE
    _ACTIVE = None


def catalog(provider: str) -> dict[str, Any]:
    """Return the effective typed catalog for one provider family."""
    return get_registry().catalog(provider)


def enabled_catalog(provider: str) -> dict[str, Any]:
    """Return the effective enabled typed catalog for one provider family."""
    return get_registry().enabled_catalog(provider)


__all__ = [
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG_FILENAMES",
    "MANAGED_PROVIDERS",
    "CompanyRegistry",
    "CompanyRegistryError",
    "catalog",
    "configure",
    "discover_config_paths",
    "effective_catalog",
    "enabled_catalog",
    "get_registry",
    "load_config",
    "reset",
    "resolve",
    "supported_providers",
    "validate_document",
]
