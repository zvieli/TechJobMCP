"""Pydantic schema for the company registry configuration format.

This module owns *validation*: it converts raw mapping data into typed models
and rejects malformed input with actionable messages. Nothing here performs
network I/O or scraping.

Supported families are exactly the provider families that already expose a
``companies=`` constructor seam and carry a per-company ``enabled`` flag.
Adding a new ATS family is out of Milestone 5 scope and is rejected as an
unsupported provider key.
"""

from __future__ import annotations

import re
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from job_mcp.sources.company_registry.defaults import BUILTIN_COMPANY_IDS
from job_mcp.sources.company_registry.entries import (
    DIRECT_TECH_DEFAULT_QUERY,
    AshbyCompany,
    DirectTechCompany,
    EightfoldCompany,
    GreenhouseCompany,
    LeverCompany,
    SmartRecruitersCompany,
    WorkableCompany,
    WorkdayCompany,
)

# Provider identifiers: lowercase words separated by single hyphens/underscores.
PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
# Company keys mirror existing catalog keys ("papaya_global", "orca_security"),
# so they are looser than provider ids but still safe identifiers.
COMPANY_KEY_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")

# Identifier shapes interpolated into request URLs by the providers.
# A DNS label (used for Workday tenants, which appear in the hostname).
HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
# A single, unreserved URL path segment (Lever slug, Greenhouse board token,
# Workday site suffix). Dots are permitted for real site names, but an
# all-dots value is rejected separately below because it is path traversal.
PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A dotted lowercase host such as `nvidia.eightfold.ai`: two or more labels, each
# non-empty and alphanumeric at both ends. Refuses `.evil.com`, `evil.com.`,
# `a..b`, `-a.example`, `a-.example`, `singlelabel` and `a_b.example`.
HOST_NAME_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$"
)


def _require_http_url(value: str | None, field_name: str) -> str | None:
    """Require an absolute http(s) URL backed by a real hostname and a valid port.

    `netloc` alone is not enough: `urlsplit("https://@")` reports netloc `"@"`
    with no hostname, so the parsed hostname is checked instead. `parts.port`
    must also be read, because it raises for a non-numeric or out-of-range port
    that would otherwise only fail inside the HTTP client.
    """
    if value is None:
        return None
    stripped = value.strip()
    if any(char.isspace() for char in stripped):
        raise ValueError(f"{field_name} must not contain whitespace, got {value!r}")
    parts = urlsplit(stripped)
    hostname = parts.hostname
    if parts.scheme not in ("http", "https") or not hostname or hostname == ".":
        raise ValueError(
            f"{field_name} must be an absolute http(s) URL with a hostname, got {value!r}"
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port, got {value!r}") from exc
    # Port 0 is not a connectable TCP port.
    if port == 0:
        raise ValueError(f"{field_name} has an invalid port, got {value!r}")
    return stripped

def _require_base_url(value: str | None, field_name: str) -> str | None:
    """Require a URL usable as a concatenation base.

    `WorkdayCompany` builds request and job URLs by appending paths to this
    value, so a query string or fragment would be swallowed into the middle of
    the path (`.../#frag/wday/cxs/...`) and the request would target the wrong
    endpoint. Reject it during validation rather than at fetch time.
    """
    checked = _require_http_url(value, field_name)
    if checked is None:
        return None
    parts = urlsplit(checked)
    if parts.query or parts.fragment:
        raise ValueError(
            f"{field_name} must not contain a query string or fragment, got {value!r}"
        )
    return checked


ASHBY = "ashby"
SMARTRECRUITERS = "smartrecruiters"
WORKABLE = "workable"
GREENHOUSE = "greenhouse"
LEVER = "lever"
EIGHTFOLD = "eightfold"
DIRECT_TECH = "direct_tech"
WORKDAY = "workday"

_MODEL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _require_host_label(value: str | None, field_name: str) -> str | None:
    """Require a single DNS label, safe to interpolate into a hostname.

    Milestone 5 moved these identifiers from Python literals into configuration,
    so they must be validated: a value containing ``/``, ``?``, ``#``, ``@`` or a
    dot can relocate the request to a different host, which for ``wd_company`` is
    interpolated as ``https://{wd_company}.wd{n}.myworkdayjobs.com``.

    DNS label length (1–63 characters, RFC 1035 § 2.3.4) is enforced as
    validation completeness so an invalid label fails here rather than at DNS
    resolution.
    """
    if value is None:
        return None
    token = value.strip()
    if not HOST_LABEL_RE.match(token):
        raise ValueError(
            f"{field_name} must be a lowercase hostname label (letters, digits and "
            f"internal hyphens), got {value!r}"
        )
    if len(token) > 63:
        raise ValueError(
            f"{field_name} exceeds the DNS per-label limit of 63 characters "
            f"({len(token)} characters), got {value!r}"
        )
    return token


def _require_path_token(value: str | None, field_name: str) -> str | None:
    """Require a value safe to interpolate into a URL path segment.

    Rejects `/`, `?`, `#`, whitespace and empty-ish values so a configured token
    cannot escape the intended path or append a query/fragment.
    """
    if value is None:
        return None
    token = value.strip()
    if not PATH_TOKEN_RE.match(token) or set(token) <= {"."}:
        raise ValueError(
            f"{field_name} must be a URL path token (letters, digits, '_', '.', '-'; "
            f"no '/', '?', '#' or whitespace), got {value!r}"
        )
    return token


def _require_workable_subdomain(value: str | None, field_name: str) -> str | None:
    """Require a Workable account subdomain (DNS label).

    Workable account subdomains are DNS labels (e.g. 'huggingface', 'foo-bar').
    Rejects '_', '.', '/', '?', '#', spaces, uppercase, empty, and labels >63 chars.
    """
    if value is None:
        return None
    token = value.strip()
    if not HOST_LABEL_RE.match(token):
        raise ValueError(
            f"{field_name} must be a lowercase DNS label (letters, digits, hyphens; "
            f"no '_', '.', '/', '?', '#', spaces), got {value!r}"
        )
    if len(token) > 63:
        raise ValueError(
            f"{field_name} exceeds the DNS per-label limit of 63 characters "
            f"({len(token)} characters), got {value!r}"
        )
    return token


def _reject_blank(value: str | None) -> str | None:
    """Reject an explicit empty string."""
    if value is not None and not value.strip():
        raise ValueError("value must be a non-empty string")
    return value


def _reject_null(value: Any) -> Any:
    """Reject an explicit null for a field whose provider entry is not nullable.

    Pydantic does not validate defaults, so this fires only when a document
    writes `field: null` outright, which distinguishes "omit and inherit" from
    "set to nothing".
    """
    if value is None:
        raise ValueError("cannot be null; omit the field to inherit the built-in value")
    return value


def _validate_bare_host(value: str | None) -> str | None:
    """Require a dotted lowercase host that cannot name an unintended endpoint.

    The value is interpolated as ``https://{hostname}/api/pcsx/search``, so besides
    rejecting scheme, path, query, fragment and userinfo characters it must also
    reject the degenerate forms that a bare character-class test lets through: an
    empty label (``a..b``), a leading or trailing dot (``.evil.com`` resolves to
    ``evil.com``), and labels beginning or ending in a hyphen. Two or more labels
    are required so a bare value cannot target an unexpected apex host. Lowercase
    is required because providers perform no case folding, keeping host-position
    identifiers consistent with ``workday.wd_company``.

    DNS length bounds are enforced as validation completeness: each label must be
    1–63 octets and the total hostname must not exceed 253 characters (RFC 1035 §
    2.3.4, RFC 1123 § 2.1).
    """
    if value is None:
        return None
    host = value.strip()
    if any(token in host for token in ("://", "/", "?", "#", "@", " ", "\t", "\n")):
        raise ValueError(f"hostname must be a bare host without scheme or path, got {value!r}")
    if not HOST_NAME_RE.match(host):
        raise ValueError(
            "hostname must be a lowercase dotted host of two or more non-empty labels "
            f"(letters, digits, internal hyphens), got {value!r}"
        )
    # DNS length bounds (RFC 1035 § 2.3.4, RFC 1123 § 2.1).
    if len(host) > 253:
        raise ValueError(
            f"hostname exceeds the DNS total-length limit of 253 characters "
            f"({len(host)} characters), got {value!r}"
        )
    for label in host.split("."):
        if len(label) > 63:
            raise ValueError(
                f"hostname label {label!r} exceeds the DNS per-label limit of 63 characters "
                f"({len(label)} characters), got {value!r}"
            )
    return host


# Shared constrained field types. Entry models and override models are built
# from the same aliases, so a built-in override can never smuggle through data
# that a brand-new company would be rejected for.
_RequiredName = Annotated[str, Field(min_length=1), AfterValidator(_reject_blank)]
_OptionalName = Annotated[str | None, Field(min_length=1), AfterValidator(_reject_blank), AfterValidator(_reject_null)]
_BareHost = Annotated[str, AfterValidator(_validate_bare_host)]
_OptionalBareHost = Annotated[
    str | None, AfterValidator(_validate_bare_host), AfterValidator(_reject_null)
]
_HttpUrl = Annotated[str, AfterValidator(lambda v: _require_http_url(v, "url"))]
_OptionalHttpUrl = Annotated[str | None, AfterValidator(lambda v: _require_http_url(v, "url"))]
# URL that must not be set to null: the corresponding provider entry field is not
# nullable, so null means "invalid" rather than "clear it". The null check runs
# first so the URL validator only ever sees a string.
_OptionalNonNullHttpUrl = Annotated[
    str | None,
    AfterValidator(_reject_null),
    AfterValidator(lambda v: _require_http_url(v, "url")),
]
# Identifier fields validated for the URL position they are interpolated into.
_HostLabel = Annotated[str, AfterValidator(lambda v: _require_host_label(v, "host label"))]
# `_reject_null` runs first so an explicit `field: null` stays rejected: the token
# helpers return early on None (meaning "omit and inherit"), which on its own
# would silently re-admit nulls these fields must refuse.
_OptionalHostLabel = Annotated[
    str | None,
    AfterValidator(_reject_null),
    AfterValidator(lambda v: _require_host_label(v, "host label")),
]
_PathToken = Annotated[str, AfterValidator(lambda v: _require_path_token(v, "path token"))]
_OptionalPathToken = Annotated[
    str | None,
    AfterValidator(_reject_null),
    AfterValidator(lambda v: _require_path_token(v, "path token")),
]
_WorkableSubdomain = Annotated[
    str, AfterValidator(lambda v: _require_workable_subdomain(v, "account_subdomain"))
]
_OptionalWorkableSubdomain = Annotated[
    str | None,
    AfterValidator(_reject_null),
    AfterValidator(lambda v: _require_workable_subdomain(v, "account_subdomain")),
]

# Nullable *and* base-shaped: WorkdayCompany.base_url may be cleared to null, and
# must not carry a query/fragment because providers append paths to it.
_OptionalBaseUrl = Annotated[
    str | None,
    AfterValidator(lambda v: _require_base_url(v, "base_url")),
]
_OptionalInt = Annotated[int | None, AfterValidator(_reject_null)]
_NonNullBool = Annotated[bool | None, AfterValidator(_reject_null)]
_StringList = Annotated[list[str], AfterValidator(lambda v: list(v))]
_OptionalStringList = Annotated[list[str] | None, AfterValidator(_reject_null)]


def _validate_company_key(value: str) -> str:
    """Reject malformed company identifiers."""
    if not COMPANY_KEY_RE.match(value):
        raise ValueError(
            "company id must be lowercase letters/digits separated by '_' or '-', "
            f"got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Complete (new-company) entry shapes.
#
# A configuration entry whose id is NOT a built-in company must fully specify
# the fields that family needs.
# ---------------------------------------------------------------------------


class _EntryBase(BaseModel):
    """Common shape: every entry is keyed by an explicit, validated id."""

    model_config = _MODEL_CONFIG

    id: str = Field(min_length=1, max_length=64)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_company_key(value)

    def to_entry(self) -> Any:  # pragma: no cover - implemented by subclasses
        """Build the typed provider entry."""
        raise NotImplementedError


class AshbyEntry(_EntryBase):
    """A new Ashby company."""

    name: _RequiredName
    board_name: _PathToken
    enabled: bool = True

    def to_entry(self) -> AshbyCompany:
        """Build the typed provider entry."""
        return AshbyCompany(
            name=self.name, board_name=self.board_name, enabled=self.enabled
        )


class SmartRecruitersEntry(_EntryBase):
    """A new SmartRecruiters company."""

    name: _RequiredName
    company_identifier: _PathToken
    enabled: bool = True

    def to_entry(self) -> SmartRecruitersCompany:
        """Build the typed provider entry."""
        return SmartRecruitersCompany(
            name=self.name, company_identifier=self.company_identifier, enabled=self.enabled
        )


class WorkableEntry(_EntryBase):
    """A new Workable company."""

    name: _RequiredName
    account_subdomain: _WorkableSubdomain
    enabled: bool = True

    def to_entry(self) -> WorkableCompany:
        """Build the typed provider entry."""
        return WorkableCompany(
            name=self.name,
            account_subdomain=self.account_subdomain,
            enabled=self.enabled,
        )


class GreenhouseEntry(_EntryBase):
    """A new Greenhouse company."""

    name: _RequiredName
    board_token: _PathToken
    enabled: bool = True

    def to_entry(self) -> GreenhouseCompany:
        """Build the typed provider entry."""
        return GreenhouseCompany(
            name=self.name, board_token=self.board_token, enabled=self.enabled
        )


class LeverEntry(_EntryBase):
    """A new Lever company."""

    name: _RequiredName
    slug: _PathToken
    enabled: bool = True

    def to_entry(self) -> LeverCompany:
        """Build the typed provider entry."""
        return LeverCompany(name=self.name, slug=self.slug, enabled=self.enabled)


class EightfoldEntry(_EntryBase):
    """A new Eightfold AI company."""

    name: _RequiredName
    hostname: _BareHost
    domain: _RequiredName
    locations: _StringList = Field(default_factory=list)
    filter_distance: str | None = "16"
    enabled: bool = True

    def to_entry(self) -> EightfoldCompany:
        """Build the typed provider entry."""
        return EightfoldCompany(
            name=self.name,
            hostname=self.hostname,
            domain=self.domain,
            locations=list(self.locations),
            filter_distance=self.filter_distance,
            enabled=self.enabled,
        )


class DirectTechEntry(_EntryBase):
    """A new direct tech career endpoint.

    ``default_query`` materially narrows coverage, so it is an explicit,
    configurable field rather than hidden provider behavior. It may be set to
    `""` to drop the filter; null is rejected because the provider entry field is
    not nullable.

    The default is the shared constant from the entry type, so this schema and
    ``DirectTechCompany`` can never disagree.
    """

    name: _RequiredName
    search_url: _HttpUrl
    default_query: Annotated[str, AfterValidator(_reject_null)] = DIRECT_TECH_DEFAULT_QUERY
    default_location: Annotated[str, AfterValidator(_reject_null)] = ""
    locations: _StringList = Field(default_factory=list)
    enabled: bool = True

    def to_entry(self) -> DirectTechCompany:
        """Build the typed provider entry."""
        return DirectTechCompany(
            provider_id=self.id,
            name=self.name,
            search_url=self.search_url,
            default_query=self.default_query,
            default_location=self.default_location,
            locations=list(self.locations),
            enabled=self.enabled,
        )


class WorkdayEntry(_EntryBase):
    """A new Workday tenant."""

    name: _RequiredName
    wd_company: _HostLabel
    wd_version: int = Field(default=1, ge=1, le=99)
    wd_suffix: _PathToken = "External"
    wd_locations: _StringList = Field(default_factory=list)
    base_url: _OptionalBaseUrl = None
    enabled: bool = True

    def to_entry(self) -> WorkdayCompany:
        """Build the typed provider entry."""
        return WorkdayCompany(
            name=self.name,
            wd_company=self.wd_company,
            wd_version=self.wd_version,
            wd_suffix=self.wd_suffix,
            wd_locations=list(self.wd_locations),
            base_url=self.base_url,
            enabled=self.enabled,
        )


ENTRY_MODELS: dict[str, type[_EntryBase]] = {
    ASHBY: AshbyEntry,
    SMARTRECRUITERS: SmartRecruitersEntry,
    WORKABLE: WorkableEntry,
    GREENHOUSE: GreenhouseEntry,
    LEVER: LeverEntry,
    EIGHTFOLD: EightfoldEntry,
    DIRECT_TECH: DirectTechEntry,
    WORKDAY: WorkdayEntry,
}

PROVIDER_KINDS = tuple(ENTRY_MODELS)

# Provider families whose company catalog is registry-managed.
MANAGED_PROVIDERS: tuple[str, ...] = PROVIDER_KINDS


# ---------------------------------------------------------------------------
# Partial (override) shapes.
#
# A configuration entry whose id IS a built-in company may specify only the
# fields it changes; unspecified fields are inherited from the built-in.
# ---------------------------------------------------------------------------


class _OverrideBase(BaseModel):
    """Common shape for overrides.

    An omitted field inherits the built-in value. An explicit `null` is accepted
    only where the provider entry field is genuinely nullable
    (``EightfoldCompany.filter_distance`` and ``WorkdayCompany.base_url``) and is
    rejected elsewhere, so "clear this" and "inherit this" are never ambiguous.
    Field constraints are shared with the matching complete-entry model, so a
    built-in override cannot accept data a new company would be rejected for.
    """

    model_config = _MODEL_CONFIG

    id: str = Field(min_length=1, max_length=64)
    enabled: _NonNullBool = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _validate_company_key(value)


class AshbyOverride(_OverrideBase):
    """Partial Ashby override."""

    name: _OptionalName = None
    board_name: _OptionalPathToken = None


class SmartRecruitersOverride(_OverrideBase):
    """Partial SmartRecruiters override."""

    name: _OptionalName = None
    company_identifier: _OptionalPathToken = None


class WorkableOverride(_OverrideBase):
    """Partial Workable override."""

    name: _OptionalName = None
    account_subdomain: _OptionalWorkableSubdomain = None


class GreenhouseOverride(_OverrideBase):
    """Partial Greenhouse override."""

    name: _OptionalName = None
    board_token: _OptionalPathToken = None


class LeverOverride(_OverrideBase):
    """Partial Lever override."""

    name: _OptionalName = None
    slug: _OptionalPathToken = None


class EightfoldOverride(_OverrideBase):
    """Partial Eightfold override.

    `filter_distance` is nullable in the provider entry, so an explicit null
    stores None rather than inheriting the built-in distance. The pre-existing
    provider still substitutes its own fallback at request time
    (`company.filter_distance or "16"`), so null clears the *stored* value but
    does not remove narrowing from the outgoing request. Rewriting that provider
    expression would alter fetch semantics, which Milestone 5 forbids.
    """

    name: _OptionalName = None
    hostname: _OptionalBareHost = None
    domain: _OptionalName = None
    locations: _OptionalStringList = None
    filter_distance: str | None = None


class DirectTechOverride(_OverrideBase):
    """Partial direct tech override."""

    name: _OptionalName = None
    # Not nullable in the entry type: a company always needs a search endpoint.
    search_url: _OptionalNonNullHttpUrl = None
    # Not nullable in the entry type, but empty IS meaningful: "" drops the
    # narrowing query, so this uses the null-rejecting alias rather than
    # _OptionalName, which also enforces a minimum length.
    default_query: Annotated[str | None, AfterValidator(_reject_null)] = None
    default_location: Annotated[str | None, AfterValidator(_reject_null)] = None
    locations: _OptionalStringList = None


class WorkdayOverride(_OverrideBase):
    """Partial Workday override.

    `base_url` is nullable in the provider entry, so an explicit null means
    "derive the URL from the tenant" rather than "inherit". Unlike the Eightfold
    distance field, this one does change request behaviour: a None base URL makes
    `get_base_url()` fall back to the tenant-derived host.
    """

    name: _OptionalName = None
    wd_company: _OptionalHostLabel = None
    # The entry field is a plain int, so null is rejected; ge/le still bound a
    # supplied value. The alias already admits None for the omitted case, exactly
    # like `_NonNullBool` does for `enabled`.
    wd_version: _OptionalInt = Field(default=None, ge=1, le=99)
    wd_suffix: _OptionalPathToken = None
    wd_locations: _OptionalStringList = None
    # Genuinely nullable on WorkdayCompany: null clears a custom base URL so the
    # tenant-derived default is used again.
    base_url: _OptionalBaseUrl = None


OVERRIDE_MODELS: dict[str, type[_OverrideBase]] = {
    ASHBY: AshbyOverride,
    SMARTRECRUITERS: SmartRecruitersOverride,
    WORKABLE: WorkableOverride,
    GREENHOUSE: GreenhouseOverride,
    LEVER: LeverOverride,
    EIGHTFOLD: EightfoldOverride,
    DIRECT_TECH: DirectTechOverride,
    WORKDAY: WorkdayOverride,
}


def parse_entry(provider: str, raw: dict[str, Any]) -> _EntryBase:
    """Validate ``raw`` as a complete new entry for ``provider``."""
    return ENTRY_MODELS[provider].model_validate(raw)


def parse_override(provider: str, raw: dict[str, Any]) -> _OverrideBase:
    """Validate ``raw`` as a partial override for an existing ``provider`` entry."""
    return OVERRIDE_MODELS[provider].model_validate(raw)


def is_builtin_company(provider: str, company_id: str) -> bool:
    """Return whether ``company_id`` exists in the built-in catalog."""
    return company_id in BUILTIN_COMPANY_IDS.get(provider, frozenset())


def _normalized_company_id(raw: dict[str, Any]) -> str:
    """Return the id the typed model will hold after whitespace stripping.

    Pydantic strips surrounding whitespace on `id`, so every comparison that
    decides override-vs-new or duplicate-conflict must use the same normalized
    value. Comparing the raw form instead lets `" jfrog "` be treated as a brand
    new company and `" newco "` evade a collision with `"newco"`.
    """
    return str(raw.get("id", "")).strip()


def parse_company(provider: str, raw: dict[str, Any]) -> _EntryBase | _OverrideBase:
    """Validate one company entry as typed data, rejecting unknown fields.

    Built-in ids are validated against the partial override shape; anything else
    must fully specify the fields a new company of that family requires.
    """
    if is_builtin_company(provider, _normalized_company_id(raw)):
        return parse_override(provider, raw)
    return parse_entry(provider, raw)


def _parse_company_in_family(
    provider: str, raw: dict[str, Any], *, index: int
) -> _EntryBase | _OverrideBase:
    """Parse one entry and re-raise shape errors with family, position and id.

    Pydantic reports the offending field, not the block it came from, so a
    cross-family payload (for example a Lever entry under `greenhouse`) would
    otherwise surface as a bare "slug: Extra inputs are not permitted". Prefixing
    the family, the list position and the company id keeps each problem
    attributable even when several entries share a shape.
    """
    location = f"companies[{index}]"
    if isinstance(raw, dict):
        company_id = _normalized_company_id(raw)
        if company_id:
            location = f"companies[{index}] (id={company_id!r})"
    if not isinstance(raw, dict):
        # ValueError, not TypeError: pydantic converts ValueError into
        # ValidationError, which core surfaces as the registry's single
        # CompanyRegistryError contract. A TypeError escapes that chain.
        raise ValueError(  # noqa: TRY004
            f"{provider}: {location}: company entries must be mappings"
        )
    try:
        return parse_company(provider, raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'entry'}: {err['msg']}"
            for err in exc.errors(include_url=False)
        )
        raise ValueError(f"{provider}: {location}: {details}") from exc


def _check_provider_key(provider: str) -> None:
    """Validate one provider key as an identifier of a supported family."""
    if provider not in ENTRY_MODELS:
        supported = ", ".join(MANAGED_PROVIDERS)
        raise ValueError(
            f"unsupported provider {provider!r}; Milestone 5 supports: {supported}"
        )
    if not PROVIDER_ID_RE.match(provider):
        raise ValueError(f"malformed provider identifier {provider!r}")


def _duplicate_conflict(provider: str, raws: list[Any]) -> str | None:
    """Return an actionable message when one document repeats an id differently.

    Ids are compared in normalized form (see :func:`_normalized_company_id`) so
    `newco` and `" newco "` cannot collapse onto the same key later while evading
    this check. Payloads are compared with `id` excluded for the same reason.
    """
    seen: dict[str, dict[str, Any]] = {}
    for raw in raws:
        if not isinstance(raw, dict) or "id" not in raw:
            return f"{provider}: every company entry must be a mapping with an explicit 'id'"
        company_id = _normalized_company_id(raw)
        payload = {key: value for key, value in raw.items() if key != "id"}
        if company_id in seen:
            if seen[company_id] == payload:
                continue
            return (
                f"{provider}: duplicate conflicting entries for company {company_id!r}; "
                "merge them into a single entry"
            )
        seen[company_id] = payload
    return None


# Concrete models that may appear in a provider block. The abstract bases are
# deliberately excluded: a union of `_EntryBase | _OverrideBase` would let
# pydantic coerce any partial mapping into `_OverrideBase`, which has no
# `to_entry()`, so an unvalidated shape could still reach the merge step.
CONCRETE_COMPANY_MODELS: tuple[type[BaseModel], ...] = (
    *ENTRY_MODELS.values(),
    *OVERRIDE_MODELS.values(),
)


class ProviderConfig(BaseModel):
    """Validated configuration block for one provider family.

    `companies` accepts only the concrete, already-validated entry models, so
    building a ProviderConfig directly (bypassing RegistryConfig's before-validator,
    which is what turns YAML mappings into typed models) cannot smuggle unvalidated
    data into the merge step.
    """

    model_config = _MODEL_CONFIG

    companies: list[Any] = Field(default_factory=list)

    @field_validator("companies")
    @classmethod
    def _check_companies_are_typed(cls, value: list[Any]) -> list[Any]:
        names = " | ".join(model.__name__ for model in CONCRETE_COMPANY_MODELS)
        for index, item in enumerate(value):
            if not isinstance(item, CONCRETE_COMPANY_MODELS):
                # ValueError, not TypeError: pydantic converts ValueError into
                # ValidationError, which core surfaces as the registry's single
                # CompanyRegistryError contract. A TypeError escapes that chain.
                raise ValueError(  # noqa: TRY004
                    f"companies[{index}] must be one of the validated entry models ({names}), "
                    f"got {type(item).__name__}; parse documents with RegistryConfig.model_validate "
                    "so entries are typed and validated"
                )
        return value


class RegistryConfig(BaseModel):
    """Top-level validated registry document with typed company entries."""

    model_config = _MODEL_CONFIG

    version: int = 1
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported registry version {value!r}; only version 1 exists")
        return value

    @model_validator(mode="before")
    @classmethod
    def _type_entries(cls, value: Any) -> Any:
        """Replace raw company mappings with typed, validated entry models."""
        if not isinstance(value, dict) or not isinstance(value.get("providers"), dict):
            return value
        typed: dict[str, Any] = {}
        for provider, block in value["providers"].items():
            # Reject unsupported or malformed provider keys here rather than
            # deferring to the field validator: a populated block for an
            # unsupported family would otherwise fail first on its company shape
            # and report "must be one of the validated entry models" instead of
            # the actual problem, which is the unknown provider.
            _check_provider_key(provider)
            if isinstance(block, ProviderConfig):
                # Already typed by a programmatic caller; nothing to convert.
                typed[provider] = block
                continue
            if not isinstance(block, dict):
                # ValueError, not TypeError: pydantic converts ValueError into
                # ValidationError, which core surfaces as the registry's single
                # CompanyRegistryError contract. A TypeError escapes that chain.
                raise ValueError(  # noqa: TRY004
                    f"{provider}: provider block must be a mapping"
                )
            raws = block.get("companies", [])
            if not isinstance(raws, list):
                # ValueError, not TypeError: pydantic wraps ValueError into
                # ValidationError, which core converts into the registry's single
                # CompanyRegistryError contract. A TypeError escapes that chain and
                # would surface as a bare TypeError to callers.
                raise ValueError(  # noqa: TRY004
                    f"{provider}: 'companies' must be a list of mappings"
                )
            conflict = _duplicate_conflict(provider, raws)
            if conflict:
                raise ValueError(conflict)
            typed[provider] = {
                **block,
                "companies": [
                    _parse_company_in_family(provider, raw, index=index) for index, raw in enumerate(raws)
                ],
            }
        return {**value, "providers": typed}

    @field_validator("providers")
    @classmethod
    def _check_provider_keys(
        cls, value: dict[str, ProviderConfig]
    ) -> dict[str, ProviderConfig]:
        for key in value:
            _check_provider_key(key)
        return value


def supported_providers() -> tuple[str, ...]:
    """Return the provider families the registry can configure."""
    return MANAGED_PROVIDERS


__all__ = [
    "ASHBY",
    "BUILTIN_COMPANY_IDS",
    "ENTRY_MODELS",
    "MANAGED_PROVIDERS",
    "OVERRIDE_MODELS",
    "PROVIDER_ID_RE",
    "SMARTRECRUITERS",
    "WORKABLE",
    "AshbyEntry",
    "AshbyOverride",
    "ProviderConfig",
    "RegistryConfig",
    "SmartRecruitersEntry",
    "SmartRecruitersOverride",
    "WorkableEntry",
    "WorkableOverride",
    "is_builtin_company",
    "parse_company",
    "parse_entry",
    "parse_override",
    "supported_providers",
]
