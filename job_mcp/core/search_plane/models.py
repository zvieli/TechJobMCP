"""Domain models and contracts for Milestone 6 Unified Search Plane (M6-A1).

Defines:
- JobSearchRequest: Provider-agnostic retrieval intent with capability negotiation filters.
- JobSearchResultItem: Compact posting summary with opaque locator ref for agent triage.
- JobSearchResultSet: Container for search results with optional pagination cursor and warnings.
- JobRef: Typed, versioned, opaque posting locator with deterministic encoding and decoding.
- FetchStatus: Factual, retrieval-only status enum (free of M7 liveness/staleness concepts).
- FetchResult: Response container for single-posting refetch operations.
- SourceCapabilities: Declared capability flags for backend adapter negotiation.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from job_mcp.models.schemas import Job, WorkMode

# Source family must be lowercase alphanumeric with underscores or hyphens
SOURCE_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

MAX_LOCATOR_LENGTH: int = 2048
MAX_ENCODED_REF_LENGTH: int = 4096


# ===========================================================================
# 1. JobSearchRequest
# ===========================================================================


class JobSearchRequest(BaseModel):
    """Provider-agnostic job search request specification.

    Derived from JobPreferences and existing tool contracts without forcing every
    underlying backend to natively support every filter.
    """

    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    location: str | None = None
    work_mode: WorkMode | None = None
    company: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    seniority: str | None = None
    limit: int = Field(default=25, ge=1, le=200)
    sources: list[str] | None = None
    cursor: str | None = None
    freshness_days: int | None = Field(default=None, ge=1)


# ===========================================================================
# 2. JobSearchResultItem & JobSearchResultSet
# ===========================================================================


class JobSearchResultItem(BaseModel):
    """Compact posting summary returned by the Search Plane for agent triage.

    Provides sufficient metadata for agent evaluation without transmitting
    the full job description payload.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    title: str
    company: str
    location: str = ""
    work_mode: WorkMode | None = None
    canonical_url: str | None = None
    source_family: str
    posted_date: str | None = None
    retrieval_method: str = "structured"


class JobSearchResultSet(BaseModel):
    """Paginated collection of job search results with diagnostic warnings."""

    model_config = ConfigDict(extra="forbid")

    items: list[JobSearchResultItem] = Field(default_factory=list)
    total_estimated: int | None = None
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ===========================================================================
# 3. JobRef (Typed, Versioned, Opaque Locator)
# ===========================================================================


class JobRef(BaseModel):
    """Typed, versioned posting locator for deterministic refetching.

    Encoded as an opaque versioned string (e.g. ``v1_<base64url>``).
    Immune to delimiter collision because the payload is canonical JSON rather than
    delimiter-joined tokens.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    source_family: str
    account: str | None = None
    locator: str

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"version must be 1 (got {v})")
        return v

    @field_validator("source_family")
    @classmethod
    def _validate_source_family(cls, v: str) -> str:
        s = v.strip()
        if not s or not SOURCE_FAMILY_RE.match(s):
            raise ValueError(
                f"source_family must be lowercase alphanumeric with underscores/hyphens, got {v!r}"
            )
        return s

    @field_validator("account")
    @classmethod
    def _validate_account(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        return s

    @field_validator("locator")
    @classmethod
    def _validate_locator(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("locator must be a non-empty string")
        if len(v) > MAX_LOCATOR_LENGTH:
            raise ValueError(
                f"locator exceeds maximum length of {MAX_LOCATOR_LENGTH} characters "
                f"({len(v)} characters)"
            )
        return v

    def encode(self) -> str:
        """Encode this JobRef into a deterministic opaque versioned string."""
        payload: dict[str, Any] = {
            "v": self.version,
            "f": self.source_family,
            "l": self.locator,
        }
        if self.account is not None:
            payload["a"] = self.account

        json_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        b64_str = base64.urlsafe_b64encode(json_bytes).decode("ascii").rstrip("=")
        return f"v{self.version}_{b64_str}"

    def __str__(self) -> str:
        return self.encode()

    @classmethod
    def decode(cls, raw: str) -> JobRef:
        """Decode and validate an opaque versioned reference string."""
        if not raw or not isinstance(raw, str):
            raise ValueError("JobRef must be a non-empty string")
        if len(raw) > MAX_ENCODED_REF_LENGTH:
            raise ValueError(
                f"JobRef exceeds maximum allowed length of {MAX_ENCODED_REF_LENGTH} characters"
            )

        if not raw.startswith("v") or "_" not in raw:
            raise ValueError(
                f"JobRef missing version prefix (expected format 'v<N>_<payload>', got {raw[:20]!r})"
            )

        prefix, _, b64_payload = raw.partition("_")
        version_str = prefix[1:]

        if version_str != "1":
            raise ValueError(f"Unsupported JobRef version: '{prefix}' (supported versions: v1)")

        if not b64_payload:
            raise ValueError("JobRef payload is empty")

        # Restore Base64 padding
        padding = (4 - len(b64_payload) % 4) % 4
        padded_b64 = b64_payload + ("=" * padding)

        try:
            raw_bytes = base64.urlsafe_b64decode(padded_b64)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Malformed JobRef: invalid base64 encoding: {exc}") from exc

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JobRef: invalid JSON payload: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Malformed JobRef: payload is not a JSON object")  # noqa: TRY004

        for req_field in ("v", "f", "l"):
            if req_field not in data:
                raise ValueError(f"Malformed JobRef: missing required field {req_field!r}")

        if data["v"] != 1:
            raise ValueError(
                f"Malformed JobRef: version mismatch in payload (expected 1, got {data['v']})"
            )

        return cls(
            version=data["v"],
            source_family=data["f"],
            account=data.get("a"),
            locator=data["l"],
        )


# ===========================================================================
# 4. FetchStatus & FetchResult
# ===========================================================================


class FetchStatus(str, Enum):
    """Factual, retrieval-only status of a single-posting fetch operation.

    Milestone 6 invariant: Contains no liveness, staleness, or expiry assumptions;
    those belong to Milestone 7 observation semantics.
    """

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    INVALID_REF = "INVALID_REF"
    UNSUPPORTED_REFETCH = "UNSUPPORTED_REFETCH"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"


class FetchResult(BaseModel):
    """Result of a single-posting refetch operation."""

    model_config = ConfigDict(extra="forbid")

    status: FetchStatus
    job: Job | None = None
    ref: str | None = None
    diagnostic: str | None = None


# ===========================================================================
# 5. SourceCapabilities
# ===========================================================================


class SourceCapabilities(BaseModel):
    """Declared retrieval capabilities of a job source backend."""

    model_config = ConfigDict(extra="forbid")

    supports_search: bool = True
    supports_native_fetch: bool = False
    supports_url_fetch: bool = False
    supports_query: bool = False
    supports_company_filter: bool = False
    supports_work_mode: bool = False
    supports_pagination: bool = False


__all__ = [
    "FetchResult",
    "FetchStatus",
    "JobRef",
    "JobSearchRequest",
    "JobSearchResultItem",
    "JobSearchResultSet",
    "SourceCapabilities",
]
