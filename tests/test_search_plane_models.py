"""Tests for Milestone 6 Unified Search Plane domain contracts (M6-A1).

Covers:
- JobSearchRequest validation, defaults, and boundary constraints
- JobSearchResultItem and JobSearchResultSet structure
- JobRef typed model, versioning, opaque base64url encoding/decoding
- Delimiter-rich and URL locator round-tripping
- Size bounds and tampering/corruption rejection
- FetchStatus retrieval-only factual invariant (no M7 expiry/liveness)
- FetchResult container
- SourceCapabilities defaults and overrides
"""

from __future__ import annotations

import base64
import json

import pytest
from pydantic import ValidationError

from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
    JobSearchRequest,
    JobSearchResultItem,
    JobSearchResultSet,
    SourceCapabilities,
)
from job_mcp.models.schemas import Job, WorkMode

# ===========================================================================
# 1. JobSearchRequest Tests
# ===========================================================================


def test_job_search_request_defaults() -> None:
    """Default request has limit=25 and empty optional filters."""
    req = JobSearchRequest()
    assert req.query is None
    assert req.location is None
    assert req.work_mode is None
    assert req.company is None
    assert req.tech_stack == []
    assert req.seniority is None
    assert req.limit == 25
    assert req.sources is None
    assert req.cursor is None
    assert req.freshness_days is None


def test_job_search_request_populated() -> None:
    """Fully populated request preserves all values and parses WorkMode."""
    req = JobSearchRequest(
        query="Senior Python Engineer",
        location="Tel Aviv",
        work_mode=WorkMode.HYBRID,
        company="JFrog",
        tech_stack=["Python", "FastAPI"],
        seniority="Senior",
        limit=50,
        sources=["greenhouse", "lever"],
        cursor="cursor_abc123",
        freshness_days=7,
    )
    assert req.query == "Senior Python Engineer"
    assert req.location == "Tel Aviv"
    assert req.work_mode == WorkMode.HYBRID
    assert req.company == "JFrog"
    assert req.tech_stack == ["Python", "FastAPI"]
    assert req.seniority == "Senior"
    assert req.limit == 50
    assert req.sources == ["greenhouse", "lever"]
    assert req.cursor == "cursor_abc123"
    assert req.freshness_days == 7


@pytest.mark.parametrize("invalid_limit", [0, -1, 201, 1000])
def test_job_search_request_rejects_out_of_range_limit(invalid_limit: int) -> None:
    """Limit must be between 1 and 200 inclusive."""
    with pytest.raises(ValidationError):
        JobSearchRequest(limit=invalid_limit)


@pytest.mark.parametrize("valid_limit", [1, 25, 100, 200])
def test_job_search_request_accepts_valid_limit(valid_limit: int) -> None:
    req = JobSearchRequest(limit=valid_limit)
    assert req.limit == valid_limit


@pytest.mark.parametrize("invalid_freshness", [0, -1, -30])
def test_job_search_request_rejects_non_positive_freshness(invalid_freshness: int) -> None:
    """freshness_days represents a request-time window and must be positive if supplied."""
    with pytest.raises(ValidationError):
        JobSearchRequest(freshness_days=invalid_freshness)


def test_job_search_request_accepts_valid_freshness() -> None:
    req = JobSearchRequest(freshness_days=14)
    assert req.freshness_days == 14


# ===========================================================================
# 2. JobSearchResultItem and JobSearchResultSet Tests
# ===========================================================================


def test_job_search_result_item_minimal() -> None:
    """Result item requires ref, title, company, and source_family."""
    item = JobSearchResultItem(
        ref="v1_testref",
        title="Backend Engineer",
        company="Acme",
        source_family="greenhouse",
    )
    assert item.ref == "v1_testref"
    assert item.title == "Backend Engineer"
    assert item.company == "Acme"
    assert item.location == ""
    assert item.work_mode is None
    assert item.canonical_url is None
    assert item.source_family == "greenhouse"
    assert item.posted_date is None
    assert item.retrieval_method == "structured"


def test_job_search_result_item_full() -> None:
    item = JobSearchResultItem(
        ref="v1_testref",
        title="AI Engineer",
        company="StartupAI",
        location="Tel Aviv",
        work_mode=WorkMode.REMOTE,
        canonical_url="https://jobs.ashbyhq.com/startupai/123",
        source_family="ashby",
        posted_date="2026-10-01",
        retrieval_method="discovered",
    )
    assert item.work_mode == WorkMode.REMOTE
    assert item.canonical_url == "https://jobs.ashbyhq.com/startupai/123"
    assert item.retrieval_method == "discovered"


def test_job_search_result_set() -> None:
    item = JobSearchResultItem(
        ref="v1_ref1",
        title="Engineer",
        company="Co",
        source_family="lever",
    )
    result_set = JobSearchResultSet(
        items=[item],
        total_estimated=1,
        next_cursor="cursor_2",
        warnings=["Rate limit approaching on provider X"],
    )
    assert len(result_set.items) == 1
    assert result_set.total_estimated == 1
    assert result_set.next_cursor == "cursor_2"
    assert len(result_set.warnings) == 1


# ===========================================================================
# 3. JobRef Encoding, Decoding, and Invariants
# ===========================================================================


def test_job_ref_creation() -> None:
    ref = JobRef(version=1, source_family="greenhouse", account="jfrog", locator="4928102")
    assert ref.version == 1
    assert ref.source_family == "greenhouse"
    assert ref.account == "jfrog"
    assert ref.locator == "4928102"


def test_job_ref_account_optional() -> None:
    """Sources without an account concept (e.g. direct url or global board) leave account None."""
    ref = JobRef(version=1, source_family="direct_tech", account=None, locator="google_12345")
    assert ref.account is None


def test_job_ref_round_trip() -> None:
    """Encoding and then decoding a JobRef produces an identical instance."""
    original = JobRef(
        version=1,
        source_family="lever",
        account="redis",
        locator="7a28b19c-4f81-42ab-9912-abcdef123456",
    )
    encoded = original.encode()
    assert isinstance(encoded, str)
    assert encoded.startswith("v1_")

    decoded = JobRef.decode(encoded)
    assert decoded == original
    assert decoded.version == 1
    assert decoded.source_family == "lever"
    assert decoded.account == "redis"
    assert decoded.locator == "7a28b19c-4f81-42ab-9912-abcdef123456"


def test_job_ref_round_trip_without_account() -> None:
    original = JobRef(
        version=1,
        source_family="linkedin",
        account=None,
        locator="4152839402",
    )
    encoded = original.encode()
    decoded = JobRef.decode(encoded)
    assert decoded == original
    assert decoded.account is None


def test_job_ref_idempotent_encoding() -> None:
    """Two identical JobRefs encode to the exact same opaque string."""
    ref1 = JobRef(version=1, source_family="ashby", account="anthropic", locator="pos_999")
    ref2 = JobRef(version=1, source_family="ashby", account="anthropic", locator="pos_999")
    assert ref1.encode() == ref2.encode()


def test_job_ref_str_returns_encoded() -> None:
    ref = JobRef(version=1, source_family="greenhouse", account="jfrog", locator="123")
    assert str(ref) == ref.encode()


# ---------------------------------------------------------------------------
# Delimiter Immunity & Arbitrary Locator Content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tricky_locator",
    [
        "JR:1234:56:EXT",                      # Colons
        "dept/subdept/pos/99",                 # Slashes
        "?query=python&source=indeed#apply",   # Query and fragments
        "https://careers.google.com/jobs/results/123456789/",  # Full URL
        "job with spaces and (parens) [brackets]",
        "מהנדס/ת תוכנה בכיר/ה - פייתון",       # Hebrew and slashes
        "token_with_underscores_and-hyphens.dots",
        '{"nested": "json", "id": 42}',        # JSON-like string
        "a" * 2000,                            # Long locator (under 2048 limit)
    ],
)
def test_job_ref_handles_arbitrary_locator_content(tricky_locator: str) -> None:
    """Because JobRef is opaque JSON-encoded rather than delimiter-split, arbitrary characters round-trip."""
    original = JobRef(
        version=1,
        source_family="test_family",
        account="acc_test",
        locator=tricky_locator,
    )
    encoded = original.encode()
    decoded = JobRef.decode(encoded)
    assert decoded.locator == tricky_locator
    assert decoded == original


# ---------------------------------------------------------------------------
# Size Bounding Tests
# ---------------------------------------------------------------------------


def test_job_ref_rejects_oversized_locator() -> None:
    """Locators over 2048 characters must be rejected to prevent memory abuse."""
    huge_locator = "x" * 2049
    with pytest.raises(ValueError, match="locator exceeds maximum length"):
        JobRef(version=1, source_family="test", account="acc", locator=huge_locator)


def test_job_ref_rejects_oversized_encoded_string() -> None:
    """decode() must reject strings exceeding maximum bounded size (4096 chars)."""
    oversized_ref = "v1_" + ("A" * 4100)
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        JobRef.decode(oversized_ref)


# ---------------------------------------------------------------------------
# Error & Tampering Cases
# ---------------------------------------------------------------------------


def test_job_ref_decode_rejects_empty_or_non_string() -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        JobRef.decode("")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        JobRef.decode(None)  # type: ignore[arg-type]


def test_job_ref_decode_rejects_missing_version_prefix() -> None:
    """A raw base64 or unversioned string must be rejected."""
    with pytest.raises(ValueError, match="missing version prefix"):
        JobRef.decode("eyJ2IjoxLCJmIjoidGVzdCJ9")


def test_job_ref_decode_rejects_unsupported_version() -> None:
    """A v2_ or other future version must fail with clear diagnostics."""
    with pytest.raises(ValueError, match="Unsupported JobRef version"):
        JobRef.decode("v2_somepayload")


def test_job_ref_decode_rejects_malformed_base64() -> None:
    with pytest.raises(ValueError, match="invalid base64 encoding"):
        JobRef.decode("v1_!!!invalid-base64-content@@@")


def test_job_ref_decode_rejects_non_json_payload() -> None:
    raw_bytes = b"hello this is not json"
    b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")
    with pytest.raises(ValueError, match="invalid JSON payload"):
        JobRef.decode(f"v1_{b64}")


def test_job_ref_decode_rejects_missing_required_fields() -> None:
    # Payload missing 'l' (locator)
    bad_payload = json.dumps({"v": 1, "f": "greenhouse", "a": "jfrog"})
    b64 = base64.urlsafe_b64encode(bad_payload.encode("utf-8")).decode("ascii").rstrip("=")
    with pytest.raises(ValueError, match="missing required field"):
        JobRef.decode(f"v1_{b64}")


def test_job_ref_decode_rejects_version_mismatch_in_payload() -> None:
    # Prefix says v1, payload says v2
    bad_payload = json.dumps({"v": 2, "f": "greenhouse", "a": "jfrog", "l": "123"})
    b64 = base64.urlsafe_b64encode(bad_payload.encode("utf-8")).decode("ascii").rstrip("=")
    with pytest.raises(ValueError, match="version mismatch"):
        JobRef.decode(f"v1_{b64}")


def test_job_ref_rejects_unsupported_version_in_init() -> None:
    with pytest.raises(ValueError, match="version must be 1"):
        JobRef(version=2, source_family="greenhouse", locator="123")


@pytest.mark.parametrize("bad_family", ["", "   ", "UPPERCASE", "invalid@family", "family with space"])
def test_job_ref_validates_source_family_format(bad_family: str) -> None:
    with pytest.raises(ValueError):
        JobRef(version=1, source_family=bad_family, locator="123")


# ===========================================================================
# 4. FetchStatus and FetchResult Tests
# ===========================================================================


def test_fetch_status_members_are_strictly_retrieval_facts() -> None:
    """Milestone 6 fetch statuses must NOT contain liveness/staleness concepts (M7)."""
    expected_members = {
        "FOUND",
        "NOT_FOUND",
        "INVALID_REF",
        "UNSUPPORTED_REFETCH",
        "UPSTREAM_ERROR",
    }
    actual_members = {status.name for status in FetchStatus}
    assert actual_members == expected_members

    # Explicit negative assertions: M7 semantics must not leak into M6
    forbidden_terms = ["EXPIRED", "STALE", "DEAD", "LIVENESS", "ACTIVE", "REPOST"]
    for member in actual_members:
        for term in forbidden_terms:
            assert term not in member, f"Forbidden M7 term {term!r} found in FetchStatus.{member}"


def test_fetch_result_found() -> None:
    sample_job = Job(
        job_id="test_1",
        title="Software Engineer",
        company="Acme Corp",
        location="Tel Aviv",
    )
    result = FetchResult(
        status=FetchStatus.FOUND,
        job=sample_job,
        ref="v1_testref",
    )
    assert result.status == FetchStatus.FOUND
    assert result.job is not None
    assert result.job.title == "Software Engineer"
    assert result.ref == "v1_testref"
    assert result.diagnostic is None


def test_fetch_result_not_found() -> None:
    result = FetchResult(
        status=FetchStatus.NOT_FOUND,
        job=None,
        ref="v1_testref",
        diagnostic="Posting was not returned by source upstream API.",
    )
    assert result.status == FetchStatus.NOT_FOUND
    assert result.job is None
    assert result.diagnostic == "Posting was not returned by source upstream API."


def test_fetch_result_unsupported_refetch() -> None:
    result = FetchResult(
        status=FetchStatus.UNSUPPORTED_REFETCH,
        ref="v1_testref",
        diagnostic="Source family 'gotfriends' does not support single-posting retrieval.",
    )
    assert result.status == FetchStatus.UNSUPPORTED_REFETCH
    assert result.job is None


# ===========================================================================
# 5. SourceCapabilities Tests
# ===========================================================================


def test_source_capabilities_defaults() -> None:
    """By default, a source is assumed searchable but without native fetch or query filters."""
    caps = SourceCapabilities()
    assert caps.supports_search is True
    assert caps.supports_native_fetch is False
    assert caps.supports_url_fetch is False
    assert caps.supports_query is False
    assert caps.supports_company_filter is False
    assert caps.supports_work_mode is False
    assert caps.supports_pagination is False


def test_source_capabilities_custom() -> None:
    caps = SourceCapabilities(
        supports_search=True,
        supports_native_fetch=True,
        supports_url_fetch=True,
        supports_query=True,
        supports_company_filter=True,
        supports_work_mode=True,
        supports_pagination=True,
    )
    assert caps.supports_native_fetch is True
    assert caps.supports_query is True
    assert caps.supports_pagination is True
