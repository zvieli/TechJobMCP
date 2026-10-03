"""Unit tests for discovery domain models (Milestone 6-C)."""

from __future__ import annotations

import pytest

from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)


def test_discovery_status_enum_values() -> None:
    """Verify all 5 required outward status values exist."""
    assert DiscoveryStatus.CONFIRMED.value == "CONFIRMED"
    assert DiscoveryStatus.AMBIGUOUS.value == "AMBIGUOUS"
    assert DiscoveryStatus.UNSUPPORTED.value == "UNSUPPORTED"
    assert DiscoveryStatus.NOT_FOUND.value == "NOT_FOUND"
    assert DiscoveryStatus.MALFORMED.value == "MALFORMED"


def test_discovery_target_validation() -> None:
    """Validate DiscoveryTarget input validation rules."""
    # Valid targets
    t1 = DiscoveryTarget(company_name="Acme Corp")
    ok, err = t1.validate_input()
    assert ok is True
    assert err is None

    t2 = DiscoveryTarget(career_url="https://jobs.ashbyhq.com/acme")
    ok, err = t2.validate_input()
    assert ok is True
    assert err is None

    t3 = DiscoveryTarget(company_name="Acme", career_url="https://jobs.ashbyhq.com/acme")
    ok, err = t3.validate_input()
    assert ok is True

    # Empty targets fail
    t_empty = DiscoveryTarget()
    ok, err = t_empty.validate_input()
    assert ok is False
    assert "at least one" in (err or "").lower()

    t_blank = DiscoveryTarget(company_name="   ", career_url="   ")
    ok, err = t_blank.validate_input()
    assert ok is False

    # Control characters fail
    t_ctrl = DiscoveryTarget(company_name="Acme\x00Corp")
    ok, err = t_ctrl.validate_input()
    assert ok is False
    assert "control characters" in (err or "").lower()


def test_discovery_evidence_model() -> None:
    """Verify DiscoveryEvidence fields and immutability."""
    ev = DiscoveryEvidence(
        kind="DIRECT_PROVIDER_URL",
        value="ashby:acme",
        source_url="https://jobs.ashbyhq.com/acme",
    )
    assert ev.kind == "DIRECT_PROVIDER_URL"
    assert ev.value == "ashby:acme"
    assert ev.source_url == "https://jobs.ashbyhq.com/acme"

    with pytest.raises(AttributeError):
        ev.kind = "OTHER"  # type: ignore[misc]


def test_discovered_portal_model() -> None:
    """Verify DiscoveredPortal fields and confidence basis."""
    ev = DiscoveryEvidence(kind="DIRECT_PROVIDER_URL", value="ashby:acme")
    portal = DiscoveredPortal(
        source_family="ashby",
        account="acme",
        confidence_basis=(ev,),
    )
    assert portal.source_family == "ashby"
    assert portal.account == "acme"
    assert portal.confidence_basis == (ev,)

    with pytest.raises(AttributeError):
        portal.account = "other"  # type: ignore[misc]


def test_discovery_result_model() -> None:
    """Verify DiscoveryResult construction and default attributes."""
    target = DiscoveryTarget(company_name="Acme")
    ev = DiscoveryEvidence(kind="REGISTRY_MATCH", value="Matched Acme")
    portal = DiscoveredPortal(source_family="workable", account="acme", confidence_basis=(ev,))

    res = DiscoveryResult(
        status=DiscoveryStatus.CONFIRMED,
        input=target,
        candidates=(portal,),
        evidence=(ev,),
        diagnostic=None,
    )

    assert res.status == DiscoveryStatus.CONFIRMED
    assert res.input == target
    assert res.candidates == (portal,)
    assert res.evidence == (ev,)
    assert res.diagnostic is None
