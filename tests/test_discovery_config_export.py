"""Tests for discovery M5 CompanyRegistry configuration projection."""

from __future__ import annotations

import pytest
import yaml

from job_mcp.core.search_plane.discovery.config import to_registry_config
from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)
from job_mcp.sources.company_registry.core import validate_document
from job_mcp.sources.company_registry.entries import (
    AshbyCompany,
    SmartRecruitersCompany,
    WorkableCompany,
)


def test_ashby_config_projection() -> None:
    """A CONFIRMED Ashby result projects to valid M5 YAML matching AshbyEntry."""
    target = DiscoveryTarget(company_name="Acme Corp", career_url="https://jobs.ashbyhq.com/acme")
    ev = DiscoveryEvidence(kind="DIRECT_PROVIDER_URL", value="ashby:acme")
    portal = DiscoveredPortal(source_family="ashby", account="acme", confidence_basis=(ev,))

    res = DiscoveryResult(
        status=DiscoveryStatus.CONFIRMED,
        input=target,
        candidates=(portal,),
        evidence=(ev,),
    )

    yaml_text = to_registry_config(res)
    doc = yaml.safe_load(yaml_text)

    # Validates cleanly via M5 CompanyRegistry schema
    validated = validate_document(doc)
    assert "ashby" in validated.providers
    entry = validated.providers["ashby"].companies[0].to_entry()
    assert isinstance(entry, AshbyCompany)
    assert entry.name == "Acme Corp"
    assert entry.board_name == "acme"
    assert entry.enabled is True


def test_smartrecruiters_config_projection() -> None:
    """A CONFIRMED SmartRecruiters result projects to valid M5 YAML matching SmartRecruitersEntry."""
    target = DiscoveryTarget(company_name="Acme Inc")
    ev = DiscoveryEvidence(kind="VERIFICATION_PROBE", value="smartrecruiters:acme-corp confirmed")
    portal = DiscoveredPortal(
        source_family="smartrecruiters", account="acme-corp", confidence_basis=(ev,)
    )

    res = DiscoveryResult(
        status=DiscoveryStatus.CONFIRMED,
        input=target,
        candidates=(portal,),
        evidence=(ev,),
    )

    yaml_text = to_registry_config(res, company_id="acme_custom", display_name="Acme Customized")
    doc = yaml.safe_load(yaml_text)

    validated = validate_document(doc)
    assert "smartrecruiters" in validated.providers
    entry = validated.providers["smartrecruiters"].companies[0].to_entry()
    assert isinstance(entry, SmartRecruitersCompany)
    assert entry.name == "Acme Customized"
    assert entry.company_identifier == "acme-corp"


def test_workable_config_projection() -> None:
    """A CONFIRMED Workable result projects to valid M5 YAML matching WorkableEntry."""
    target = DiscoveryTarget(career_url="https://apply.workable.com/acme-tech/")
    ev = DiscoveryEvidence(kind="DIRECT_PROVIDER_URL", value="workable:acme-tech")
    portal = DiscoveredPortal(source_family="workable", account="acme-tech", confidence_basis=(ev,))

    res = DiscoveryResult(
        status=DiscoveryStatus.CONFIRMED,
        input=target,
        candidates=(portal,),
        evidence=(ev,),
    )

    yaml_text = res.to_registry_config()
    doc = yaml.safe_load(yaml_text)

    validated = validate_document(doc)
    assert "workable" in validated.providers
    entry = validated.providers["workable"].companies[0].to_entry()
    assert isinstance(entry, WorkableCompany)
    assert entry.account_subdomain == "acme-tech"


def test_non_confirmed_results_cannot_project() -> None:
    """AMBIGUOUS, NOT_FOUND, UNSUPPORTED, and MALFORMED results refuse to project config."""
    target = DiscoveryTarget(company_name="Ambiguous Corp")

    # Ambiguous
    res_ambiguous = DiscoveryResult(
        status=DiscoveryStatus.AMBIGUOUS,
        input=target,
        candidates=(
            DiscoveredPortal(source_family="ashby", account="a1"),
            DiscoveredPortal(source_family="workable", account="w1"),
        ),
    )
    with pytest.raises(ValueError, match="Cannot project configuration for non-CONFIRMED"):
        to_registry_config(res_ambiguous)

    # Not Found
    res_not_found = DiscoveryResult(status=DiscoveryStatus.NOT_FOUND, input=target)
    with pytest.raises(ValueError, match="Cannot project configuration for non-CONFIRMED"):
        to_registry_config(res_not_found)

    # Unsupported
    res_unsupported = DiscoveryResult(status=DiscoveryStatus.UNSUPPORTED, input=target)
    with pytest.raises(ValueError, match="Cannot project configuration for non-CONFIRMED"):
        to_registry_config(res_unsupported)

    # Malformed
    res_malformed = DiscoveryResult(status=DiscoveryStatus.MALFORMED, input=target)
    with pytest.raises(ValueError, match="Cannot project configuration for non-CONFIRMED"):
        to_registry_config(res_malformed)


def test_no_silent_persistence() -> None:
    """Generating configuration must not touch filesystem or modify portals.yml."""
    import os

    target = DiscoveryTarget(company_name="Acme")
    portal = DiscoveredPortal(source_family="ashby", account="acme")
    res = DiscoveryResult(status=DiscoveryStatus.CONFIRMED, input=target, candidates=(portal,))

    # Record mod times or existence of portals.yml if present
    exists_before = os.path.exists("portals.yml")
    mtime_before = os.path.getmtime("portals.yml") if exists_before else None

    # Call to_registry_config multiple times
    _ = to_registry_config(res)
    _ = res.to_registry_config()

    exists_after = os.path.exists("portals.yml")
    assert exists_after == exists_before
    if exists_after:
        assert os.path.getmtime("portals.yml") == mtime_before
