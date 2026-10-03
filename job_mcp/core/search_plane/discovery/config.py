"""Milestone 5 CompanyRegistry configuration projection for discovery results."""

from __future__ import annotations

import re
from typing import Any

import yaml

from job_mcp.core.search_plane.discovery.models import DiscoveryResult, DiscoveryStatus
from job_mcp.sources.company_registry.core import validate_document
from job_mcp.sources.company_registry.schema import _validate_company_key


def to_registry_config(
    result: DiscoveryResult,
    *,
    company_id: str | None = None,
    display_name: str | None = None,
) -> str:
    """Project a CONFIRMED DiscoveryResult into a valid M5 portals.yml YAML document.

    Raises:
        ValueError: If result status is not CONFIRMED or if candidates count != 1.
    """
    if result.status != DiscoveryStatus.CONFIRMED:
        raise ValueError(
            f"Cannot project configuration for non-CONFIRMED result (status: {result.status.value})"
        )

    if len(result.candidates) != 1:
        raise ValueError(
            f"Cannot project configuration: expected exactly 1 candidate, got {len(result.candidates)}"
        )

    candidate = result.candidates[0]
    family = candidate.source_family.lower()

    if family not in ("ashby", "smartrecruiters", "workable"):
        raise ValueError(f"Unsupported provider family for configuration projection: {family!r}")

    # Determine company id
    if company_id is not None:
        cid = _validate_company_key(company_id)
    else:
        raw_source = result.input.company_name or candidate.account
        clean_slug = re.sub(r"[^a-z0-9]+", "_", raw_source.lower()).strip("_")
        if not clean_slug:
            clean_slug = "company"
        # Ensure starts with a letter or digit
        if not clean_slug[0].isalnum():
            clean_slug = f"c_{clean_slug}"
        clean_slug = clean_slug[:64].rstrip("_")
        cid = _validate_company_key(clean_slug)

    # Determine display name
    name = (
        display_name
        or result.input.company_name
        or candidate.account.replace("-", " ").replace("_", " ").title()
    ).strip()
    if not name:
        name = cid.replace("_", " ").title()

    company_entry: dict[str, Any] = {
        "id": cid,
        "name": name,
    }

    if family == "ashby":
        company_entry["board_name"] = candidate.account
    elif family == "smartrecruiters":
        company_entry["company_identifier"] = candidate.account
    elif family == "workable":
        company_entry["account_subdomain"] = candidate.account

    doc: dict[str, Any] = {
        "version": 1,
        "providers": {
            family: {
                "companies": [company_entry],
            },
        },
    }

    # Validate against M5 CompanyRegistry schema to guarantee 100% compliance
    validate_document(doc)

    return yaml.safe_dump(doc, sort_keys=False)
