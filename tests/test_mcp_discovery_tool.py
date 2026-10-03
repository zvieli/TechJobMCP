"""Unit and integration tests for FastMCP discover_companies tool (Milestone 6-D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from job_mcp.core.search_plane.discovery.models import (
    DiscoveredPortal,
    DiscoveryEvidence,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryTarget,
)
from job_mcp.main import MAX_DISCOVERY_TARGETS_PER_CALL, discover_companies
from job_mcp.sources.company_registry.core import validate_document


def _mock_result(
    status: DiscoveryStatus,
    family: str = "ashby",
    account: str = "acme",
    company_name: str | None = None,
    career_url: str | None = None,
    diagnostic: str | None = None,
) -> DiscoveryResult:
    ev = DiscoveryEvidence(kind="DIRECT_PROVIDER_URL", value=f"{family}:{account}", source_url=career_url)
    portal = DiscoveredPortal(source_family=family, account=account, confidence_basis=(ev,))
    target = DiscoveryTarget(company_name=company_name, career_url=career_url)
    candidates = (portal,) if status in (DiscoveryStatus.CONFIRMED, DiscoveryStatus.AMBIGUOUS) else ()
    evidence = (ev,) if status != DiscoveryStatus.MALFORMED else ()
    return DiscoveryResult(
        status=status,
        input=target,
        candidates=candidates,
        evidence=evidence,
        diagnostic=diagnostic,
    )


# ===========================================================================
# 1. Parameter Resolution & Batch Bounds
# ===========================================================================


@pytest.mark.asyncio
async def test_discover_companies_single_company_name():
    """Verify single company_name parameter is resolved and dispatched."""
    mock_res = _mock_result(DiscoveryStatus.CONFIRMED, company_name="Vercel", family="ashby", account="vercel")

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = [mock_res]

        res = await discover_companies(company_name="Vercel")

        assert res["success"] is True
        assert res["data"]["total"] == 1
        assert res["data"]["confirmed_count"] == 1
        assert "result" in res["data"]
        result_item = res["data"]["result"]
        assert result_item["status"] == "CONFIRMED"
        assert result_item["candidates"][0]["source_family"] == "ashby"
        assert result_item["candidates"][0]["account"] == "vercel"
        assert result_item["registry_config"] is not None
        assert "ashby" in result_item["registry_config"]

        mock_svc.assert_called_once()
        targets = mock_svc.call_args.args[0]
        assert len(targets) == 1
        assert targets[0].company_name == "Vercel"


@pytest.mark.asyncio
async def test_discover_companies_single_career_url():
    """Verify single career_url parameter is resolved and dispatched."""
    url = "https://jobs.smartrecruiters.com/AcmeCorp"
    mock_res = _mock_result(
        DiscoveryStatus.CONFIRMED,
        career_url=url,
        family="smartrecruiters",
        account="AcmeCorp",
    )

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = [mock_res]

        res = await discover_companies(career_url=url)

        assert res["success"] is True
        assert res["data"]["confirmed_count"] == 1
        mock_svc.assert_called_once()
        targets = mock_svc.call_args.args[0]
        assert len(targets) == 1
        assert targets[0].career_url == url


@pytest.mark.asyncio
async def test_discover_companies_batch_targets():
    """Verify batch targets list (dicts, strings, DiscoveryTarget) are parsed in order."""
    batch_input = [
        {"company_name": "Stripe"},
        {"career_url": "https://apply.workable.com/deliveroo"},
        "https://jobs.ashbyhq.com/linear",
        "Retool",
    ]

    mock_results = [
        _mock_result(DiscoveryStatus.CONFIRMED, family="ashby", account="stripe", company_name="Stripe"),
        _mock_result(
            DiscoveryStatus.CONFIRMED,
            family="workable",
            account="deliveroo",
            career_url="https://apply.workable.com/deliveroo",
        ),
        _mock_result(
            DiscoveryStatus.CONFIRMED,
            family="ashby",
            account="linear",
            career_url="https://jobs.ashbyhq.com/linear",
        ),
        _mock_result(DiscoveryStatus.NOT_FOUND, company_name="Retool"),
    ]

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = mock_results

        res = await discover_companies(targets=batch_input)

        assert res["success"] is True
        assert res["data"]["total"] == 4
        assert res["data"]["confirmed_count"] == 3
        assert len(res["data"]["results"]) == 4

        # Check order preserved
        assert res["data"]["results"][0]["candidates"][0]["account"] == "stripe"
        assert res["data"]["results"][1]["candidates"][0]["account"] == "deliveroo"
        assert res["data"]["results"][2]["candidates"][0]["account"] == "linear"
        assert res["data"]["results"][3]["status"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_discover_companies_zero_targets_rejected():
    """Verify empty target input raises ValueError."""
    with pytest.raises(ValueError, match="At least one discovery target must be provided"):
        await discover_companies()

    with pytest.raises(ValueError, match="At least one discovery target must be provided"):
        await discover_companies(targets=[])


@pytest.mark.asyncio
async def test_discover_companies_batch_limit_enforced():
    """Verify batch limit (10) allows 10 targets and rejects 11 targets."""
    # 10 targets should succeed
    targets_10 = [{"company_name": f"Company_{i}"} for i in range(MAX_DISCOVERY_TARGETS_PER_CALL)]
    mock_results_10 = [
        _mock_result(DiscoveryStatus.NOT_FOUND, company_name=f"Company_{i}")
        for i in range(MAX_DISCOVERY_TARGETS_PER_CALL)
    ]

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = mock_results_10
        res = await discover_companies(targets=targets_10)
        assert res["success"] is True
        assert res["data"]["total"] == 10

    # 11 targets should raise ValueError
    targets_11 = [{"company_name": f"Company_{i}"} for i in range(MAX_DISCOVERY_TARGETS_PER_CALL + 1)]
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        await discover_companies(targets=targets_11)


# ===========================================================================
# 2. Output Contract & Config Projection
# ===========================================================================


@pytest.mark.asyncio
async def test_discover_companies_confirmed_projects_valid_yaml():
    """Verify CONFIRMED discovery result generates valid M5 registry YAML config."""
    mock_res = _mock_result(
        DiscoveryStatus.CONFIRMED,
        family="workable",
        account="supermetrics",
        career_url="https://apply.workable.com/supermetrics",
    )

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = [mock_res]

        res = await discover_companies(career_url="https://apply.workable.com/supermetrics")

        assert res["success"] is True
        item = res["data"]["result"]
        yaml_str = item["registry_config"]
        assert yaml_str is not None

        # Verify generated YAML re-validates via CompanyRegistry.validate_document
        parsed = yaml.safe_load(yaml_str)
        assert "providers" in parsed
        assert "workable" in parsed["providers"]
        validated = validate_document(parsed)
        assert "workable" in validated.providers


@pytest.mark.asyncio
async def test_discover_companies_non_confirmed_has_no_registry_config():
    """Verify non-CONFIRMED statuses (AMBIGUOUS, UNSUPPORTED, NOT_FOUND, MALFORMED) omit registry_config."""
    statuses = [
        DiscoveryStatus.AMBIGUOUS,
        DiscoveryStatus.UNSUPPORTED,
        DiscoveryStatus.NOT_FOUND,
        DiscoveryStatus.MALFORMED,
    ]

    for st in statuses:
        mock_res = _mock_result(st, family="ashby", account="foo", company_name="Foo", diagnostic="Test diag")
        with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
            mock_svc.return_value = [mock_res]
            res = await discover_companies(company_name="Foo")
            assert res["success"] is True
            item = res["data"]["result"]
            assert item["status"] == st.value
            assert item["registry_config"] is None


@pytest.mark.asyncio
async def test_discover_companies_malformed_target_serializes_as_domain_result():
    """Verify a syntactically supplied target classified MALFORMED by M6-C serializes as a normal discovery result.

    This exercises the real (unmocked) discovery service with an SSRF-blocked career URL, which M6-C
    classifies as MALFORMED. The MCP wrapper must surface it as a domain outcome, not DISCOVERY_ERROR.
    """
    res = await discover_companies(career_url="http://127.0.0.1:8080/internal")

    assert res["success"] is True
    assert res["error_code"] is None
    item = res["data"]["result"]
    assert item["status"] == DiscoveryStatus.MALFORMED.value
    assert item["registry_config"] is None
    assert item["diagnostic"]


@pytest.mark.asyncio
async def test_discover_companies_never_writes_to_disk():
    """Verify discover_companies is strictly read-only and never writes portals.yml."""
    mock_res = _mock_result(DiscoveryStatus.CONFIRMED, family="ashby", account="vercel", company_name="Vercel")

    with patch("job_mcp.main.service_discover_companies", new_callable=AsyncMock) as mock_svc:
        mock_svc.return_value = [mock_res]

        with patch("builtins.open") as mock_open:
            res = await discover_companies(company_name="Vercel")
            assert res["success"] is True
            mock_open.assert_not_called()
