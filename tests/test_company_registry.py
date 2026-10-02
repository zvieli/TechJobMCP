"""Milestone 5 verification of the configuration-driven company registry.

Covers the registry contract (typed validation, deterministic merge, zero-config
compatibility) and the acceptance requirement that companies can be added,
overridden, enabled and disabled for existing provider families through
configuration only, with zero provider source diffs.

Deterministic and zero-cost: no network, no paid APIs, no real ATS endpoints.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from job_mcp.sources.company_registry import (
    CompanyRegistryError,
    catalog,
    configure,
    discover_config_paths,
    effective_catalog,
    enabled_catalog,
    get_registry,
    resolve,
    supported_providers,
    validate_document,
)
from job_mcp.sources.company_registry import defaults as builtin_defaults
from job_mcp.sources.company_registry.core import (
    _BUILTIN_CATALOGS,
    BUILTINS_ONLY_ENV_VAR,
)
from job_mcp.sources.company_registry.entries import (
    DIRECT_TECH_DEFAULT_QUERY,
    DirectTechCompany,
    EightfoldCompany,
    GreenhouseCompany,
)
from job_mcp.sources.enterprise.direct_tech import (
    DIRECT_TECH_COMPANIES,
    DirectTechSource,
)
from job_mcp.sources.enterprise.workday import WORKDAY_COMPANIES, WorkdaySource
from job_mcp.sources.public.ashby import ASHBY_COMPANIES, AshbySource
from job_mcp.sources.public.eightfold import EIGHTFOLD_COMPANIES, EightfoldAISource
from job_mcp.sources.public.greenhouse import GREENHOUSE_COMPANIES, GreenhouseSource
from job_mcp.sources.public.lever import LEVER_COMPANIES, LeverSource
from job_mcp.sources.public.smartrecruiters import (
    SMARTRECRUITERS_COMPANIES,
    SmartRecruitersSource,
)

ROOT = Path(__file__).resolve().parents[1]

# Each family is exercised through the same abstraction. `new` carries exactly
# the fields needed to declare a brand-new company for that family.
FAMILIES: dict[str, dict[str, Any]] = {
    "ashby": {
        "source": AshbySource,
        "catalog": ASHBY_COMPANIES,
        "new": {"name": "Config Only Ashby", "board_name": "configonlyashby"},
        # Ashby assigns the catalog wholesale, so it keys by registry id.
        "probe_key": "configonlycorp",
    },
    "smartrecruiters": {
        "source": SmartRecruitersSource,
        "catalog": SMARTRECRUITERS_COMPANIES,
        "new": {
            "name": "Config Only SmartRecruiters",
            "company_identifier": "configonlysmartrecruiters",
        },
        # SmartRecruiters assigns the catalog wholesale, so it keys by registry id.
        "probe_key": "configonlycorp",
    },
    "greenhouse": {
        "source": GreenhouseSource,
        "catalog": GREENHOUSE_COMPANIES,
        "new": {"name": "Config Only Corp", "board_token": "configonlycorp"},
        # Greenhouse assigns the catalog wholesale, so it keys by registry id.
        "probe_key": "configonlycorp",
    },
    "lever": {
        "source": LeverSource,
        "catalog": LEVER_COMPANIES,
        "new": {"name": "Config Only Lever", "slug": "configonlylever"},
        # Lever assigns the catalog wholesale, so it keys by registry id.
        "probe_key": "configonlycorp",
    },
    "eightfold": {
        "source": EightfoldAISource,
        "catalog": EIGHTFOLD_COMPANIES,
        "new": {
            "name": "Config Only Fold",
            "hostname": "configonly.eightfold.example",
            "domain": "configonly.example",
        },
        # Eightfold keys its internal map by domain, not registry id.
        "probe_key": "configonly.example",
    },
    "direct_tech": {
        "source": DirectTechSource,
        "catalog": DIRECT_TECH_COMPANIES,
        "new": {
            "name": "Config Only Direct",
            "search_url": "https://careers.configonly.example/api/jobs",
        },
        # DirectTech indexes internally by provider_id, which is the registry id.
        "probe_key": "configonlycorp",
    },
    "workday": {
        "source": WorkdaySource,
        "catalog": WORKDAY_COMPANIES,
        "new": {"name": "Config Only Day", "wd_company": "configonly"},
        # Workday keys its internal map by tenant, not registry id.
        "probe_key": "configonly",
    },
}

PROVIDERS = sorted(FAMILIES)
NEW_ID = "configonlycorp"


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch, tmp_path):
    """Keep every test independent of ambient config and the cached registry.

    Runs each test from an isolated empty directory so discovery can never
    observe a developer's own project ``portals.yml``; zero-config assertions
    must not depend on how the local checkout happens to be configured.
    """
    monkeypatch.delenv("COMPANY_REGISTRY_PATH", raising=False)
    monkeypatch.delenv(BUILTINS_ONLY_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    configure([])
    yield
    monkeypatch.delenv(BUILTINS_ONLY_ENV_VAR, raising=False)
    configure([])


def _config_file(tmp_path: Path, document: dict[str, Any], name: str = "portals.yml") -> Path:
    """Write ``document`` as YAML and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _company_entries(source: Any) -> list[Any]:
    """Return the companies a provider instance actually resolved."""
    return list(source._companies.values())


# ---------------------------------------------------------------------------
# Zero-config compatibility
# ---------------------------------------------------------------------------


def test_registry_manages_exactly_the_company_driven_families() -> None:
    assert set(supported_providers()) == set(FAMILIES)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_zero_config_effective_catalog_equals_builtin_defaults(provider: str) -> None:
    """With no configuration the effective catalog *is* the curated built-in set."""
    assert resolve([]).catalog(provider) == _BUILTIN_CATALOGS[provider]
    assert catalog(provider) == _BUILTIN_CATALOGS[provider]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_moving_the_catalog_changed_no_field_and_no_ordering(provider: str) -> None:
    """Regression guard for the extraction itself: values and order identical."""
    expected = FAMILIES[provider]["catalog"]
    actual = resolve([]).catalog(provider)
    assert list(actual) == list(expected)
    for company_id, entry in expected.items():
        assert actual[company_id] == entry


@pytest.mark.parametrize("provider", PROVIDERS)
def test_zero_config_provider_constructs_the_builtin_companies(provider: str) -> None:
    spec = FAMILIES[provider]
    source = spec["source"]()
    assert len(_company_entries(source)) == len(spec["catalog"])
    assert {entry.name for entry in _company_entries(source)} == {
        entry.name for entry in spec["catalog"].values()
    }


def test_builtin_catalogs_are_never_mutated_by_configuration(tmp_path: Path) -> None:
    before = dict(_BUILTIN_CATALOGS["greenhouse"])
    resolve(
        [
            _config_file(
                tmp_path,
                {"providers": {"greenhouse": {"companies": [{"id": "lightricks", "name": "Renamed"}]}}},
            )
        ]
    )
    assert _BUILTIN_CATALOGS["greenhouse"] == before
    assert builtin_defaults.GREENHOUSE_COMPANIES["lightricks"].name == "Lightricks"


# ---------------------------------------------------------------------------
# Config-only company addition — the Milestone 5 acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_new_company_reaches_the_provider_via_config_only(tmp_path: Path, provider: str) -> None:
    spec = FAMILIES[provider]
    module = inspect.getmodule(spec["source"])
    source_text_before = Path(module.__file__).read_text(encoding="utf-8")

    path = _config_file(
        tmp_path,
        {"providers": {provider: {"companies": [{"id": NEW_ID, **spec["new"]}]}}},
    )
    configure([path])

    registry_entry = catalog(provider)[NEW_ID]
    assert registry_entry.name == spec["new"]["name"]

    source = spec["source"]()
    names = {getattr(entry, "name", None) for entry in _company_entries(source)}
    assert spec["new"]["name"] in names, f"{provider} did not receive the configured company"
    # Each family keys its internal map its own way (registry id, domain or
    # tenant); assert the entry is reachable through that key.
    assert spec["probe_key"] in source._companies, (
        f"{provider} did not index the configured company under {spec['probe_key']!r}"
    )

    # Adding a company required zero provider source change.
    assert Path(module.__file__).read_text(encoding="utf-8") == source_text_before


@pytest.mark.parametrize("provider", PROVIDERS)
def test_adding_a_company_needs_no_provider_code_change(tmp_path: Path, provider: str) -> None:
    """Constructor signature and module source are untouched by configuration."""
    spec = FAMILIES[provider]
    module = inspect.getmodule(spec["source"])
    signature_before = str(inspect.signature(spec["source"].__init__))
    source_text_before = Path(module.__file__).read_text(encoding="utf-8")

    configure([_config_file(tmp_path, {"providers": {provider: {"companies": [{"id": NEW_ID, **spec["new"]}]}}})])

    assert str(inspect.signature(spec["source"].__init__)) == signature_before
    assert Path(module.__file__).read_text(encoding="utf-8") == source_text_before
    assert NEW_ID in catalog(provider)


def test_greenhouse_new_company_is_queried_through_existing_provider_logic(tmp_path: Path) -> None:
    """The configured company flows into the URL the provider would request."""
    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "greenhouse": {
                            "companies": [
                                {"id": "brandnewco", "name": "Brand New Co", "board_token": "bnc-token"}
                            ]
                        }
                    }
                },
            )
        ]
    )
    source = GreenhouseSource()
    assert source._companies["brandnewco"].board_token == "bnc-token"


def test_lever_new_company_is_queried_through_existing_provider_logic(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "lever": {"companies": [{"id": "brandnewlever", "name": "New", "slug": "bnl"}]}
                    }
                },
            )
        ]
    )
    source = LeverSource()
    assert source._companies["brandnewlever"].slug == "bnl"


def test_third_family_eightfold_uses_the_same_abstraction(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "eightfold": {
                            "companies": [
                                {
                                    "id": "foldco",
                                    "name": "Fold Co",
                                    "hostname": "fold.eightfold.example",
                                    "domain": "fold.example",
                                    "locations": ["Tel Aviv"],
                                }
                            ]
                        }
                    }
                },
            )
        ]
    )
    entry = EightfoldAISource()._companies["fold.example"]
    assert entry.hostname == "fold.eightfold.example"
    assert entry.get_search_url() == "https://fold.eightfold.example/api/pcsx/search"


def test_fourth_family_workday_uses_the_same_abstraction(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "workday": {
                            "companies": [{"id": "dayco", "name": "Day Co", "wd_company": "daycorp"}]
                        }
                    }
                },
            )
        ]
    )
    entry = WorkdaySource()._companies["daycorp"]
    assert entry.get_cxs_url() == "https://daycorp.wd1.myworkdayjobs.com/wday/cxs/daycorp/External/jobs"


# ---------------------------------------------------------------------------
# Enable / disable / override
# ---------------------------------------------------------------------------


def test_builtin_company_can_be_disabled_without_code_change(tmp_path: Path) -> None:
    configure(
        [_config_file(tmp_path, {"providers": {"greenhouse": {"companies": [{"id": "jfrog", "enabled": False}]}}})]
    )
    entries = catalog("greenhouse")
    assert "jfrog" in entries, "disabling must not delete the entry"
    assert entries["jfrog"].enabled is False
    assert "jfrog" not in enabled_catalog("greenhouse")
    # `enabled_catalog` filters on the flag, so it always agrees with the map.
    assert set(enabled_catalog("greenhouse")) == {
        key for key, entry in entries.items() if entry.enabled
    }
    assert "jfrog" not in set(enabled_catalog("greenhouse"))


def test_disabled_entry_can_be_re_enabled_by_a_later_document(tmp_path: Path) -> None:
    off = _config_file(
        tmp_path, {"providers": {"lever": {"companies": [{"id": "redis", "enabled": False}]}}}, "a.yml"
    )
    on = _config_file(
        tmp_path, {"providers": {"lever": {"companies": [{"id": "redis", "enabled": True}]}}}, "b.yml"
    )
    assert resolve([off]).catalog("lever")["redis"].enabled is False
    assert resolve([off, on]).catalog("lever")["redis"].enabled is True


def test_override_changes_only_the_fields_present_in_configuration(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {"providers": {"workday": {"companies": [{"id": "cisco", "wd_version": 2, "enabled": False}]}}},
            )
        ]
    )
    entry = catalog("workday")["cisco"]
    built_in = builtin_defaults.WORKDAY_COMPANIES["cisco"]
    assert entry.wd_version == 2
    assert entry.enabled is False
    assert entry.name == built_in.name
    assert entry.wd_company == built_in.wd_company
    assert entry.wd_suffix == built_in.wd_suffix


def test_later_document_wins_and_reverse_order_reverses_the_result(tmp_path: Path) -> None:
    lower = _config_file(
        tmp_path, {"providers": {"greenhouse": {"companies": [{"id": "yotpo", "name": "Lower"}]}}}, "a.yml"
    )
    higher = _config_file(
        tmp_path, {"providers": {"greenhouse": {"companies": [{"id": "yotpo", "name": "Higher"}]}}}, "b.yml"
    )
    assert resolve([lower, higher]).catalog("greenhouse")["yotpo"].name == "Higher"
    assert resolve([higher, lower]).catalog("greenhouse")["yotpo"].name == "Lower"


def test_identical_duplicates_collapse_across_and_within_documents(tmp_path: Path) -> None:
    entry = {"id": "jfrog", "name": "JFrog", "board_token": "jfrog", "enabled": True}
    first = _config_file(tmp_path, {"providers": {"greenhouse": {"companies": [entry]}}}, "a.yml")
    second = _config_file(tmp_path, {"providers": {"greenhouse": {"companies": [dict(entry)]}}}, "b.yml")
    registry = resolve([first, second])
    assert list(registry.catalog("greenhouse")).count("jfrog") == 1

    same_document = _config_file(
        tmp_path, {"providers": {"greenhouse": {"companies": [entry, dict(entry)]}}}
    )
    assert list(resolve([same_document]).catalog("greenhouse")).count("jfrog") == 1


def test_duplicate_conflicting_entries_are_rejected(tmp_path: Path) -> None:
    document = {
        "providers": {
            "greenhouse": {
                "companies": [
                    {"id": "clashco", "name": "First", "board_token": "a"},
                    {"id": "clashco", "name": "Second", "board_token": "a"},
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="duplicate conflicting"):
        resolve([_config_file(tmp_path, document)])


def test_new_companies_are_appended_after_builtins_in_configuration_order(tmp_path: Path) -> None:
    document = {
        "providers": {
            "greenhouse": {
                "companies": [
                    {"id": "zeta", "name": "Zeta", "board_token": "z"},
                    {"id": "alpha", "name": "Alpha", "board_token": "a"},
                ]
            }
        }
    }
    ids = list(resolve([_config_file(tmp_path, document)]).catalog("greenhouse"))
    assert ids[: len(GREENHOUSE_COMPANIES)] == list(GREENHOUSE_COMPANIES)
    assert ids[-2:] == ["zeta", "alpha"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (
            {
                "providers": {
                    "greenhouse": {
                        "companies": [
                            {"id": "x", "name": "X", "board_token": "x", "colour": "red"}
                        ]
                    }
                }
            },
            "colour",
        ),
        ({"providers": {"greenhouse": {"unexpected": True}}}, "unexpected"),
        ({"unexpected_top": 1}, "unexpected_top"),
        ({"version": 2}, "version"),
        ({"providers": {"not_a_family": {"companies": []}}}, "unsupported provider"),
        ({"providers": {"greenhouse": {"companies": [{"name": "missing id"}]}}}, "explicit 'id'"),
        ({"providers": {"greenhouse": {"companies": "not-a-list"}}}, "companies"),
        (
            {"providers": {"greenhouse": {"companies": [{"id": "Bad ID!", "name": "X", "board_token": "x"}]}}},
            "company id",
        ),
        ({"providers": {"Greenhouse": {"companies": []}}}, "unsupported provider"),
        ({"providers": {"greenhouse": {"companies": [{"id": "ok", "name": "", "board_token": "b"}]}}}, "name"),
        ({"providers": {"greenhouse": {"companies": [{"id": "ok", "board_token": "b"}]}}}, "name"),
        ({"providers": {"greenhouse": {"companies": [{"id": "ok", "name": "N"}]}}}, "board_token"),
        ({"providers": {"greenhouse": []}}, "provider block must be a mapping"),
    ],
)
def test_invalid_configuration_is_rejected_before_execution(document: dict, expected: str) -> None:
    with pytest.raises(CompanyRegistryError) as excinfo:
        validate_document(document, source="inline")
    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("provider", "entry_id", "payload", "expected"),
    [
        ("eightfold", "nvidia", {"hostname": "http://evil.example/x"}, "hostname"),
        ("eightfold", "nvidia", {"hostname": "host/path"}, "hostname"),
        ("greenhouse", "jfrog", {"name": ""}, "name"),
        ("greenhouse", "jfrog", {"board_token": ""}, "board_token"),
        ("lever", "redis", {"slug": ""}, "slug"),
        ("workday", "intel", {"wd_company": ""}, "wd_company"),
        ("workday", "intel", {"wd_suffix": ""}, "wd_suffix"),
        ("direct_tech", "google", {"name": ""}, "name"),
        ("direct_tech", "google", {"search_url": "https://@"}, "search_url"),
        ("direct_tech", "google", {"search_url": "not-a-url"}, "search_url"),
        ("workday", "intel", {"base_url": "https://@"}, "base_url"),
        ("lever", "redis", {"enabled": None}, "enabled"),
        ("lever", "redis", {"slug": None}, "slug"),
        ("greenhouse", "jfrog", {"colour": "red"}, "colour"),
    ],
)
def test_malformed_builtin_override_is_rejected_with_entry_parity(
    provider: str, entry_id: str, payload: dict[str, Any], expected: str
) -> None:
    """A built-in override cannot accept data that a new company would refuse.

    Override models share the constrained field types of the complete entry
    models, so partial updates validate as strictly as additions.
    """
    document = {"providers": {provider: {"companies": [{"id": entry_id, **payload}]}}}
    with pytest.raises(CompanyRegistryError, match=expected):
        validate_document(document, source="override-parity")


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("direct_tech", {"default_query": "software engineer"}),
        ("direct_tech", {"default_query": ""}),
        ("direct_tech", {"default_location": "Tel Aviv"}),
        ("eightfold", {"locations": []}),
        ("workday", {"wd_version": 7}),
        ("greenhouse", {"name": "Renamed", "enabled": False}),
        ("lever", {"enabled": True}),
    ],
)
def test_legitimate_builtin_overrides_are_still_accepted(
    tmp_path: Path, provider: str, payload: dict[str, Any]
) -> None:
    """Strictness must not reject honest partial updates."""
    entry_id = next(iter(_BUILTIN_CATALOGS[provider]))
    document = {"providers": {provider: {"companies": [{"id": entry_id, **payload}]}}}
    registry = resolve([_config_file(tmp_path, document)])
    merged = registry.catalog(provider)[entry_id]
    for field_name, value in payload.items():
        assert getattr(merged, field_name) == value


def test_explicit_null_clears_a_nullable_entry_field(tmp_path: Path) -> None:
    """`field: null` must mean "clear it", not "inherit the built-in value".

    Only the fields whose provider dataclass is genuinely nullable accept null.
    """
    document = {
        "providers": {
            "eightfold": {"companies": [{"id": "micron", "filter_distance": None}]},
            "workday": {"companies": [{"id": "nvidia", "base_url": None}]},
        }
    }
    registry = resolve([_config_file(tmp_path, document)])
    assert builtin_defaults.EIGHTFOLD_COMPANIES["micron"].filter_distance == "16"
    assert registry.catalog("eightfold")["micron"].filter_distance is None
    assert registry.catalog("workday")["nvidia"].base_url is None


def test_omitting_a_nullable_field_inherits_the_builtin_value(tmp_path: Path) -> None:
    document = {"providers": {"eightfold": {"companies": [{"id": "micron", "name": "Micron"}]}}}
    registry = resolve([_config_file(tmp_path, document)])
    assert registry.catalog("eightfold")["micron"].filter_distance == "16"


def test_nullable_and_non_nullable_null_parity_with_entry_shapes() -> None:
    """Eightfold's distance accepts null; Lever's slug does not."""
    assert validate_document(
        {"providers": {"eightfold": {"companies": [{"id": "micron", "filter_distance": None}]}}}
    )
    with pytest.raises(CompanyRegistryError, match="slug"):
        validate_document({"providers": {"lever": {"companies": [{"id": "redis", "slug": None}]}}})


def test_provider_config_refuses_untyped_company_mappings() -> None:
    """The typed boundary is enforced even when models are built directly.

    `RegistryConfig`'s before-validator is what turns YAML mappings into typed
    models; constructing `ProviderConfig` by hand must not accept raw dicts, or
    the merge step would receive unvalidated data.
    """
    from pydantic import ValidationError

    from job_mcp.sources.company_registry.schema import ProviderConfig

    for raw in ({"id": "jfrog", "enabled": False}, "not-even-a-mapping", 42):
        with pytest.raises(ValidationError, match="validated entry models"):
            ProviderConfig(companies=[raw])


def test_partially_shaped_new_company_reports_configuration_error(tmp_path: Path) -> None:
    """An override-shaped payload for an unknown id is a config error, not a crash."""
    from job_mcp.sources.company_registry.core import effective_catalog
    from job_mcp.sources.company_registry.schema import (
        GreenhouseOverride,
        ProviderConfig,
        RegistryConfig,
    )

    config = RegistryConfig(providers={"greenhouse": ProviderConfig(companies=[GreenhouseOverride(id="ghostco")])})
    with pytest.raises(CompanyRegistryError, match="must supply every required field"):
        effective_catalog([config])


# ---------------------------------------------------------------------------
# Round-2 P1 regression guards
# ---------------------------------------------------------------------------


NULLABLE_OVERRIDE_FIELDS: set[tuple[str, str]] = {
    ("eightfold", "filter_distance"),
    ("workday", "base_url"),
}


def _override_payload(provider: str, entry_id: str, field: str) -> dict[str, Any]:
    """One-field partial override of a built-in company."""
    return {"providers": {provider: {"companies": [{"id": entry_id, field: None}]}}}


def test_every_non_nullable_override_field_rejects_explicit_null() -> None:
    """Derived from the models themselves, so coverage cannot silently drift.

    Asserting over `OVERRIDE_MODELS.model_fields` rather than a hand-written list
    means adding an override field automatically gains a null-parity assertion,
    and the test's coverage claim always matches what it actually checks.
    """
    from job_mcp.sources.company_registry.schema import OVERRIDE_MODELS

    checked = 0
    for provider, model in OVERRIDE_MODELS.items():
        entry_id = next(iter(_BUILTIN_CATALOGS[provider]))
        for field in model.model_fields:
            if field == "id":
                continue
            document = _override_payload(provider, entry_id, field)
            nullable = (provider, field) in NULLABLE_OVERRIDE_FIELDS
            if nullable:
                validate_document(document, source=f"nullable:{field}")
                continue
            with pytest.raises(CompanyRegistryError, match=field):
                validate_document(document, source=f"null-parity:{field}")
            checked += 1
    # The two nullable fields are covered above; every other override field is a
    # rejection case. Sanity-check the count so an accidental model edit is loud.
    total_fields = sum(
        len(model.model_fields) - 1 for model in OVERRIDE_MODELS.values()
    )
    assert checked == total_fields - len(NULLABLE_OVERRIDE_FIELDS)
    # Floor on the count: a model edit that quietly deletes fields must fail here
    # rather than shrink the claim to nothing.
    assert checked >= 20, f"suspiciously few override fields found: {checked}"


@pytest.mark.parametrize(("provider", "entry_id", "field", "value"), [
    ("direct_tech", "google", "search_url", None),
    ("workday", "intel", "wd_version", None),
    ("greenhouse", "jfrog", "name", None),
    ("lever", "redis", "slug", None),
])
def test_explicit_null_override_fails_before_resolution(
    tmp_path: Path, provider: str, entry_id: str, field: str, value: Any
) -> None:
    """Rejection happens at validation, so the null never reaches a provider entry."""
    document = _override_payload(provider, entry_id, field)
    with pytest.raises(CompanyRegistryError):
        resolve([_config_file(tmp_path, document, f"null-{provider}-{field}.yml")])


def test_nullable_override_fields_still_accept_explicit_null(tmp_path: Path) -> None:
    """The two genuinely nullable entry fields keep their clear-to-null contract.

    Note the two differ in effect: `workday.base_url = null` changes request
    behaviour, while `eightfold.filter_distance = null` clears only the stored
    value because the provider applies its own `or "16"` fallback at fetch time.
    """
    document = {
        "providers": {
            "eightfold": {"companies": [{"id": "micron", "filter_distance": None}]},
            "workday": {"companies": [{"id": "nvidia", "base_url": None}]},
        }
    }
    registry = resolve([_config_file(tmp_path, document, "nullable.yml")])
    assert registry.catalog("eightfold")["micron"].filter_distance is None
    assert registry.catalog("workday")["nvidia"].base_url is None


@pytest.mark.parametrize("bad_base_url", [
    "https://example.com/#fragment",
    "https://example.com/?tenant=1",
    "https://example.com/search?a=1#frag",
])
def test_workday_base_url_rejects_query_and_fragment(bad_base_url: str) -> None:
    """Providers append paths to `base_url`, so query/fragment values break URLs.

    `get_cxs_url()` concatenates onto the base, so `.../#frag` would produce
    `.../#frag/wday/cxs/...` and target the wrong endpoint. Rejecting it during
    validation keeps the failure pre-fetch and actionable.
    """
    for payload in (
        {"id": "zz", "name": "Z", "wd_company": "z", "base_url": bad_base_url},
        {"id": "intel", "base_url": bad_base_url},
    ):
        with pytest.raises(CompanyRegistryError, match="base_url"):
            validate_document({"providers": {"workday": {"companies": [payload]}}})


@pytest.mark.parametrize("good_base_url", [
    "https://intel.wd1.myworkdayjobs.com",
    "https://x.example.com/wd",
    "https://x.example.com:443/base",
])
def test_valid_workday_base_urls_still_accepted(good_base_url: str) -> None:
    payload = {"id": "q", "name": "Q", "wd_company": "q", "base_url": good_base_url}
    assert validate_document({"providers": {"workday": {"companies": [payload]}}})


# Company identifiers that Milestone 5 moved from Python literals into config are
# interpolated straight into request URLs, so they need validation: an unchecked
# value can move the request host (`workday.wd_company`) or escape the intended
# path (`lever.slug`, `greenhouse.board_token`, `workday.wd_suffix`).
@pytest.mark.parametrize(
    ("provider", "payload", "expected"),
    [
        # lever.slug -> https://api.lever.co/v0/postings/{slug}  (path position)
        ("lever", {"id": "evil", "name": "E", "slug": "evil.example/"}, "slug"),
        ("lever", {"id": "q", "name": "Q", "slug": "a?x=1"}, "slug"),
        ("lever", {"id": "h", "name": "H", "slug": "a#b"}, "slug"),
        ("lever", {"id": "s", "name": "S", "slug": "a b"}, "slug"),
        ("lever", {"id": "enc", "name": "E", "slug": "a%2Fb"}, "slug"),
        ("lever", {"id": "redis", "slug": "x?y=1"}, "slug"),
        # workday.wd_company -> https://{wd_company}.wd{n}.myworkdayjobs.com (host)
        ("workday", {"id": "evil", "name": "E", "wd_company": "evil.example/#"}, "wd_company"),
        ("workday", {"id": "w2", "name": "W", "wd_company": "a?x=1"}, "wd_company"),
        ("workday", {"id": "w3", "name": "W", "wd_company": "a/b"}, "wd_company"),
        ("workday", {"id": "intel", "wd_company": "x#y"}, "wd_company"),
        # Host labels are case-insensitive in DNS, but the provider lowercases
        # nothing, so keep the configured tenant strictly lowercase.
        ("workday", {"id": "u", "name": "U", "wd_company": "UPPER"}, "wd_company"),
        ("workday", {"id": "dots", "name": "D", "wd_company": "a.b"}, "wd_company"),
        # greenhouse.board_token -> .../boards/{token}/jobs  (path position)
        ("greenhouse", {"id": "g2", "name": "G", "board_token": "../../evil"}, "board_token"),
        ("greenhouse", {"id": "g3", "name": "G", "board_token": "a?x=1"}, "board_token"),
        ("greenhouse", {"id": "g4", "name": "G", "board_token": "a#b"}, "board_token"),
        ("greenhouse", {"id": "g5", "name": "G", "board_token": "a b"}, "board_token"),
        ("greenhouse", {"id": "jfrog", "board_token": "x/y"}, "board_token"),
        # smartrecruiters.company_identifier -> path position
        ("smartrecruiters", {"id": "sr2", "name": "SR", "company_identifier": "../../evil"}, "company_identifier"),
        ("smartrecruiters", {"id": "sr3", "name": "SR", "company_identifier": "a?x=1"}, "company_identifier"),
        ("smartrecruiters", {"id": "sr4", "name": "SR", "company_identifier": "a#b"}, "company_identifier"),
        ("smartrecruiters", {"id": "sr5", "name": "SR", "company_identifier": "a b"}, "company_identifier"),
        ("smartrecruiters", {"id": "smartrecruiters", "company_identifier": "x/y"}, "company_identifier"),
        # workday.wd_suffix -> path position
        ("workday", {"id": "w4", "name": "W", "wd_company": "ok", "wd_suffix": "a/b"}, "wd_suffix"),
        ("workday", {"id": "w5", "name": "W", "wd_company": "ok", "wd_suffix": "a?x=1"}, "wd_suffix"),
        ("workday", {"id": "cisco", "wd_suffix": "x#y"}, "wd_suffix"),
    ],
)
def test_host_and_path_position_identifiers_are_constrained(
    provider: str, payload: dict[str, Any], expected: str
) -> None:
    """A company identifier must not be able to move the request host or path."""
    with pytest.raises(CompanyRegistryError, match=expected):
        validate_document({"providers": {provider: {"companies": [payload]}}})


def test_eightfold_hostname_requires_a_full_hostname() -> None:
    """`hostname` forms the request host, so degenerate and suffix values fail.

    A leading dot such as `.evil.com` would append onto an existing trust
    boundary (`https://.evil.com/api/...` resolves to `evil.com`, and a value
    like `attacker.eightfold.ai` is a different tenant), so at least two
    non-empty labels are required. Every curated hostname still validates.
    """
    bad = [
        "..",
        ".",
        "a..b",
        ".evil.com",
        "evil.com.",
        "-a.example",
        "a-.example",
        "singlelabel",
        "a b.example",
        "a:b.example",
        "%65vil.example",
    ]
    for value in bad:
        with pytest.raises(CompanyRegistryError, match="hostname"):
            validate_document(
                {
                    "providers": {
                        "eightfold": {
                            "companies": [
                                {"id": "h", "name": "H", "hostname": value, "domain": "d.example"}
                            ]
                        }
                    }
                }
            )
    # Overrides are held to the same rule.
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(
            {"providers": {"eightfold": {"companies": [{"id": "nvidia", "hostname": ".evil.com"}]}}}
        )
    for company_id, entry in builtin_defaults.EIGHTFOLD_COMPANIES.items():
        validate_document(
            {
                "providers": {
                    "eightfold": {
                        "companies": [
                            {
                                "id": company_id,
                                "name": entry.name,
                                "hostname": entry.hostname,
                                "domain": entry.domain,
                            }
                        ]
                    }
                }
            }
        )


def test_legitimate_host_and_path_identifiers_are_accepted() -> None:
    """Charset tightening must not reject real-world identifier shapes."""
    good = [
        ("lever", {"id": "ok", "name": "O", "slug": "papayaglobal"}),
        ("lever", {"id": "ok2", "name": "O", "slug": "khealth"}),
        ("workday", {"id": "ok3", "name": "O", "wd_company": "nvidia"}),
        (
            "workday",
            {"id": "ok4", "name": "O", "wd_company": "dell", "wd_suffix": "jobs-and-careers"},
        ),
        (
            "workday",
            {"id": "ok5", "name": "O", "wd_company": "cisco", "wd_suffix": "Cisco_Careers"},
        ),
        ("greenhouse", {"id": "ok6", "name": "O", "board_token": "mondaydotcom"}),
        ("smartrecruiters", {"id": "ok7", "name": "O", "company_identifier": "smart_recruiters"}),
        ("smartrecruiters", {"id": "ok8", "name": "O", "company_identifier": "smart-recruiters"}),
        ("smartrecruiters", {"id": "ok9", "name": "O", "company_identifier": "smart.recruiters"}),
    ]
    for provider, payload in good:
        assert validate_document({"providers": {provider: {"companies": [payload]}}})


def test_all_builtin_identifiers_still_pass_the_tightened_charsets() -> None:
    """Zero-config catalogs must validate unchanged after tightening.

    A charset that silently rejected a curated default would be a
    backward-compatibility regression that config-only tests cannot see.
    """
    for company_id, entry in builtin_defaults.SMARTRECRUITERS_COMPANIES.items():
        validate_document(
            {
                "providers": {
                    "smartrecruiters": {
                        "companies": [
                            {
                                "id": company_id,
                                "name": entry.name,
                                "company_identifier": entry.company_identifier,
                            }
                        ]
                    }
                }
            }
        )
    for company_id, entry in builtin_defaults.GREENHOUSE_COMPANIES.items():
        validate_document(
            {
                "providers": {
                    "greenhouse": {
                        "companies": [
                            {
                                "id": company_id,
                                "name": entry.name,
                                "board_token": entry.board_token,
                            }
                        ]
                    }
                }
            }
        )
    for company_id, entry in builtin_defaults.LEVER_COMPANIES.items():
        validate_document(
            {
                "providers": {
                    "lever": {
                        "companies": [
                            {"id": company_id, "name": entry.name, "slug": entry.slug}
                        ]
                    }
                }
            }
        )
    for company_id, entry in builtin_defaults.WORKDAY_COMPANIES.items():
        validate_document(
            {
                "providers": {
                    "workday": {
                        "companies": [
                            {
                                "id": company_id,
                                "name": entry.name,
                                "wd_company": entry.wd_company,
                                "wd_suffix": entry.wd_suffix,
                            }
                        ]
                    }
                }
            }
        )


def test_unsupported_provider_reports_the_provider_not_the_entry_shape() -> None:
    """An unsupported family with a populated block must name the real problem.

    Previously the company shape was validated first, so users of a new ATS family
    (for example `ashby`) were told their entries were the wrong model type rather
    than that the provider itself is unsupported in Milestone 5.
    """
    for block in (
        {"companies": [{"id": "x", "name": "N", "board_token": "b"}]},
        {"companies": "not-a-list"},
        None,
        {},
    ):
        with pytest.raises(CompanyRegistryError, match="unsupported provider"):
            validate_document({"providers": {"workable": block}}, source="unsupported")


def test_validation_error_identifies_the_offending_entry_index_and_id() -> None:
    """With several similar entries, the diagnostic must say which one failed."""
    document = {
        "providers": {
            "greenhouse": {
                "companies": [
                    {"id": "aaa", "name": "A", "board_token": "a"},
                    {"id": "bbb", "name": "B"},
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError) as excinfo:
        validate_document(document, source="indexed")
    message = str(excinfo.value)
    assert "companies[1]" in message
    assert "bbb" in message


def test_effective_catalog_isolated_even_with_injected_builtins() -> None:
    """The merge step must not alias caller-supplied built-in catalogs either.

    `effective_catalog` is public and exported; the read-boundary copy in
    `catalog()` does not cover a caller that consumes its result directly.
    """
    injected = {
        "greenhouse": {"seed": GreenhouseCompany(name="Seed", board_token="seed")},
        "lever": {},
        "eightfold": {},
        "direct_tech": {},
        "workday": {},
    }
    merged = effective_catalog([], builtin=injected)
    assert merged["greenhouse"]["seed"] is not injected["greenhouse"]["seed"]
    merged["greenhouse"]["seed"].name = "Mutated"
    assert injected["greenhouse"]["seed"].name == "Seed"


def test_narrowing_default_has_one_source_of_truth() -> None:
    """The entry type and the config schema must not drift apart."""
    from job_mcp.sources.company_registry.schema import DirectTechEntry

    assert (
        DirectTechCompany.__dataclass_fields__["default_query"].default
        is DIRECT_TECH_DEFAULT_QUERY
    )
    assert DirectTechEntry.model_fields["default_query"].default is DIRECT_TECH_DEFAULT_QUERY


def test_cross_family_typed_model_is_rejected_in_a_provider_block() -> None:
    """A Lever entry cannot be injected into the greenhouse block.

    Round-1 accepted any concrete company model in any family, which let
    ``effective_catalog`` place a ``LeverCompany`` into the Greenhouse catalog.
    """
    from job_mcp.sources.company_registry.schema import (
        LeverEntry,
        ProviderConfig,
        RegistryConfig,
    )

    with pytest.raises(CompanyRegistryError, match="greenhouse"):
        validate_document(
            {"providers": {"greenhouse": {"companies": [{"id": "x", "name": "X", "slug": "s"}]}}}
        )

    config = RegistryConfig(providers={"greenhouse": ProviderConfig(companies=[LeverEntry(id="x", name="X", slug="s")])})
    with pytest.raises(CompanyRegistryError, match="greenhouse"):
        effective_catalog([config])


def test_cross_family_override_is_rejected() -> None:
    from job_mcp.sources.company_registry.core import effective_catalog as ec
    from job_mcp.sources.company_registry.schema import (
        ProviderConfig,
        RegistryConfig,
        WorkdayOverride,
    )

    config = RegistryConfig(
        providers={"lever": ProviderConfig(companies=[WorkdayOverride(id="redis", wd_version=3)])}
    )
    with pytest.raises(CompanyRegistryError, match="lever"):
        ec([config])


def test_duplicate_ids_that_normalize_to_the_same_key_are_rejected() -> None:
    """`newco` and `" newco "` are the same company after whitespace stripping.

    Round-1 compared raw ids before pydantic stripped them, so the pair slipped
    through the conflicting-duplicate check and silently merged.
    """
    document = {
        "providers": {
            "greenhouse": {
                "companies": [
                    {"id": "newco", "name": "First", "board_token": "a"},
                    {"id": " newco ", "name": "Second", "board_token": "a"},
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="duplicate conflicting"):
        validate_document(document, source="normalize-dup")


def test_duplicate_identical_ids_after_normalization_collapse() -> None:
    document = {
        "providers": {
            "lever": {
                "companies": [
                    {"id": "newco", "name": "N", "slug": "s"},
                    {"id": " newco", "name": "N", "slug": "s"},
                ]
            }
        }
    }
    registry = validate_document(document)
    assert [entry.id for entry in registry.providers["lever"].companies] == ["newco", "newco"]


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.com:bad",
        "https://example.com:99999",
        "https://example.com:65536",
        "https://example.com:0",
        "https://example.com:-1",
        "https://example.com:443x",
    ],
)
def test_malformed_url_ports_are_rejected(bad_url: str) -> None:
    """`urlsplit(...).port` raises for non-numeric/out-of-range ports; it must be read."""
    for provider, field, base in (
        ("direct_tech", "search_url", {"id": "q", "name": "Q"}),
        ("workday", "base_url", {"id": "q", "name": "Q", "wd_company": "q"}),
    ):
        payload = {**base, field: bad_url}
        with pytest.raises(CompanyRegistryError, match=field):
            validate_document({"providers": {provider: {"companies": [payload]}}})


@pytest.mark.parametrize(
    "good_url",
    ["https://example.com", "https://example.com:443/api", "http://localhost:8000/jobs", "https://a.b:65535/x"],
)
def test_valid_url_ports_still_accepted(good_url: str) -> None:
    payload = {"id": "q", "name": "Q", "search_url": good_url}
    assert validate_document({"providers": {"direct_tech": {"companies": [payload]}}})


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://example.com/jobs", "https://", "https://@", "https:// /x", "javascript:alert(1)"])
def test_invalid_direct_tech_search_url_is_rejected(bad_url: str) -> None:
    document = {
        "providers": {
            "direct_tech": {"companies": [{"id": "urltest", "name": "T", "search_url": bad_url}]}
        }
    }
    with pytest.raises(CompanyRegistryError, match="search_url"):
        validate_document(document)


def test_invalid_workday_base_url_is_rejected() -> None:
    document = {
        "providers": {
            "workday": {
                "companies": [
                    {"id": "wdco", "name": "WD", "wd_company": "wd", "base_url": "ftp://x.example"}
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="base_url"):
        validate_document(document)


def test_workday_accepts_a_valid_base_url() -> None:
    document = {
        "providers": {
            "workday": {
                "companies": [
                    {
                        "id": "wdco",
                        "name": "WD",
                        "wd_company": "wd",
                        "base_url": "https://wd.wd2.myworkdayjobs.com",
                    }
                ]
            }
        }
    }
    entry = validate_document(document).providers["workday"].companies[0]
    assert entry.base_url is not None and entry.base_url.startswith("https://")


@pytest.mark.parametrize("hostname", ["http://bad.example", "bad/example", "host:name"])
def test_eightfold_hostname_must_be_a_bare_host(hostname: str) -> None:
    document = {
        "providers": {
            "eightfold": {
                "companies": [{"id": "h", "name": "H", "hostname": hostname, "domain": "d.example"}]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(document)


@pytest.mark.parametrize("bad_version", [0, -1, 100])
def test_workday_version_bounds_are_enforced(bad_version: int) -> None:
    document = {
        "providers": {
            "workday": {
                "companies": [{"id": "v", "name": "V", "wd_company": "v", "wd_version": bad_version}]
            }
        }
    }
    with pytest.raises(CompanyRegistryError):
        validate_document(document)


def test_wrong_field_type_yields_an_actionable_message() -> None:
    document = {
        "providers": {
            "greenhouse": {
                "companies": [
                    {"id": "t", "name": "T", "board_token": "t", "enabled": "maybe"}
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError) as excinfo:
        validate_document(document)
    message = str(excinfo.value)
    assert "enabled" in message
    assert "company registry" in message
    assert "configuration rejected" in message


def test_error_message_names_the_offending_file(tmp_path: Path) -> None:
    path = _config_file(tmp_path, {"providers": {"greenhouse": {"nope": 1}}})
    with pytest.raises(CompanyRegistryError) as excinfo:
        resolve([path])
    assert str(path) in str(excinfo.value)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "portals.yml"
    path.write_text("providers: [unclosed", encoding="utf-8")
    with pytest.raises(CompanyRegistryError, match="invalid YAML"):
        resolve([path])


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "portals.yml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(CompanyRegistryError, match="mapping"):
        resolve([path])


def test_empty_document_is_accepted_as_defaults_only(tmp_path: Path) -> None:
    empty = tmp_path / "portals.yml"
    empty.write_text("", encoding="utf-8")
    registry = resolve([empty])
    assert registry.catalog("greenhouse") == builtin_defaults.GREENHOUSE_COMPANIES


def test_invalid_configuration_raises_before_any_network_call(tmp_path: Path, monkeypatch) -> None:
    """Validation happens at resolve time, not lazily inside a provider fetch."""
    attempts: list[str] = []

    def _blocked(*args: Any, **kwargs: Any) -> None:
        attempts.append("network")
        raise AssertionError("network must not be reached")

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "get", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "post", _blocked)

    with pytest.raises(CompanyRegistryError):
        configure(
            [
                _config_file(
                    tmp_path,
                    {"providers": {"lever": {"companies": [{"id": "BAD SLUG!", "name": "X", "slug": "y"}]}}},
                )
            ]
        )
    assert attempts == []


# ---------------------------------------------------------------------------
# Query defaults (SPEC requirement 10)
# ---------------------------------------------------------------------------


def test_direct_tech_narrowing_default_query_is_preserved_by_default() -> None:
    """Backward compatibility: no silent coverage broadening in Milestone 5."""
    assert {entry.default_query for entry in resolve([]).catalog("direct_tech").values()} == {"student"}
    assert {k: v.default_query for k, v in DIRECT_TECH_COMPANIES.items()} == {
        key: "student" for key in DIRECT_TECH_COMPANIES
    }


def test_direct_tech_default_query_is_overridable_via_config_only(tmp_path: Path) -> None:
    module = inspect.getmodule(DirectTechSource)
    source_text_before = Path(module.__file__).read_text(encoding="utf-8")

    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "direct_tech": {
                            "companies": [
                                {"id": "google", "name": "Google", "default_query": "software engineer"}
                            ]
                        }
                    }
                },
            )
        ]
    )

    source = DirectTechSource()
    assert source._companies["google"].default_query == "software engineer"
    # Siblings keep the built-in default: the override is per-company.
    assert source._companies["amazon"].default_query == "student"
    assert Path(module.__file__).read_text(encoding="utf-8") == source_text_before


def test_direct_tech_default_query_can_be_emptied_explicitly(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {"providers": {"direct_tech": {"companies": [{"id": "ibm", "name": "IBM", "default_query": ""}]}}},
            )
        ]
    )
    assert DirectTechSource()._companies["ibm"].default_query == ""


def test_new_direct_tech_company_declares_its_own_query(tmp_path: Path) -> None:
    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "direct_tech": {
                            "companies": [
                                {
                                    "id": "cfgco",
                                    "name": "Cfg Co",
                                    "search_url": "https://careers.cfgco.example/api/jobs",
                                    "default_query": "machine learning intern",
                                }
                            ]
                        }
                    }
                },
            )
        ]
    )
    entry = DirectTechSource()._companies["cfgco"]
    assert entry.default_query == "machine learning intern"
    assert entry.provider_id == "cfgco"


def test_direct_tech_dict_coercion_defers_to_the_typed_entry_default() -> None:
    """The narrowing default is declared once, in the typed entry.

    The provider's legacy dict input path must not carry its own copy of the
    literal, otherwise a registry change would silently disagree with it.
    """
    module = inspect.getmodule(DirectTechSource)
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert '"student"' not in text, "direct_tech still duplicates the narrowing default"

    # Behaviour is unchanged: omitting the key yields the typed default.
    source = DirectTechSource(
        companies=[{"provider_id": "d1", "name": "D1", "search_url": "https://d1.example/jobs"}]
    )
    assert source._companies["d1"].default_query == "student"
    explicit = DirectTechSource(
        companies=[
            {
                "provider_id": "d2",
                "name": "D2",
                "search_url": "https://d2.example/jobs",
                "default_query": "staff engineer",
                "enabled": False,
            }
        ]
    )
    assert explicit._companies["d2"].default_query == "staff engineer"
    assert explicit._companies["d2"].enabled is False


def test_direct_tech_query_default_is_registry_data_not_a_fetch_literal() -> None:
    """The narrowing value lives in typed registry data, not in fetch logic.

    Every fetch path must read `company.default_query`, so the default is
    configurable through the registry. The dict-coercion convenience branch in
    the constructor is pre-existing behaviour and is not part of this invariant.
    """
    module = inspect.getmodule(DirectTechSource)
    text = Path(module.__file__).read_text(encoding="utf-8")
    # Every fetch path reads the value off the typed entry.
    assert "company.default_query" in text
    # The narrowing default no longer exists as a provider-side literal; it lives
    # on the registry entry type. Asserted through the resolved data, not by
    # grepping for a specific source-code spelling.
    assert '"student"' not in text
    default = resolve([]).catalog("direct_tech")["google"].default_query
    assert default == DIRECT_TECH_DEFAULT_QUERY


# ---------------------------------------------------------------------------
# Discovery, explicit reload, and layering
# ---------------------------------------------------------------------------


def test_shipped_example_configuration_validates_and_behaves_as_documented() -> None:
    """`portals.example.yml` is documentation; a broken example is a defect."""
    example = ROOT / "portals.example.yml"
    assert example.is_file(), "documented example configuration is missing"
    registry = resolve([example])

    # New companies are appended per family.
    assert registry.catalog("greenhouse")["example-startup"].board_token == "examplestartup"
    assert registry.catalog("lever")["example-limited"].slug == "examplelimited"
    assert registry.catalog("eightfold")["example-ai"].filter_distance == "25"
    # A built-in can be disabled while keeping its inherited metadata.
    dell = registry.catalog("workday")["dell"]
    assert dell.enabled is False and dell.name == builtin_defaults.WORKDAY_COMPANIES["dell"].name
    # A narrowing default query is overridable, and can be cleared explicitly.
    assert registry.catalog("direct_tech")["google"].default_query == "software engineer"
    assert registry.catalog("direct_tech")["example-direct"].default_query == ""
    # Every other family entry is untouched relative to built-ins.
    assert registry.catalog("direct_tech")["apple"] == builtin_defaults.DIRECT_TECH_COMPANIES["apple"]


def test_shipped_example_is_never_applied_automatically() -> None:
    """`portals.example.yml` is documentation; only real config names load.

    Deliberately does not assert that a developer-local `portals.yml` is absent:
    that file is the supported way to configure the registry locally, and a test
    that fails because a user configured their own checkout would be hostile.
    """
    assert discover_config_paths.__doc__  # discovery contract
    from job_mcp.sources.company_registry.core import DEFAULT_CONFIG_FILENAMES

    assert DEFAULT_CONFIG_FILENAMES == ("portals.yml", "portals.yaml")
    assert "portals.example.yml" not in DEFAULT_CONFIG_FILENAMES
    assert (ROOT / "portals.example.yml").is_file()
    # Resolution with an explicit empty path list is the zero-config baseline.
    assert resolve([]).catalog("greenhouse") == builtin_defaults.GREENHOUSE_COMPANIES


def test_env_var_selects_one_config_file(tmp_path: Path, monkeypatch) -> None:
    path = _config_file(
        tmp_path,
        {"providers": {"greenhouse": {"companies": [{"id": "envco", "name": "Env", "board_token": "e"}]}}},
    )
    monkeypatch.setenv("COMPANY_REGISTRY_PATH", str(path))
    assert discover_config_paths() == [path]
    assert "envco" in resolve().catalog("greenhouse")


def test_env_var_pointing_at_a_missing_file_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("COMPANY_REGISTRY_PATH", "/nonexistent/portals.yml")
    with pytest.raises(CompanyRegistryError, match="missing file"):
        discover_config_paths()


def test_only_one_project_config_file_is_discovered(tmp_path: Path, monkeypatch) -> None:
    """`portals.yml` wins outright; `portals.yaml` is not also merged.

    Merging both would let two files that describe the same project silently
    combine, so discovery selects exactly one project file.
    """
    project = tmp_path / "proj"
    project.mkdir()
    (project / "portals.yml").write_text(
        yaml.safe_dump({"providers": {"lever": {"companies": [{"id": "ymlco", "name": "Y", "slug": "y"}]}}}),
        encoding="utf-8",
    )
    (project / "portals.yaml").write_text(
        yaml.safe_dump({"providers": {"lever": {"companies": [{"id": "yamlco", "name": "Z", "slug": "z"}]}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    assert discover_config_paths() == [project / "portals.yml"]


def test_project_config_is_the_only_implicit_source(tmp_path: Path, monkeypatch) -> None:
    """A config outside the working directory is never picked up implicitly.

    Regression guard: discovery used to scan the installed package's parent
    directories too, which applied a repository-root file after the working
    directory file and let the less specific location win.
    """
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    assert discover_config_paths() == []

    other = tmp_path / "other"
    other.mkdir()
    (other / "portals.yml").write_text(
        yaml.safe_dump({"providers": {"lever": {"companies": [{"id": "outsidco", "name": "O", "slug": "o"}]}}}),
        encoding="utf-8",
    )
    assert discover_config_paths() == []
    assert "outsidco" not in resolve().catalog("lever")


def test_a_developer_local_config_does_not_disturb_zero_config_baselines(
    tmp_path: Path, monkeypatch
) -> None:
    """Ambient project config is real behaviour, not a test-time default.

    Discovery legitimately picks up the working directory's `portals.yml`, so
    tests that pin the curated baseline must resolve with an explicit empty path
    list rather than depend on how the checkout happens to be configured.
    """
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "portals.yml").write_text(
        yaml.safe_dump(
            {"providers": {"greenhouse": {"companies": [{"id": "devlocal", "name": "D", "board_token": "d"}]}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(dev)
    assert "devlocal" in resolve().catalog("greenhouse")
    assert "devlocal" not in resolve([]).catalog("greenhouse")
    assert configure([]).catalog("greenhouse") == builtin_defaults.GREENHOUSE_COMPANIES


def test_project_file_is_discovered_from_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _config_file(
        project, {"providers": {"lever": {"companies": [{"id": "discocovr", "name": "D", "slug": "d"}]}}}
    )
    monkeypatch.chdir(project)
    assert discover_config_paths() == [project / "portals.yml"]


def test_reload_is_explicit_and_never_implicit(tmp_path: Path) -> None:
    path = _config_file(
        tmp_path, {"providers": {"lever": {"companies": [{"id": "firstco", "name": "F", "slug": "f"}]}}}
    )
    assert "firstco" not in get_registry().catalog("lever")
    configure([path])
    assert "firstco" in get_registry().catalog("lever")

    path.write_text(
        yaml.safe_dump({"providers": {"lever": {"companies": [{"id": "secondco", "name": "S", "slug": "s"}]}}}),
        encoding="utf-8",
    )
    assert "secondco" not in get_registry().catalog("lever"), "cached registry must not hot-reload"
    configure([path])
    assert "secondco" in get_registry().catalog("lever")


def test_reset_forces_re_resolution(tmp_path: Path, monkeypatch) -> None:
    from job_mcp.sources.company_registry import reset

    path = _config_file(
        tmp_path, {"providers": {"lever": {"companies": [{"id": "lateco", "name": "L", "slug": "l"}]}}}
    )
    assert "lateco" not in get_registry().catalog("lever")
    monkeypatch.setenv("COMPANY_REGISTRY_PATH", str(path))
    reset()
    assert "lateco" in get_registry().catalog("lever")


def test_configure_with_no_paths_installs_defaults_only() -> None:
    assert configure([]).catalog("greenhouse") == builtin_defaults.GREENHOUSE_COMPANIES


def test_effective_catalog_accepts_injected_defaults_for_isolation() -> None:
    empty_builtin = {provider: {} for provider in PROVIDERS}
    config = validate_document(
        {"providers": {"greenhouse": {"companies": [{"id": "only", "name": "Only", "board_token": "o"}]}}}
    )
    merged = effective_catalog([config], builtin=empty_builtin)
    assert list(merged["greenhouse"]) == ["only"]
    assert merged["lever"] == {}


def test_registry_repr_is_informative() -> None:
    text = repr(resolve([]))
    assert "CompanyRegistry" in text
    for provider in PROVIDERS:
        assert provider in text


def test_querying_an_unmanaged_family_fails_loudly() -> None:
    with pytest.raises(CompanyRegistryError, match="unsupported provider"):
        resolve([]).catalog("bamboohr")


@pytest.mark.parametrize("family", ["workable", "bamboohr", "jobsapi"])
def test_no_new_ats_family_is_accepted_in_milestone_5(family: str) -> None:
    with pytest.raises(CompanyRegistryError, match="unsupported provider"):
        validate_document({"providers": {family: {"companies": []}}})


# ---------------------------------------------------------------------------
# Architectural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_modules_do_not_read_config_files(provider: str) -> None:
    """Providers receive typed entries; they never load registry config.

    Scope note: parsing a JSON *response body* is normal provider behaviour and
    is deliberately not covered by this invariant, which is about configuration
    ownership.
    """
    module = inspect.getmodule(FAMILIES[provider]["source"])
    text = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("yaml", "safe_load", "portals.yml", "COMPANY_REGISTRY_PATH"):
        assert forbidden not in text, f"{module.__name__} references {forbidden!r}"
    # The catalog literal left the provider module entirely.
    assert "Company(name=" not in text, f"{module.__name__} still embeds company data"
    # Construction resolves through the registry, not through a module constant.
    calls = [f'registry_catalog("{provider}")', f"registry_catalog('{provider}')"]
    assert any(call in text for call in calls), f"{module.__name__} bypasses the registry"


def test_registry_layer_performs_no_network_or_scraping_work() -> None:
    package = ROOT / "job_mcp" / "sources" / "company_registry"
    modules = sorted(package.glob("*.py"))
    assert modules, "registry package not found"
    for module in modules:
        text = module.read_text(encoding="utf-8")
        for forbidden in ("httpx", "playwright", "requests", "asyncio", "fetch_jobs", "BeautifulSoup"):
            assert forbidden not in text, f"{module.name} references {forbidden!r}"


def entry_sort_key(entry: Any) -> tuple[Any, ...]:
    """Stable ordering for comparing two catalogs entry-by-entry."""
    return (type(entry).__name__, *[str(value) for value in dataclasses.astuple(entry)])


def test_registry_output_is_typed_before_provider_construction() -> None:
    expected_types = {
        "ashby": AshbySource,
        "smartrecruiters": SmartRecruitersSource,
        "greenhouse": GreenhouseSource,
        "lever": LeverSource,
        "eightfold": EightfoldAISource,
        "direct_tech": DirectTechSource,
        "workday": WorkdaySource,
    }
    for provider, source_class in expected_types.items():
        entry_type = type(
            next(iter(builtin_defaults.__dict__[f"{provider.upper()}_COMPANIES"].values()))
        )
        active = get_registry()
        # Output is typed provider entry data, never raw mappings.
        resolved = active.catalog(provider)
        assert resolved, f"{provider} registry catalog resolved empty"
        for entry in resolved.values():
            assert type(entry) is entry_type
        # The constructed provider holds exactly the companies the active
        # registry resolved. Identity is deliberately not asserted: catalog reads
        # return copies so no caller can mutate this snapshot through a provider
        # instance. Keys are not compared either, because eightfold and workday
        # index internally by domain/tenant (documented limitation), so only the
        # entry values can be equated.
        source = source_class()
        expected = sorted(active.catalog(provider).values(), key=entry_sort_key)
        actual = sorted(source._companies.values(), key=entry_sort_key)
        assert actual == expected, f"{provider} resolved a different catalog than the registry"
        for entry in source._companies.values():
            assert type(entry) is entry_type
        assert len(source._companies) == len(resolved)
    assert set(expected_types) == set(PROVIDERS)


def test_active_registry_reads_are_isolated_from_caller_mutation() -> None:
    """`catalog()` hands out copies, so the snapshot cannot be poisoned (P2).

    Without this, mutating an entry obtained from the cached registry (or held by
    one provider instance) would change what a later provider constructs.
    """
    registry = get_registry()
    original = registry.catalog("greenhouse")["jfrog"]
    snapshot = registry.catalog("greenhouse")["jfrog"]
    snapshot.name = "Poisoned"

    assert registry.catalog("greenhouse")["jfrog"] == original
    assert registry.catalog("greenhouse")["jfrog"].name == "JFrog"

    # Nested list fields are copies too.
    fold = registry.catalog("eightfold")["nvidia"]
    fold.locations.append("Phantom City")
    assert "Phantom City" not in registry.catalog("eightfold")["nvidia"].locations

    # A mutation made through one provider instance cannot reach the next.
    configure([])
    first = GreenhouseSource()
    first._companies["jfrog"].name = "Poisoned"
    assert GreenhouseSource()._companies["jfrog"].name == "JFrog"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_resolved_entries_are_isolated_from_builtin_defaults(provider: str) -> None:
    """A resolved catalog is a deep copy: mutating it must not reach defaults.

    Covers both scalar fields and nested list fields, which were previously
    aliased to the built-in catalog objects.
    """
    builtins = builtin_defaults.__dict__[f"{provider.upper()}_COMPANIES"]
    sample_id = next(iter(builtins))
    before = copy.deepcopy(builtins[sample_id])

    entry = resolve([]).catalog(provider)[sample_id]
    assert entry is not builtins[sample_id], "registry handed out the built-in object itself"
    assert entry == before

    entry.name = "Mutated"
    mutated_list = None
    for field_name in ("locations", "wd_locations"):
        if isinstance(getattr(entry, field_name, None), list):
            mutated_list = getattr(entry, field_name)
            mutated_list.append("Mutated City")
    assert builtins[sample_id] == before, f"defaults.{provider} was mutated through a resolved entry"

    # A second resolution is unaffected too.
    again = resolve([]).catalog(provider)[sample_id]
    assert again == before
    assert mutated_list is None or "Mutated City" not in getattr(
        builtins[sample_id], "locations", getattr(builtins[sample_id], "wd_locations", [])
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_each_resolution_returns_fresh_objects(tmp_path: Path, provider: str) -> None:
    """Two independent resolutions must not alias entry objects."""
    path = _config_file(
        tmp_path,
        {"providers": {provider: {"companies": [{"id": "shared", **FAMILIES[provider]["new"]}]}}},
    )
    first = resolve([path]).catalog(provider)["shared"]
    second = resolve([path]).catalog(provider)["shared"]
    assert first == second
    assert first is not second


def test_comeet_is_not_migrated_because_it_lacks_an_enabled_seam() -> None:
    """Documented boundary: Comeet entries carry no `enabled` flag, so wiring it
    would change provider semantics, which Milestone 5 forbids."""
    from job_mcp.sources.public.comeet import ComeetCompany

    assert "enabled" not in ComeetCompany.__dataclass_fields__
    assert "comeet" not in supported_providers()


# ---------------------------------------------------------------------------
# Round-4 P1 regression: DNS length bounds
# ---------------------------------------------------------------------------


def test_dns_label_at_63_characters_is_accepted() -> None:
    """A 63-character label is the DNS maximum and must be accepted."""
    label_63 = "a" * 63
    hostname = f"{label_63}.eightfold.ai"
    document = {
        "providers": {
            "eightfold": {
                "companies": [
                    {"id": "dns63", "name": "DNS 63", "hostname": hostname, "domain": "dns63.example"}
                ]
            }
        }
    }
    assert validate_document(document)


def test_dns_label_at_64_characters_is_rejected() -> None:
    """A 64-character label exceeds the DNS per-label limit and must be rejected."""
    label_64 = "a" * 64
    hostname = f"{label_64}.eightfold.ai"
    document = {
        "providers": {
            "eightfold": {
                "companies": [
                    {"id": "dns64", "name": "DNS 64", "hostname": hostname, "domain": "dns64.example"}
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(document)


def test_every_label_in_a_dotted_hostname_obeys_the_63_char_limit() -> None:
    """Each individual component of a dotted hostname must be ≤63 chars."""
    label_64 = "b" * 64
    # First label is too long
    hostname = f"{label_64}.eightfold.ai"
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(
            {"providers": {"eightfold": {"companies": [
                {"id": "h1", "name": "H", "hostname": hostname, "domain": "d.example"}
            ]}}}
        )
    # Middle label is too long
    hostname_mid = f"ok.{label_64}.ai"
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(
            {"providers": {"eightfold": {"companies": [
                {"id": "h2", "name": "H", "hostname": hostname_mid, "domain": "d.example"}
            ]}}}
        )
    # Last label is too long
    hostname_last = f"ok.eightfold.{label_64}"
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(
            {"providers": {"eightfold": {"companies": [
                {"id": "h3", "name": "H", "hostname": hostname_last, "domain": "d.example"}
            ]}}}
        )


def test_total_hostname_at_253_characters_is_accepted() -> None:
    """A hostname at exactly 253 characters is the DNS maximum and must be accepted."""
    # Build a hostname of exactly 253 characters from valid labels
    # Use labels of 63 chars + dots: "aaa...a.aaa...a.aaa...a.aa...a"
    # 63 + 1 + 63 + 1 + 63 + 1 + 61 = 253
    label_63 = "a" * 63
    label_61 = "a" * 61
    hostname = f"{label_63}.{label_63}.{label_63}.{label_61}"
    assert len(hostname) == 253
    document = {
        "providers": {
            "eightfold": {
                "companies": [
                    {"id": "dns253", "name": "DNS 253", "hostname": hostname, "domain": "dns253.example"}
                ]
            }
        }
    }
    assert validate_document(document)


def test_total_hostname_exceeding_253_characters_is_rejected() -> None:
    """A hostname longer than 253 characters exceeds DNS total-length limit."""
    # 63 + 1 + 63 + 1 + 63 + 1 + 62 = 254
    label_63 = "a" * 63
    label_62 = "a" * 62
    hostname = f"{label_63}.{label_63}.{label_63}.{label_62}"
    assert len(hostname) == 254
    document = {
        "providers": {
            "eightfold": {
                "companies": [
                    {"id": "dns254", "name": "DNS 254", "hostname": hostname, "domain": "dns254.example"}
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="hostname"):
        validate_document(document)


def test_workday_host_label_at_63_characters_is_accepted() -> None:
    """wd_company is a DNS label; 63 chars is the maximum."""
    label_63 = "a" * 63
    document = {
        "providers": {
            "workday": {
                "companies": [
                    {"id": "wd63", "name": "WD63", "wd_company": label_63}
                ]
            }
        }
    }
    assert validate_document(document)


def test_workday_host_label_at_64_characters_is_rejected() -> None:
    """wd_company is a DNS label; 64 chars exceeds the limit."""
    label_64 = "a" * 64
    document = {
        "providers": {
            "workday": {
                "companies": [
                    {"id": "wd64", "name": "WD64", "wd_company": label_64}
                ]
            }
        }
    }
    with pytest.raises(CompanyRegistryError, match="wd_company"):
        validate_document(document)


def test_builtin_eightfold_hostnames_obey_dns_length_bounds() -> None:
    """Every curated Eightfold hostname must satisfy DNS length constraints."""
    for company_id, entry in builtin_defaults.EIGHTFOLD_COMPANIES.items():
        hostname = entry.hostname
        assert len(hostname) <= 253, f"{company_id}: hostname {hostname!r} exceeds 253 chars"
        for label in hostname.split("."):
            assert 1 <= len(label) <= 63, (
                f"{company_id}: hostname label {label!r} in {hostname!r} exceeds 63 chars"
            )


def test_builtin_workday_companies_obey_dns_label_length() -> None:
    """Every curated Workday wd_company must be ≤63 chars."""
    for company_id, entry in builtin_defaults.WORKDAY_COMPANIES.items():
        assert 1 <= len(entry.wd_company) <= 63, (
            f"{company_id}: wd_company {entry.wd_company!r} exceeds DNS label limit"
        )


# ---------------------------------------------------------------------------
# Round-4 P2 regression: constructor-input isolation
# ---------------------------------------------------------------------------


def test_constructor_input_mapping_mutation_does_not_change_registry(tmp_path: Path) -> None:
    """CompanyRegistry must own an isolated snapshot of its input catalogs.

    A caller that retains a reference to the catalogs mapping passed to the
    constructor must not be able to mutate the registry's internal state.
    """
    from job_mcp.sources.company_registry.core import CompanyRegistry

    original_entry = GreenhouseCompany(name="Stable", board_token="stable")
    catalogs: dict[str, dict[str, Any]] = {
        provider: {} for provider in supported_providers()
    }
    catalogs["greenhouse"]["testco"] = original_entry

    registry = CompanyRegistry(catalogs, [])

    # Mutate the original mapping after construction
    catalogs["greenhouse"]["injected"] = GreenhouseCompany(name="Injected", board_token="inj")
    del catalogs["greenhouse"]["testco"]

    # The registry must still contain the original entry and not the injection
    result = registry.catalog("greenhouse")
    assert "testco" in result, "registry lost an entry when the input mapping was mutated"
    assert result["testco"].name == "Stable"
    assert "injected" not in result, "registry gained an entry from post-construction mutation"


def test_constructor_input_entry_mutation_does_not_change_registry(tmp_path: Path) -> None:
    """Mutating an entry object passed to the constructor must not change the snapshot."""
    from job_mcp.sources.company_registry.core import CompanyRegistry

    entry = GreenhouseCompany(name="Original", board_token="orig")
    catalogs: dict[str, dict[str, Any]] = {
        provider: {} for provider in supported_providers()
    }
    catalogs["greenhouse"]["mutco"] = entry

    registry = CompanyRegistry(catalogs, [])

    # Mutate the original entry after construction
    entry.name = "Mutated"
    entry.board_token = "mutated"

    result = registry.catalog("greenhouse")
    assert result["mutco"].name == "Original", "constructor did not isolate entry objects"
    assert result["mutco"].board_token == "orig"


def test_constructor_input_nested_list_mutation_does_not_change_registry() -> None:
    """Mutating a nested list field on an input entry must not reach the registry."""
    from job_mcp.sources.company_registry.core import CompanyRegistry

    locs = ["Tel Aviv", "Haifa"]
    entry = EightfoldCompany(
        name="ListTest", hostname="listtest.eightfold.ai",
        domain="listtest.com", locations=locs,
    )
    catalogs: dict[str, dict[str, Any]] = {
        provider: {} for provider in supported_providers()
    }
    catalogs["eightfold"]["listco"] = entry

    registry = CompanyRegistry(catalogs, [])

    # Mutate the list on the original entry
    entry.locations.append("Phantom City")
    locs.append("Ghost Town")

    result = registry.catalog("eightfold")
    assert "Phantom City" not in result["listco"].locations
    assert "Ghost Town" not in result["listco"].locations
    assert result["listco"].locations == ["Tel Aviv", "Haifa"]


# ---------------------------------------------------------------------------
# Provider-family registry boundary (must remain untouched)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag",
    [
        "ENABLE_GREENHOUSE",
        "ENABLE_LEVER",
        "ENABLE_EIGHTFOLD",
        "ENABLE_DIRECT_TECH",
        "ENABLE_WORKDAY",
    ],
)
def test_family_level_env_flags_still_control_registration(flag: str, monkeypatch) -> None:
    from job_mcp.sources.registry import create_default_registry

    source_id = flag.removeprefix("ENABLE_").lower()
    monkeypatch.setenv(flag, "true")
    assert source_id in create_default_registry()
    monkeypatch.setenv(flag, "false")
    assert source_id not in create_default_registry()


def test_company_registry_does_not_disable_a_whole_family_when_companies_are_off(tmp_path: Path) -> None:
    """Company-level `enabled` and family-level registration stay independent."""
    from job_mcp.sources.registry import create_default_registry

    configure(
        [
            _config_file(
                tmp_path,
                {
                    "providers": {
                        "greenhouse": {
                            "companies": [
                                {"id": key, "enabled": False} for key in GREENHOUSE_COMPANIES
                            ]
                        }
                    }
                },
            )
        ]
    )
    assert enabled_catalog("greenhouse") == {}
    assert "greenhouse" in create_default_registry()
