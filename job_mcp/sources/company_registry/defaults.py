"""Built-in curated company catalogs for provider families.

This is the zero-configuration default set. It is data only: the entries here
are the same companies that were previously hardcoded in each provider module,
moved verbatim so provider implementation files no longer carry the catalog.

Milestone 5 invariant: the effective registry equals these defaults when no
user or project configuration is present.
"""

from __future__ import annotations

from job_mcp.sources.company_registry.entries import (
    AshbyCompany,
    DirectTechCompany,
    EightfoldCompany,
    GreenhouseCompany,
    LeverCompany,
    WorkdayCompany,
)

# Curated directory of tech companies using Ashby ATS
ASHBY_COMPANIES: dict[str, AshbyCompany] = {
    "ashby": AshbyCompany(name="Ashby", board_name="ashby", enabled=True),
}


# Curated directory of Israeli AI/tech companies using Greenhouse
GREENHOUSE_COMPANIES: dict[str, GreenhouseCompany] = {
    "lightricks": GreenhouseCompany(name="Lightricks", board_token="lightricks", enabled=True),
    "appsflyer": GreenhouseCompany(name="AppsFlyer", board_token="appsflyer", enabled=True),
    "datadog": GreenhouseCompany(name="Datadog Israel", board_token="datadog", enabled=True),
    "jfrog": GreenhouseCompany(name="JFrog", board_token="jfrog", enabled=True),
    "orca_security": GreenhouseCompany(name="Orca Security", board_token="orcasecurity", enabled=True),
    "yotpo": GreenhouseCompany(name="Yotpo", board_token="yotpo", enabled=True),
    "riskified": GreenhouseCompany(name="Riskified", board_token="riskified", enabled=True),
    # Companies that migrated off public Greenhouse API (kept for test compatibility)
    "ai21labs": GreenhouseCompany(name="AI21 Labs", board_token="ai21labs", enabled=False),
    "tabnine": GreenhouseCompany(name="Tabnine", board_token="tabnine", enabled=False),
    "bria": GreenhouseCompany(name="Bria AI", board_token="baborstudio", enabled=False),
    "gong": GreenhouseCompany(name="Gong", board_token="gong", enabled=False),
    "wiz": GreenhouseCompany(name="Wiz", board_token="wiz", enabled=False),
    "orcaai": GreenhouseCompany(name="Orca AI", board_token="orcaai", enabled=False),
    "lemonade": GreenhouseCompany(name="Lemonade", board_token="lemonade", enabled=False),
    "monday": GreenhouseCompany(name="monday.com", board_token="mondaydotcom", enabled=False),
    "fiverr": GreenhouseCompany(name="Fiverr", board_token="fiverr", enabled=False),
    "deepchecks": GreenhouseCompany(name="Deepchecks", board_token="deepchecks", enabled=False),
    "snyk": GreenhouseCompany(name="Snyk", board_token="snyk", enabled=False),
}


# Curated directory of Israeli tech companies and startups using Lever
LEVER_COMPANIES: dict[str, LeverCompany] = {
    "drivenets": LeverCompany(name="DriveNets", slug="drivenets", enabled=True),
    "here": LeverCompany(name="HERE Technologies", slug="here", enabled=True),
    "yotpo": LeverCompany(name="Yotpo", slug="yotpo", enabled=True),
    "redis": LeverCompany(name="Redis", slug="redis", enabled=True),
    "melio": LeverCompany(name="Melio", slug="melio", enabled=True),
    "papaya_global": LeverCompany(name="Papaya Global", slug="papayaglobal", enabled=True),
    "hibob": LeverCompany(name="HiBob", slug="hibob", enabled=True),
    "palo_alto_networks": LeverCompany(name="Palo Alto Networks Israel", slug="paloaltonetworks", enabled=True),
    "riskified": LeverCompany(name="Riskified", slug="riskified", enabled=True),
    "k_health": LeverCompany(name="K Health", slug="khealth", enabled=True),
    "tipalti": LeverCompany(name="Tipalti", slug="tipalti", enabled=True),
    "deel": LeverCompany(name="Deel Israel", slug="deel", enabled=True),
    "run_ai": LeverCompany(name="Run:ai", slug="runai", enabled=True),
    "deci": LeverCompany(name="Deci AI", slug="deci", enabled=True),
}


# Curated directory of tech enterprises using Eightfold AI ATS
EIGHTFOLD_COMPANIES: dict[str, EightfoldCompany] = {
    "nvidia": EightfoldCompany(
        name="NVIDIA",
        hostname="nvidia.eightfold.ai",
        domain="nvidia.com",
        locations=["Yokne'am Illit", "Tel Aviv", "Israel"],
        filter_distance="16",
        enabled=True,
    ),
    "micron": EightfoldCompany(
        name="Micron",
        hostname="micron.eightfold.ai",
        domain="micron.com",
        locations=[],
        filter_distance="16",
        enabled=True,
    ),
    "paypal": EightfoldCompany(
        name="PayPal",
        hostname="paypal.eightfold.ai",
        domain="paypal.com",
        locations=["Israel", "Tel Aviv"],
        filter_distance="16",
        enabled=True,
    ),
    "intel": EightfoldCompany(
        name="Intel",
        hostname="intel.eightfold.ai",
        domain="intel.com",
        locations=["Israel", "Haifa", "Petach Tikva", "Jerusalem"],
        filter_distance="16",
        enabled=False,  # Intel careers use Workday (intel.wd1.myworkdayjobs.com)
    ),
    "elbit_systems": EightfoldCompany(
        name="Elbit Systems",
        hostname="elbitsystems.eightfold.ai",
        domain="elbitsystems.com",
        locations=["Israel"],
        filter_distance="16",
        enabled=False,  # Elbit Systems does not use Eightfold AI public PCSX
    ),
}


# Curated directory of direct tech company career endpoints
DIRECT_TECH_COMPANIES: dict[str, DirectTechCompany] = {
    "google": DirectTechCompany(
        provider_id="google",
        name="Google",
        search_url="https://www.google.com/about/careers/applications/jobs/results/",
        default_query="student",
        default_location="Haifa, Israel",
    ),
    "amazon": DirectTechCompany(
        provider_id="amazon",
        name="Amazon",
        search_url="https://www.amazon.jobs/api/jobs/search",
        default_query="student",
        locations=["Haifa"],
    ),
    "apple": DirectTechCompany(
        provider_id="apple",
        name="Apple",
        search_url="https://jobs.apple.com/api/v1/search",
        default_query="student",
        locations=["postLocation-state1312"],
    ),
    "ibm": DirectTechCompany(
        provider_id="ibm",
        name="IBM",
        search_url="https://www-api.ibm.com/search/api/v2",
        default_query="student",
        default_location="Israel",
    ),
}


# Curated directory of tech enterprises using Workday ATS
WORKDAY_COMPANIES: dict[str, WorkdayCompany] = {
    "intel": WorkdayCompany(
        name="Intel",
        wd_company="intel",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=True,
    ),
    "nvidia": WorkdayCompany(
        name="NVIDIA",
        wd_company="nvidia",
        wd_version=5,
        wd_suffix="NVIDIAExternalCareerSite",
        wd_locations=[],
        enabled=True,
    ),
    "cisco": WorkdayCompany(
        name="Cisco",
        wd_company="cisco",
        wd_version=5,
        wd_suffix="Cisco_Careers",
        wd_locations=[],
        enabled=True,
    ),
    "philips": WorkdayCompany(
        name="Philips",
        wd_company="philips",
        wd_version=3,
        wd_suffix="jobs-and-careers",
        wd_locations=[],
        enabled=True,
    ),
    "dell": WorkdayCompany(
        name="Dell",
        wd_company="dell",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # Dell manages careers via custom portal (jobs.dell.com)
    ),
    "autodesk": WorkdayCompany(
        name="Autodesk",
        wd_company="autodesk",
        wd_version=1,
        wd_suffix="Ext",
        wd_locations=[],
        enabled=True,
    ),
    "microsoft": WorkdayCompany(
        name="Microsoft",
        wd_company="microsoft",
        wd_version=2,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # Microsoft uses careers.microsoft.com custom portal
    ),
    "qualcomm": WorkdayCompany(
        name="Qualcomm",
        wd_company="qualcomm",
        wd_version=5,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # Qualcomm Workday blocks direct automated POSTs
    ),
    "ptc": WorkdayCompany(
        name="PTC",
        wd_company="ptc",
        wd_version=1,
        wd_suffix="External",
        wd_locations=[],
        enabled=False,  # PTC endpoint changed
    ),
}

# Ordered default view over each curated catalog.
DEFAULT_EIGHTFOLD_COMPANIES: list[EightfoldCompany] = list(EIGHTFOLD_COMPANIES.values())
DEFAULT_DIRECT_TECH_COMPANIES: list[DirectTechCompany] = list(DIRECT_TECH_COMPANIES.values())
DEFAULT_WORKDAY_COMPANIES: list[WorkdayCompany] = list(WORKDAY_COMPANIES.values())


# Built-in company identifiers per provider family. Configuration entries whose
# id appears here are validated as partial overrides; anything else must fully
# specify the fields a new company of that family requires.
BUILTIN_COMPANY_IDS: dict[str, frozenset[str]] = {
    "ashby": frozenset(ASHBY_COMPANIES),
    "greenhouse": frozenset(GREENHOUSE_COMPANIES),
    "lever": frozenset(LEVER_COMPANIES),
    "eightfold": frozenset(EIGHTFOLD_COMPANIES),
    "direct_tech": frozenset(DIRECT_TECH_COMPANIES),
    "workday": frozenset(WORKDAY_COMPANIES),
}


__all__ = [
    "ASHBY_COMPANIES",
    "BUILTIN_COMPANY_IDS",
    "DEFAULT_DIRECT_TECH_COMPANIES",
    "DEFAULT_EIGHTFOLD_COMPANIES",
    "DEFAULT_WORKDAY_COMPANIES",
    "DIRECT_TECH_COMPANIES",
    "EIGHTFOLD_COMPANIES",
    "GREENHOUSE_COMPANIES",
    "LEVER_COMPANIES",
    "WORKDAY_COMPANIES",
]
