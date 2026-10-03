"""ATS fingerprinting and URL classification for Milestone 6-C discovery."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from job_mcp.core.search_plane.discovery.models import DiscoveryEvidence
from job_mcp.sources.company_registry.schema import (
    _require_path_token,
    _require_workable_subdomain,
)

# Anchored regex patterns for supported ATS families
ASHBY_URL_RE = re.compile(
    r"https?://jobs\.ashbyhq\.com/([A-Za-z0-9._-]+)", re.IGNORECASE
)
ASHBY_API_RE = re.compile(
    r"https?://api\.ashbyhq\.com/posting-api/job-board/([A-Za-z0-9._-]+)", re.IGNORECASE
)

SMARTRECRUITERS_URL_RE = re.compile(
    r"https?://careers\.smartrecruiters\.com/([A-Za-z0-9._-]+)", re.IGNORECASE
)
SMARTRECRUITERS_API_RE = re.compile(
    r"https?://api\.smartrecruiters\.com/v1/companies/([A-Za-z0-9._-]+)/postings",
    re.IGNORECASE,
)

WORKABLE_URL_RE = re.compile(
    r"https?://apply\.workable\.com/([a-z0-9-]+)", re.IGNORECASE
)
WORKABLE_API_RE = re.compile(
    r"https?://(?:www\.)?workable\.com/api/accounts/([a-z0-9-]+)", re.IGNORECASE
)

# Unsupported ATS URL signatures for positive UNSUPPORTED classification
UNSUPPORTED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("greenhouse", re.compile(r"https?://boards\.greenhouse\.io/([A-Za-z0-9._-]+)", re.IGNORECASE)),
    ("lever", re.compile(r"https?://jobs\.lever\.co/([A-Za-z0-9._-]+)", re.IGNORECASE)),
    ("workday", re.compile(r"https?://[a-zA-Z0-9.-]+\.myworkdayjobs\.com", re.IGNORECASE)),
    ("taleo", re.compile(r"https?://[a-zA-Z0-9.-]+\.taleo\.net", re.IGNORECASE)),
    ("eightfold", re.compile(r"https?://[a-zA-Z0-9.-]+\.eightfold\.ai", re.IGNORECASE)),
    ("icims", re.compile(r"https?://[a-zA-Z0-9.-]+\.icims\.com", re.IGNORECASE)),
    ("bamboohr", re.compile(r"https?://[a-zA-Z0-9.-]+\.bamboohr\.com/(?:careers|jobs)", re.IGNORECASE)),
)


def classify_direct_url(
    url: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Classify an authoritative ATS direct URL without network I/O.

    Returns:
        (source_family, account, unsupported_family, error_message)
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return None, None, None, f"Malformed URL: {exc}"

    host = (parts.hostname or "").lower()
    path = parts.path

    # Strip leading/trailing slashes from path
    clean_path = path.strip("/")
    path_segments = clean_path.split("/") if clean_path else []

    # 1. Ashby
    if host == "jobs.ashbyhq.com":
        if not path_segments:
            return None, None, None, "Ashby URL missing board name segment"
        if len(path_segments) > 1:
            return None, None, None, f"Ashby URL contains multiple path segments: {clean_path!r}"
        raw_board = path_segments[0]
        try:
            board_name = _require_path_token(raw_board, "board_name")
            return "ashby", board_name, None, None
        except ValueError as exc:
            return None, None, None, f"Invalid Ashby board name: {exc}"

    # 2. SmartRecruiters
    if host in ("careers.smartrecruiters.com", "jobs.smartrecruiters.com"):
        if not path_segments:
            return None, None, None, "SmartRecruiters URL missing company identifier segment"
        if len(path_segments) > 1:
            return None, None, None, f"SmartRecruiters URL contains multiple path segments: {clean_path!r}"
        raw_company = path_segments[0]
        try:
            company_id = _require_path_token(raw_company, "company_identifier")
            return "smartrecruiters", company_id, None, None
        except ValueError as exc:
            return None, None, None, f"Invalid SmartRecruiters company identifier: {exc}"

    # 3. Workable (apply.workable.com)
    if host == "apply.workable.com":
        if not path_segments:
            return None, None, None, "Workable URL missing account subdomain segment"
        if len(path_segments) > 1:
            return None, None, None, f"Workable URL contains multiple path segments: {clean_path!r}"
        raw_account = path_segments[0].lower()
        try:
            account = _require_workable_subdomain(raw_account, "account_subdomain")
            return "workable", account, None, None
        except ValueError as exc:
            return None, None, None, f"Invalid Workable account subdomain: {exc}"

    # 4. Workable public API (www.workable.com/api/accounts/{account})
    if host in ("www.workable.com", "workable.com") and path.startswith("/api/accounts/"):
        api_segments = path.strip("/").split("/")
        if len(api_segments) == 3 and api_segments[0] == "api" and api_segments[1] == "accounts":
            raw_account = api_segments[2].lower()
            try:
                account = _require_workable_subdomain(raw_account, "account_subdomain")
                return "workable", account, None, None
            except ValueError as exc:
                return None, None, None, f"Invalid Workable account subdomain: {exc}"

    # 5. Unsupported ATS URL recognition
    if host == "boards.greenhouse.io":
        return None, None, "greenhouse", None
    if host == "jobs.lever.co":
        return None, None, "lever", None
    if host.endswith(".myworkdayjobs.com"):
        return None, None, "workday", None
    if host.endswith(".taleo.net"):
        return None, None, "taleo", None
    if host.endswith(".eightfold.ai"):
        return None, None, "eightfold", None
    if host.endswith(".icims.com"):
        return None, None, "icims", None
    if host.endswith(".bamboohr.com"):
        return None, None, "bamboohr", None

    return None, None, None, None


def inspect_html_body(
    html_text: str,
    source_url: str | None = None,
) -> tuple[list[tuple[str, str, DiscoveryEvidence]], list[str]]:
    """Scan HTML body content (max 256 KiB) for anchored provider signals and unsupported ATS references.

    Returns:
        (found_candidates, unsupported_families_found)
        where found_candidates is a list of (source_family, account, DiscoveryEvidence).
    """
    candidates: list[tuple[str, str, DiscoveryEvidence]] = []
    seen_keys: set[tuple[str, str]] = set()
    unsupported_found: list[str] = []

    # Check for unsupported ATS signatures first
    for fam_name, pattern in UNSUPPORTED_PATTERNS:
        if pattern.search(html_text) and fam_name not in unsupported_found:
            unsupported_found.append(fam_name)

    # 1. Ashby matches
    for pattern, kind_label in (
        (ASHBY_URL_RE, "HTML_PROVIDER_LINK"),
        (ASHBY_API_RE, "API_REFERENCE"),
    ):
        for match in pattern.finditer(html_text):
            token = match.group(1).strip()
            try:
                board = _require_path_token(token, "board_name")
                key = ("ashby", board)
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence = DiscoveryEvidence(
                        kind=kind_label,
                        value=f"jobs.ashbyhq.com/{board}",
                        source_url=source_url,
                    )
                    candidates.append(("ashby", board, evidence))
            except ValueError:
                continue

    # 2. SmartRecruiters matches
    for pattern, kind_label in (
        (SMARTRECRUITERS_URL_RE, "HTML_PROVIDER_LINK"),
        (SMARTRECRUITERS_API_RE, "API_REFERENCE"),
    ):
        for match in pattern.finditer(html_text):
            token = match.group(1).strip()
            try:
                company_id = _require_path_token(token, "company_identifier")
                key = ("smartrecruiters", company_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence = DiscoveryEvidence(
                        kind=kind_label,
                        value=f"careers.smartrecruiters.com/{company_id}",
                        source_url=source_url,
                    )
                    candidates.append(("smartrecruiters", company_id, evidence))
            except ValueError:
                continue

    # 3. Workable matches
    for pattern, kind_label in (
        (WORKABLE_URL_RE, "HTML_PROVIDER_LINK"),
        (WORKABLE_API_RE, "API_REFERENCE"),
    ):
        for match in pattern.finditer(html_text):
            token = match.group(1).strip().lower()
            try:
                subdomain = _require_workable_subdomain(token, "account_subdomain")
                key = ("workable", subdomain)
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence = DiscoveryEvidence(
                        kind=kind_label,
                        value=f"apply.workable.com/{subdomain}",
                        source_url=source_url,
                    )
                    candidates.append(("workable", subdomain, evidence))
            except ValueError:
                continue

    return candidates, unsupported_found
