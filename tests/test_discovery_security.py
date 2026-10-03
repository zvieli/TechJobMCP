"""Security and SSRF tests for discovery URL handling (Milestone 6-C)."""

from __future__ import annotations

from job_mcp.core.search_plane.discovery.security import (
    default_dns_resolver,
    validate_url_safety,
)


def test_valid_public_urls_pass() -> None:
    """Standard public HTTP and HTTPS URLs must pass validation."""
    safe_urls = [
        "https://jobs.ashbyhq.com/example",
        "http://careers.smartrecruiters.com/example",
        "https://apply.workable.com/example/",
        "https://example.com:8443/careers",
        "http://93.184.216.34/jobs",  # example.com public IP
    ]
    for url in safe_urls:
        ok, reason = validate_url_safety(url)
        assert ok is True, f"Expected {url} to be safe, got reason: {reason}"
        assert reason is None


def test_whitespace_and_empty_urls_rejected() -> None:
    """Whitespace or empty strings must be rejected."""
    bad_urls = [
        "",
        "   ",
        "https://example .com/jobs",
        "https://example.com/ jobs",
    ]
    for url in bad_urls:
        ok, reason = validate_url_safety(url)
        assert ok is False
        assert reason is not None


def test_unsupported_schemes_rejected() -> None:
    """Non-http(s) schemes like file:, ftp:, data:, javascript: must be rejected."""
    bad_schemes = [
        "file:///etc/passwd",
        "ftp://example.com/careers",
        "data:text/html,<h1>Jobs</h1>",
        "javascript:alert(1)",
        "gopher://example.com/",
    ]
    for url in bad_schemes:
        ok, reason = validate_url_safety(url)
        assert ok is False
        assert "unsupported scheme" in (reason or "").lower()


def test_userinfo_credentials_rejected() -> None:
    """Embedded credentials (user:pass@host) must be rejected."""
    bad_urls = [
        "https://user:pass@example.com/careers",
        "http://admin@example.com/jobs",
        "https://foo:bar@jobs.ashbyhq.com/example",
    ]
    for url in bad_urls:
        ok, reason = validate_url_safety(url)
        assert ok is False
        assert "credentials" in (reason or "").lower()


def test_invalid_ports_rejected() -> None:
    """Port 0, negative ports, or ports > 65535 must be rejected."""
    bad_ports = [
        "http://example.com:0/jobs",
        "http://example.com:65536/jobs",
        "http://example.com:abc/jobs",
    ]
    for url in bad_ports:
        ok, reason = validate_url_safety(url)
        assert ok is False
        assert "port" in (reason or "").lower()


def test_localhost_and_loopback_rejected() -> None:
    """localhost, .localhost, 127.0.0.1, and ::1 must be rejected."""
    bad_hosts = [
        "http://localhost/careers",
        "http://localhost:8080/jobs",
        "https://foo.localhost/jobs",
        "http://127.0.0.1/",
        "http://127.0.0.1:8000/api",
        "http://[::1]/",
        "http://[::1]:8080/jobs",
        "http://[::ffff:127.0.0.1]/",
    ]
    for url in bad_hosts:
        ok, reason = validate_url_safety(url)
        assert ok is False, f"Expected {url} to be blocked, but it passed"
        assert any(term in (reason or "").lower() for term in ("localhost", "blocked", "loopback"))


def test_rfc1918_private_ranges_rejected() -> None:
    """RFC 1918 private IPv4 addresses must be rejected."""
    bad_ips = [
        "http://10.0.0.1/jobs",
        "http://10.254.254.254/",
        "http://172.16.0.1/",
        "http://172.31.255.255/",
        "http://192.168.1.1/careers",
        "http://192.168.0.254/",
    ]
    for url in bad_ips:
        ok, reason = validate_url_safety(url)
        assert ok is False, f"Expected {url} to be blocked, but it passed"
        assert "blocked" in (reason or "").lower()


def test_link_local_and_metadata_rejected() -> None:
    """Link-local (169.254.0.0/16) and AWS/GCP metadata IP (169.254.169.254) must be rejected."""
    bad_ips = [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.1.1/jobs",
        "http://[fe80::1]/",
    ]
    for url in bad_ips:
        ok, reason = validate_url_safety(url)
        assert ok is False, f"Expected {url} to be blocked, but it passed"
        assert "blocked" in (reason or "").lower()


def test_dns_resolution_private_ip_rejected() -> None:
    """Hostnames that resolve to private IP addresses via DNS must be blocked."""
    def mock_dns_resolver(hostname: str) -> list[str]:
        if hostname == "internal.company.com":
            return ["10.0.0.5"]
        if hostname == "metadata.spoof.com":
            return ["169.254.169.254"]
        if hostname == "safe.example.com":
            return ["93.184.216.34"]
        return default_dns_resolver(hostname)

    ok, reason = validate_url_safety(
        "https://internal.company.com/careers", dns_resolver=mock_dns_resolver
    )
    assert ok is False
    assert "resolved to blocked ip" in (reason or "").lower()

    ok, reason = validate_url_safety(
        "http://metadata.spoof.com/latest", dns_resolver=mock_dns_resolver
    )
    assert ok is False
    assert "resolved to blocked ip" in (reason or "").lower()

    ok, reason = validate_url_safety(
        "https://safe.example.com/careers", dns_resolver=mock_dns_resolver
    )
    assert ok is True
    assert reason is None
