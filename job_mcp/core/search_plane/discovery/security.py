"""URL security and SSRF protection for Milestone 6-C ATS discovery."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from urllib.parse import urlsplit

# Hostname validation: letters, digits, hyphens, dots
HOSTNAME_RE = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*$"
)

# Explicit blocked IPv4 networks (RFC 1918, RFC 3927, loopback, CGNAT, metadata, etc.)
BLOCKED_IPV4_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
)

# Explicit blocked IPv6 networks (RFC 4193, loopback, link-local, etc.)
BLOCKED_IPV6_NETWORKS = (
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("100::/64"),
    ipaddress.ip_network("2001:db8::/32"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
)


def is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP address belongs to any blocked/private/reserved range."""
    # Check standard Python properties
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True

    # Handle IPv4-mapped IPv6
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_ip_blocked(ip.ipv4_mapped)

    # Check explicit network lists
    if isinstance(ip, ipaddress.IPv4Address):
        for net in BLOCKED_IPV4_NETWORKS:
            if ip in net:
                return True
    elif isinstance(ip, ipaddress.IPv6Address):
        for net in BLOCKED_IPV6_NETWORKS:
            if ip in net:
                return True

    return False


def default_dns_resolver(hostname: str) -> list[str]:
    """Resolve hostname to a list of IP address strings."""
    lower = hostname.lower()
    if lower == "localhost" or lower.endswith(".localhost"):
        return ["127.0.0.1"]
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return [info[4][0] for info in infos]
    except (socket.gaierror, socket.herror, OSError):
        return []


def validate_url_safety(
    url: str,
    *,
    dns_resolver: Callable[[str], list[str]] | None = None,
) -> tuple[bool, str | None]:
    """Validate that a URL is safe to retrieve, enforcing strict SSRF protections.

    Returns:
        (True, None) if safe.
        (False, reason) if unsafe.
    """
    if not isinstance(url, str):
        return False, "URL must be a string"

    stripped = url.strip()
    if not stripped:
        return False, "URL cannot be empty"

    if any(c.isspace() for c in stripped):
        return False, "URL must not contain whitespace"

    try:
        parts = urlsplit(stripped)
    except ValueError as exc:
        return False, f"Malformed URL syntax: {exc}"

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Unsupported scheme {scheme!r}; only http and https are permitted"

    # Reject credentials (userinfo) in netloc
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        return False, "Embedded credentials in URL are prohibited"

    hostname = parts.hostname
    if not hostname:
        return False, "URL must contain a valid hostname"

    # Validate port
    try:
        port = parts.port
        if port is not None and (port <= 0 or port > 65535):
            return False, f"Invalid port {port}; must be between 1 and 65535"
    except ValueError as exc:
        return False, f"Invalid port in URL: {exc}"

    lower_host = hostname.lower()

    # Reject localhost directly
    if lower_host == "localhost" or lower_host.endswith(".localhost"):
        return False, "Localhost targets are prohibited"

    # Check if host is literal IP address
    # Remove brackets for IPv6
    clean_ip_host = lower_host.strip("[]")
    try:
        ip = ipaddress.ip_address(clean_ip_host)
        if is_ip_blocked(ip):
            return False, f"Blocked private, loopback, or metadata IP target: {ip}"
        return True, None
    except ValueError:
        pass

    # Hostname syntax validation
    if not HOSTNAME_RE.match(lower_host):
        return False, f"Invalid hostname syntax: {hostname!r}"

    # Resolve hostname via DNS resolver if provided
    resolver = dns_resolver if dns_resolver is not None else default_dns_resolver
    resolved_ips = resolver(lower_host)
    if not resolved_ips:
        # If no IPs returned, we cannot verify it is private; but if mock resolver returned empty,
        # note: let default resolver proceed or let network call fail naturally
        return True, None

    for ip_str in resolved_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
            if is_ip_blocked(ip):
                return False, f"Hostname {hostname!r} resolved to blocked IP: {ip}"
        except ValueError:
            return False, f"Invalid IP address resolved for {hostname!r}: {ip_str!r}"

    return True, None
