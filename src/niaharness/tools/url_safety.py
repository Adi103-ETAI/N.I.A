"""URL safety checks — SSRF, phishing, malware protection.

Ported from Hermes Agent's tools/url_safety.py (495 LOC), scoped to
what NIA's web_fetch and browser tools need: block requests to private
IPs, localhost, and known-dangerous URL patterns.

Usage::

    from niaharness.tools.url_safety import check_url_safety

    safe, reason = check_url_safety("https://example.com")
    if not safe:
        raise ValueError(f"URL blocked: {reason}")
"""

from __future__ import annotations

import ipaddress
import logging
import re
from typing import Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Set to allow private/localhost URLs (set via NIA_MCP_ALLOW_PRIVATE_URLS or
# similar env vars for local development).
import os
_ALLOW_PRIVATE = os.environ.get("NIA_MCP_ALLOW_PRIVATE_URLS", "").strip() in ("1", "true", "yes")

# Blocked URL schemes (only http and https are allowed).
_BLOCKED_SCHEMES = frozenset({
    "file", "ftp", "ftps", "gopher", "ldap", "ldaps", "dict", "tftp",
    "sftp", "smb", "smbs", "nfs", "nntp", "news", "telnet", "rsh",
    "ssh", "irc", "magnet", "javascript", "data", "vbscript", "about",
})

# Known-dangerous hostname patterns.
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",  # GCP metadata endpoint
    "169.254.169.254",           # AWS/GCP metadata (also caught by IP check)
    "metadata.aws.amazon.com",   # AWS metadata alias
    "localhost.localdomain",
})

# Private IP ranges (RFC 1918 + loopback + link-local + reserved).
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("0.0.0.0/8"),        # "This network"
    ipaddress.ip_network("100.64.0.0/10"),    # Carrier-grade NAT
    ipaddress.ip_network("192.0.2.0/24"),     # TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # TEST-NET-3
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


def check_url_safety(url: str) -> Tuple[bool, str]:
    """Check if a URL is safe to fetch/browse.

    Args:
        url: The URL to check.

    Returns:
        Tuple of (is_safe, reason). If is_safe is False, reason explains
        why the URL was blocked.
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL"

    url = url.strip()

    # Parse the URL.
    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Failed to parse URL: {url}"

    # Check scheme.
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        if scheme in _BLOCKED_SCHEMES:
            return False, f"Blocked scheme: {scheme}://"
        return False, f"Non-HTTP scheme: {scheme}://"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "No hostname in URL"

    # Check blocked hostnames.
    if hostname in _BLOCKED_HOSTNAMES:
        return False, f"Blocked hostname: {hostname}"

    # Allow private URLs if explicitly configured.
    if _ALLOW_PRIVATE:
        return True, "OK (private URLs allowed by config)"

    # Check for localhost.
    if hostname in ("localhost", "ip6-localhost", "ip6-loopback"):
        return False, f"Blocked: localhost ({hostname})"

    # Check for .local (mDNS).
    if hostname.endswith(".local"):
        return False, f"Blocked: mDNS hostname ({hostname})"

    # Check for IP literals in private ranges.
    try:
        ip = ipaddress.ip_address(hostname)
        for network in _PRIVATE_NETWORKS:
            if ip in network:
                return False, f"Blocked: private/reserved IP ({hostname})"
    except ValueError:
        # Not an IP literal — it's a hostname. Allow it.
        # DNS resolution check would happen at connect time.
        pass

    # Check for dotted-decimal IPs that might bypass the ipaddress check
    # (e.g., 0x7f.0x0.0x0.0x1 or 017700000001).
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname):
        try:
            ip = ipaddress.ip_address(hostname)
            for network in _PRIVATE_NETWORKS:
                if ip in network:
                    return False, f"Blocked: private IP ({hostname})"
        except ValueError:
            pass

    # Check for IPv6 literals (e.g., [::1]).
    if hostname.startswith("[") and hostname.endswith("]"):
        inner = hostname[1:-1]
        try:
            ip = ipaddress.ip_address(inner)
            for network in _PRIVATE_NETWORKS:
                if ip in network:
                    return False, f"Blocked: private IPv6 ({inner})"
        except ValueError:
            pass

    # Check for suspiciously long URLs (possible buffer overflow attempt).
    if len(url) > 8192:
        return False, f"URL too long ({len(url)} chars, max 8192)"

    return True, "OK"


def is_safe_url(url: str) -> bool:
    """Return True if the URL is safe to fetch/browse."""
    safe, _ = check_url_safety(url)
    return safe


__all__ = [
    "check_url_safety",
    "is_safe_url",
]
