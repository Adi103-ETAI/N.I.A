"""Security checks for user-configured MCP server entries.

Ported from Hermes Agent's hermes_cli/mcp_security.py.

Blocks three high-signal abuse shapes:
1. Network exfiltration: a shell interpreter whose inline script invokes
   network egress tooling (curl, wget, nc, /dev/tcp).
2. OS persistence: a shell interpreter whose inline script writes to
   authorized_keys, .ssh/, /etc/pam.d, /etc/sudoers, crontab, shell rc
   files (the June 2026 hermes-0day campaign shape).
3. Hardcoded IOC blocklist for the hermes-0day campaign (attacker SSH
   key + source IPs).

These checks run at spawn time (_connect_stdio) so a hand-edited or
pre-planned entry is caught before it can execute.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

logger = logging.getLogger(__name__)

_SHELL_INTERPRETERS = frozenset({
    "bash",
    "sh",
    "zsh",
    "dash",
    "fish",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
})

_EGRESS_PATTERN = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b"
    r"|\bInvoke-RestMethod\b"
    r"|\bSystem\.Net\.WebClient\b",
    re.IGNORECASE,
)

_EXFIL_HINT_PATTERN = re.compile(
    r"\.env\b|--data-binary|--data-raw|\b-X\s+POST\b|\bPOST\b|<\s*[^\s]+",
    re.IGNORECASE,
)

_PERSISTENCE_PATTERN = re.compile(
    r"authorized_keys"
    r"|\.ssh/"
    r"|/etc/ssh\b"
    r"|/etc/pam\.d\b|pam_[\w-]+\.so"
    r"|/etc/sudoers"
    r"|/etc/cron|crontab\b"
    r"|/etc/rc\.local|/etc/systemd"
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",
    re.IGNORECASE,
)

# Indicators of compromise: June 2026 hermes-0day campaign
_IOC_SUBSTRINGS = (
    "AAAAC3NzaC1lZDI1NTE5AAAAICBoh1oDC4DnsO1m5mJ4yfEKrQebaFh",
    "hermes-0day",
    "60.165.167.",
    "118.182.244.156",
    "61.178.123.196",
)


def _command_basename(command: Any) -> str:
    """Extract the basename from a command string."""
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    first = parts[0] if parts else text
    return os.path.basename(first).lower()


def _inline_script(args: Any) -> str:
    """Flatten args into a single string for pattern matching."""
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(item) for item in args)
    return str(args)


def _entry_text(command: Any, args: Any, env: Any) -> str:
    """Flatten command + args + env values into one string for IOC scanning."""
    parts: list[str] = [str(command or "")]
    parts.append(_inline_script(args))
    if isinstance(env, dict):
        parts.extend(str(v) for v in env.values())
    return " ".join(parts)


def validate_mcp_stdio_command(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Return security warnings for an MCP stdio server entry.

    Empty return means the entry is not suspicious. Non-empty means the
    entry matches a known attack shape and should be blocked.

    Args:
        name: The MCP server name (for error messages).
        command: The command to execute (e.g. "bash", "python", "npx").
        args: The command arguments.
        env: The environment variables for the subprocess.

    Returns:
        A list of warning strings. Empty = safe, non-empty = suspicious.
    """
    issues: list[str] = []

    # 1. Hardcoded IOC blocklist — applies regardless of command shape.
    flat = _entry_text(command, args, env)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            issues.append(
                f"MCP server '{name}' contains a known indicator-of-compromise "
                f"('{ioc}') — refusing to spawn"
            )
            return issues

    # 2. Shell interpreter checks — only apply when the command IS a shell.
    basename = _command_basename(command)
    if basename not in _SHELL_INTERPRETERS:
        return issues

    script = _inline_script(args)
    if not script:
        return issues

    # 3. Network exfiltration shape.
    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    # 4. OS persistence shape (SSH key / PAM / sudoers / cron / rc files).
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is a known backdoor shape, not a real MCP server"
        )

    return issues


def is_mcp_stdio_suspicious(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Return True if the MCP stdio server entry is suspicious."""
    return bool(validate_mcp_stdio_command(name, command, args, env))


__all__ = [
    "is_mcp_stdio_suspicious",
    "validate_mcp_stdio_command",
]
