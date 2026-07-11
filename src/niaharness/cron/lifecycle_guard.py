"""P1 Cron lifecycle guard — block gateway-lifecycle commands at job-creation time.

Ported from Hermes Agent's ``cron/lifecycle_guard.py`` (142 LOC).

An agent running inside a gateway can schedule a cron job that calls
``nia gateway restart`` (or ``launchctl kickstart ai.nia.gateway`` or
``systemctl restart nia-gateway``). When the cron fires, the gateway
dies, the supervisor (launchd KeepAlive / systemd Restart=) revives it,
auto-resume picks up the offending session, and the resumed turn
re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command. It is enforced at
``cron.upsert_cron_job`` so it fires on every job-creation path.

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``nia gateway``, ``launchctl ... nia-gateway``,
``systemctl ... nia-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each
# branch is anchored on a concrete command identifier so a match can
# only fire on actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `nia gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside
    # a gateway is benign (a no-op or "already running" error).
    r"(?:nia\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a nia-gateway label. macOS launchd
    # labels look like `ai.nia.gateway` / `nia-gateway`.
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart)\b[^\n]*\bnia[.\-]?gateway)"
    # Branch C: systemctl ops on a nia-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bnia[.\-]?gateway)"
    # Branch D: pkill / kill targeting the nia gateway process.
    r"|(?:p?kill\b[^\n]*\bnia\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bnia)"
)


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(text))


def _resolve_script_path(script_path: str) -> Path:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler resolves a bare/relative script path under
    ``<NIA_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will
    actually run.
    """
    try:
        from niaharness.config.paths import get_nia_home
        nia_home = Path(get_nia_home())
    except Exception:
        import os
        nia_home = Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        return raw
    return nia_home / "scripts" / raw


def _read_script_for_scanning(script_path: str) -> str:
    """Read a script file for lifecycle-pattern scanning.

    Decodes with ``errors="replace"`` so binary or non-UTF-8 content
    does not silently bypass the check. Returns an empty string only
    when the file cannot be read at all.
    """
    try:
        return _resolve_script_path(script_path).read_bytes().decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a
    gateway-lifecycle command pattern.

    ``prompt`` is scanned directly. ``script``, when supplied, is read
    from disk and concatenated for the scan. Both are considered
    together so a job cannot slip through by splitting the command
    across the prompt and the script.
    """
    combined = prompt or ""
    if script:
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if contains_gateway_lifecycle_command(combined):
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command "
            "(restart/stop/kill). This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision. "
            "Run `nia gateway restart` from a shell outside the running "
            "gateway instead."
        )


__all__ = [
    "GatewayLifecycleBlocked",
    "check_gateway_lifecycle",
    "contains_gateway_lifecycle_command",
]
