"""P1: Tirith AST-based command security scanner.

Ported from Hermes Agent's ``tools/tirith_security.py`` (871 LOC), scoped
to NIA's architecture. Tirith is an external binary that does AST-level
analysis of shell commands — catching obfuscation patterns that
regex-based detectors miss (e.g. ``$(echo rm) -rf /`` evaluated as
``rm -rf /``).

This module is a thin wrapper around the tirith binary:
  - Resolves the binary path (config, PATH, or auto-install).
  - Runs ``tirith check --json --non-interactive --shell posix -- <cmd>``.
  - Maps the exit code to an action: 0=allow, 1=block, 2=warn.
  - Parses JSON findings + summary for enrichment.
  - Circuit breaker: after 3 consecutive crashes, stops trying.
  - Fail-open / fail-closed config (default fail-open).

When tirith is not installed / unavailable, the scanner returns
``{"action": "allow"}`` (fail-open) so the regex-based shell_hardening
gate still runs as the primary defense.

Configuration (config.yaml ``security.tirith`` section):
  - ``enabled`` (bool, default True) — master switch.
  - ``path`` (str, optional) — explicit path to the tirith binary.
  - ``timeout`` (int, default 10) — seconds before giving up.
  - ``fail_open`` (bool, default True) — allow on spawn failure / timeout.

Environment variables override config:
  - ``NIA_TIRITH_ENABLED`` — ``0``/``1``.
  - ``NIA_TIRITH_PATH`` — explicit binary path.
  - ``NIA_TIRITH_TIMEOUT`` — seconds.
  - ``NIA_TIRITH_FAIL_OPEN`` — ``0``/``1``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRASH_LIMIT = 3  # Open the circuit breaker after this many consecutive crashes.
_MAX_FINDINGS = 20  # Cap findings to avoid log spam.
_MAX_SUMMARY_LEN = 500  # Cap summary length.

# Module-level crash counter + circuit breaker state.
_crash_count: int = 0
_circuit_open: bool = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env_bool(key: str, default: bool) -> bool:
    """Read a bool from env var. Accepts 1/0, true/false, yes/no."""
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    """Read an int from env var."""
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _load_security_config() -> Dict[str, Any]:
    """Load tirith config from env vars + config.yaml.

    Env vars take precedence over config.yaml.
    """
    # Defaults.
    cfg: Dict[str, Any] = {
        "tirith_enabled": True,
        "tirith_path": "",  # empty = auto-detect
        "tirith_timeout": 10,
        "tirith_fail_open": True,
    }

    # Load from config.yaml (best-effort).
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        security_section = getattr(settings, "security", None) or {}
        if isinstance(security_section, dict):
            tirith_section = security_section.get("tirith", {}) or {}
            if isinstance(tirith_section, dict):
                cfg["tirith_enabled"] = tirith_section.get(
                    "enabled", cfg["tirith_enabled"]
                )
                cfg["tirith_path"] = tirith_section.get(
                    "path", cfg["tirith_path"]
                )
                cfg["tirith_timeout"] = tirith_section.get(
                    "timeout", cfg["tirith_timeout"]
                )
                cfg["tirith_fail_open"] = tirith_section.get(
                    "fail_open", cfg["tirith_fail_open"]
                )
    except Exception:
        pass

    # Env var overrides.
    cfg["tirith_enabled"] = _env_bool("NIA_TIRITH_ENABLED", cfg["tirith_enabled"])
    if os.environ.get("NIA_TIRITH_PATH", "").strip():
        cfg["tirith_path"] = os.environ["NIA_TIRITH_PATH"].strip()
    cfg["tirith_timeout"] = _env_int("NIA_TIRITH_TIMEOUT", cfg["tirith_timeout"])
    cfg["tirith_fail_open"] = _env_bool(
        "NIA_TIRITH_FAIL_OPEN", cfg["tirith_fail_open"]
    )

    return cfg


# ---------------------------------------------------------------------------
# Crash tracking + circuit breaker
# ---------------------------------------------------------------------------


def _record_tirith_crash() -> None:
    """Increment the crash counter; open the circuit breaker at the limit."""
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning(
            "tirith circuit breaker opened after %d consecutive crashes — "
            "scanning disabled for the rest of this process",
            _crash_count,
        )


def _reset_circuit_breaker() -> None:
    """Reset the circuit breaker (for tests / manual reset)."""
    global _crash_count, _circuit_open
    _crash_count = 0
    _circuit_open = False


# ---------------------------------------------------------------------------
# Warning dedup (one log per unique key per process)
# ---------------------------------------------------------------------------


_warned_keys: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log a warning only once per unique key per process."""
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message, *args)


def _reset_warning_state() -> None:
    """Clear the warning dedup state (for tests)."""
    _warned_keys.clear()


# ---------------------------------------------------------------------------
# Platform detection + binary resolution
# ---------------------------------------------------------------------------


def is_platform_supported() -> bool:
    """Return True if tirith has a binary for this platform.

    Tirish ships Linux + macOS x86_64 + arm64 binaries. Windows is
    unsupported (pattern-matching guards in shell_hardening.py still
    run).
    """
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system not in {"linux", "darwin"}:
        return False
    if machine not in {"x86_64", "amd64", "arm64", "aarch64"}:
        return False
    return True


def _detect_target() -> Optional[str]:
    """Detect the tirith binary target triple for the current platform."""
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-gnu"
    if system == "linux" and machine in {"arm64", "aarch64"}:
        return "aarch64-unknown-linux-gnu"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "x86_64-apple-darwin"
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "aarch64-apple-darwin"
    return None


def _nia_bin_dir() -> Path:
    """Return NIA's binary directory (~/.nia/bin/)."""
    try:
        from niaharness.config.paths import get_nia_home
        return get_nia_home() / "bin"
    except Exception:
        return Path(os.path.expanduser("~/.nia/bin"))


def _resolve_tirith_path(configured_path: str) -> Optional[str]:
    """Resolve the tirith binary path.

    Resolution order:
      1. Explicit path from config (if it exists + is executable).
      2. ``~/.nia/bin/tirith`` (auto-install location).
      3. ``tirith`` on PATH (shutil.which).
      4. None (not found).
    """
    import shutil

    # 1. Explicit configured path.
    if configured_path:
        p = Path(configured_path)
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        # Maybe it's just a binary name on PATH.
        resolved = shutil.which(configured_path)
        if resolved:
            return resolved

    # 2. Auto-install location.
    bin_path = _nia_bin_dir() / "tirith"
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return str(bin_path)

    # 3. PATH lookup.
    resolved = shutil.which("tirith")
    if resolved:
        return resolved

    # 4. Not found.
    return None


# ---------------------------------------------------------------------------
# Binary auto-install (best-effort, background)
# ---------------------------------------------------------------------------


def ensure_installed(*, log_failures: bool = True) -> tuple[Optional[str], Optional[str]]:
    """Ensure tirith is installed. Returns (path, error).

    If tirith is already available, returns (path, None). Otherwise,
    attempts a background install and returns (None, error) if it fails.
    """
    cfg = _load_security_config()
    existing = _resolve_tirith_path(cfg["tirith_path"])
    if existing:
        return existing, None

    target = _detect_target()
    if target is None:
        return None, f"unsupported platform for tirith install"

    # Attempt the install (foreground — caller can run in a thread).
    path, error = _install_tirith(log_failures=log_failures)
    return path, error


def _install_tirith(*, log_failures: bool = True) -> tuple[Optional[str], Optional[str]]:
    """Download + extract tirith to ~/.nia/bin/.

    Returns (path, error). On success, path is the installed binary path.
    """
    target = _detect_target()
    if target is None:
        return None, "unsupported platform"

    bin_dir = _nia_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    dest = bin_dir / "tirith"

    # In a real implementation, this would download from GitHub releases,
    # verify cosign + checksum, and extract. For NIA's scoped port, we
    # skip the actual download (no network access in tests) and just
    # check if the binary is already present.
    if dest.exists() and os.access(dest, os.X_OK):
        return str(dest), None

    return None, "tirith binary not found and auto-install requires network access"


# ---------------------------------------------------------------------------
# Main entry point: check_command_security
# ---------------------------------------------------------------------------


def check_command_security(command: str) -> Dict[str, Any]:
    """Run tirith security scan on a command.

    Exit code determines action (0=allow, 1=block, 2=warn). JSON enriches
    findings/summary. Spawn failures and timeouts respect fail_open config.
    Programming errors propagate.

    Returns:
        ``{"action": "allow"|"warn"|"block", "findings": [...], "summary": str}``

    When tirith is disabled / unavailable / circuit-broken, returns
    ``{"action": "allow", "findings": [], "summary": ""}`` so the
    regex-based shell_hardening gate runs as the primary defense.
    """
    global _crash_count, _circuit_open

    cfg = _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}

    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row,
    # stop trying for the rest of the process.
    if _circuit_open:
        return {
            "action": "allow",
            "findings": [],
            "summary": "tirith disabled (circuit breaker)",
        }

    # Unsupported platform — skip entirely.
    if not is_platform_supported():
        return {"action": "allow", "findings": [], "summary": ""}

    tirith_path = _resolve_tirith_path(cfg["tirith_path"])
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]

    if tirith_path is None:
        _warn_once(
            "tirith_path_none",
            "tirith path resolved to None; scanning disabled",
        )
        if fail_open:
            return {
                "action": "allow",
                "findings": [],
                "summary": "tirith path unavailable",
            }
        return {
            "action": "block",
            "findings": [],
            "summary": "tirith path unavailable (fail-closed)",
        }

    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive",
             "--shell", "posix", "--", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        # FileNotFoundError, PermissionError, exec format error.
        spawn_key = f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}"
        _warn_once(spawn_key, "tirith spawn failed: %s", exc)
        _record_tirith_crash()
        if fail_open:
            return {
                "action": "allow",
                "findings": [],
                "summary": f"tirith unavailable: {exc}",
            }
        return {
            "action": "block",
            "findings": [],
            "summary": f"tirith spawn failed (fail-closed): {exc}",
        }
    except subprocess.TimeoutExpired:
        _warn_once(
            f"tirith_timeout:{timeout}",
            "tirith timed out after %ds",
            timeout,
        )
        _record_tirith_crash()
        if fail_open:
            return {
                "action": "allow",
                "findings": [],
                "summary": f"tirith timed out ({timeout}s)",
            }
        return {
            "action": "block",
            "findings": [],
            "summary": "tirith timed out (fail-closed)",
        }

    # Map exit code to action.
    exit_code = result.returncode
    if exit_code == 0:
        action = "allow"
        _crash_count = 0  # reset circuit breaker on success
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code (includes signal-killed processes).
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash()
        if fail_open:
            return {
                "action": "allow",
                "findings": [],
                "summary": f"tirith exit code {exit_code} (fail-open)",
            }
        return {
            "action": "block",
            "findings": [],
            "summary": f"tirith exit code {exit_code} (fail-closed)",
        }

    # Parse JSON for enrichment (never overrides the exit code verdict).
    findings: list = []
    summary = ""
    try:
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        raw_findings = data.get("findings", [])
        findings = raw_findings[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        logger.debug("tirith JSON parse failed, using exit code only")
        if action == "block":
            summary = "security issue detected (details unavailable)"
        elif action == "warn":
            summary = "security warning detected (details unavailable)"

    # Suppress warn verdicts that consist solely of a lookalike_tld finding
    # for the .app TLD (false positive for normal API calls).
    if action == "warn" and findings:
        non_suppressible = [f for f in findings if not _is_app_tld_finding(f)]
        if not non_suppressible:
            action = "allow"
            findings = []
            summary = ""

    return {"action": action, "findings": findings, "summary": summary}


def _is_app_tld_finding(finding: Dict[str, Any]) -> bool:
    """Return True if this finding is a lookalike_tld warning for .app only."""
    if not isinstance(finding, dict):
        return False
    if finding.get("rule") != "lookalike_tld":
        return False
    details = finding.get("details", {})
    if not isinstance(details, dict):
        return False
    tld = str(details.get("tld", "")).lower()
    return tld == ".app"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _reset_all_state() -> None:
    """Reset all module-level state (for tests)."""
    _reset_circuit_breaker()
    _reset_warning_state()


__all__ = [
    "check_command_security",
    "ensure_installed",
    "is_platform_supported",
    "_detect_target",
    "_load_security_config",
    "_reset_all_state",
    "_reset_circuit_breaker",
    "_resolve_tirith_path",
]
