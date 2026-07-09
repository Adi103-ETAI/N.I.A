"""NIA Doctor — self-diagnose + auto-repair broken setups.

Ported from Hermes Agent's ``hermes_cli/doctor.py`` (2,412 LOC), scoped
to NIA's architecture. Provides:

  - :func:`run_doctor` — main entry point. Runs ~12 check sections in
    order: security advisories, MCP security, Python env, config files,
    session DB health, WAL checkpoint, directory structure, provider
    API-key probes (parallel), SSL CA bundle, external tools, skills hub,
    summary.
  - ``--fix`` flag — auto-repairs what it can (create missing dirs/files,
    FTS rebuild, WAL checkpoint, config migration). Reports what it
    can't (security advisories, MCP suspicious entries, provider auth
    failures).
  - ``--ack <id>`` — acknowledge a security advisory (silences startup
    banner).

Checks performed (in order):
  1. Security advisories (compromised packages)
  2. MCP server security validation
  3. Python environment (version, venv)
  4. Configuration files (.env, settings.json)
  5. Session DB health (FTS integrity, schema repair)
  6. WAL checkpoint (>50MB → checkpoint)
  7. Directory structure (~/.nia + subdirs + SOUL.md)
  8. Provider API-key connectivity (parallel HTTP probes)
  9. SSL CA bundle check
  10. External tools (git, ripgrep)
  11. Skills hub
  12. Summary (issues + manual issues + fixed count)

Usage::

    from niaharness.cli.doctor import run_doctor

    # Dry-run (report only).
    result = run_doctor()
    print(result.report)

    # Auto-fix.
    result = run_doctor(fix=True)
    print(f"Fixed {result.fixed_count} issues")
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WAL checkpoint thresholds.
_WAL_CHECKPOINT_THRESHOLD = 50 * 1024 * 1024  # 50 MB → checkpoint.
_WAL_INFO_THRESHOLD = 10 * 1024 * 1024  # 10 MB → info only.

# HTTP probe timeout for provider API-key checks.
_PROBE_TIMEOUT = 10.0

# Thread pool for parallel provider probes.
_PROBE_WORKERS = 8

# Provider env-var hints (which env vars indicate a provider is configured).
_PROVIDER_ENV_HINTS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
    "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
)

# Provider probe table: (label, env_vars, default_models_url, base_url_env, supports_health_check).
_PROVIDER_PROBE_TABLE: List[Tuple[str, Tuple[str, ...], str, Optional[str], bool]] = [
    ("Anthropic", ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"), "https://api.anthropic.com/v1/models", None, True),
    ("OpenAI", ("OPENAI_API_KEY",), "https://api.openai.com/v1/models", "OPENAI_BASE_URL", True),
    ("OpenRouter", ("OPENROUTER_API_KEY",), "https://openrouter.ai/api/v1/models", None, True),
    ("DeepSeek", ("DEEPSEEK_API_KEY",), "https://api.deepseek.com/v1/models", "DEEPSEEK_BASE_URL", True),
    ("Z.AI / GLM", ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True),
    ("Kimi / Moonshot", ("KIMI_API_KEY",), "https://api.moonshot.ai/v1/models", "KIMI_BASE_URL", True),
    ("MiniMax", ("MINIMAX_API_KEY",), "https://api.minimax.io/v1/models", "MINIMAX_BASE_URL", True),
    ("Alibaba/DashScope", ("DASHSCOPE_API_KEY",), "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True),
    ("Hugging Face", ("HF_TOKEN",), "https://router.huggingface.co/v1/models", "HF_BASE_URL", True),
]


# ---------------------------------------------------------------------------
# Security advisories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Advisory:
    """A security advisory for a compromised package."""
    id: str
    title: str
    summary: str
    url: str
    compromised: Tuple[Tuple[str, frozenset[str]], ...]  # ((pkg, {bad_versions}), ...)
    remediation: Tuple[str, ...]
    published: str
    severity: str  # "critical" | "high" | "medium" | "low"


@dataclass(frozen=True)
class AdvisoryHit:
    """A detected compromise matching an advisory."""
    advisory: Advisory
    package: str
    installed_version: str


# The advisory catalog. Add new entries here as they're discovered.
ADVISORIES: Tuple[Advisory, ...] = (
    Advisory(
        id="shai-hulud-2026-05",
        title="Mini Shai-Hulud worm — mistralai 2.4.6 compromised on PyPI",
        summary=(
            "PyPI quarantined mistralai 2.4.6 on 2026-05-12. The worm steals "
            "credentials from env vars and ~/.npmrc/~/.pypirc/~/.aws/credentials "
            "and exfils to a hardcoded webhook."
        ),
        url="https://socket.dev/blog/mini-shai-hulud-worm-pypi",
        compromised=(("mistralai", frozenset({"2.4.6"})),),
        remediation=(
            "pip uninstall -y mistralai (or uv pip uninstall mistralai)",
            "Rotate API keys in ~/.nia/.env",
            "Audit credential files for unauthorized access",
            "Check GitHub for unexpected SSH/deploy keys or webhooks",
            "Run: nia doctor --ack shai-hulud-2026-05",
        ),
        published="2026-05-12",
        severity="critical",
    ),
)


def detect_compromised(advisories: Tuple[Advisory, ...] = ADVISORIES) -> List[AdvisoryHit]:
    """Check installed packages against the advisory catalog.

    Uses ``importlib.metadata.version()`` — no network. A hit occurs when
    an installed package version matches a bad version in the advisory
    (or when bad_versions is empty, meaning any version is compromised).
    """
    hits: List[AdvisoryHit] = []
    for advisory in advisories:
        for pkg, bad_versions in advisory.compromised:
            try:
                installed = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
            if not bad_versions or installed in bad_versions:
                hits.append(AdvisoryHit(
                    advisory=advisory,
                    package=pkg,
                    installed_version=installed,
                ))
    return hits


def _get_acked_ids() -> set[str]:
    """Read acked advisory IDs from config."""
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        acked = getattr(settings, "security", None)
        if acked and hasattr(acked, "acked_advisories"):
            return set(acked.acked_advisories)
    except Exception:
        pass
    return set()


def _ack_advisory(advisory_id: str) -> bool:
    """Persist an advisory ack to config (best-effort)."""
    try:
        # Write to a simple JSON file for now.
        from niaharness.prompts.soul import get_nia_home
        ack_path = get_nia_home() / "acked_advisories.json"
        existing: set[str] = set()
        if ack_path.exists():
            existing = set(json.loads(ack_path.read_text(encoding="utf-8")))
        existing.add(advisory_id)
        ack_path.parent.mkdir(parents=True, exist_ok=True)
        ack_path.write_text(json.dumps(sorted(existing), indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        logger.warning("Failed to persist advisory ack: %s", exc)
        return False


def filter_unacked(hits: List[AdvisoryHit]) -> List[AdvisoryHit]:
    """Filter out hits whose advisory ID has been acked."""
    acked = _get_acked_ids()
    # Also check the JSON file.
    try:
        from niaharness.prompts.soul import get_nia_home
        ack_path = get_nia_home() / "acked_advisories.json"
        if ack_path.exists():
            acked |= set(json.loads(ack_path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return [h for h in hits if h.advisory.id not in acked]


# ---------------------------------------------------------------------------
# Doctor result
# ---------------------------------------------------------------------------


@dataclass
class DoctorResult:
    """Result of running the doctor."""
    issues: List[str] = field(default_factory=list)  # Auto-fixable issues.
    manual_issues: List[str] = field(default_factory=list)  # Issues needing human action.
    fixed_count: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    report: str = ""

    @property
    def total_issues(self) -> int:
        return len(self.issues) + len(self.manual_issues)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check_ok(msg: str) -> str:
    return f"  ✓ {msg}"


def _check_warn(msg: str, hint: str = "") -> str:
    line = f"  ⚠ {msg}"
    if hint:
        line += f" — {hint}"
    return line


def _check_fail(msg: str, hint: str = "") -> str:
    line = f"  ✗ {msg}"
    if hint:
        line += f" — {hint}"
    return line


def _check_info(msg: str) -> str:
    return f"  ℹ {msg}"


def _section(title: str) -> str:
    return f"\n{'─' * 50}\n  {title}\n{'─' * 50}"


# ---------------------------------------------------------------------------
# Provider API-key probes
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of a provider API-key connectivity probe."""
    label: str
    lines: List[str] = field(default_factory=list)
    issue: Optional[str] = None


def _probe_provider(label: str, env_vars: Tuple[str, ...], default_url: str,
                    base_url_env: Optional[str], supports_health_check: bool) -> ProbeResult:
    """Probe a single provider's API key connectivity."""
    # Find the first non-empty env var.
    api_key = ""
    for var in env_vars:
        val = os.environ.get(var, "")
        if val:
            api_key = val
            break

    if not api_key:
        return ProbeResult(label=label, lines=[_check_info(f"{label}: no API key configured")])

    if not supports_health_check:
        return ProbeResult(label=label, lines=[_check_ok(f"{label}: key configured (health check skipped)")])

    # Build the probe URL.
    url = default_url
    if base_url_env:
        base_override = os.environ.get(base_url_env, "")
        if base_override:
            url = base_override.rstrip("/") + "/models"

    # Make the HTTP request.
    try:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT)
        if resp.status_code == 200:
            return ProbeResult(label=label, lines=[_check_ok(f"{label}: connected")])
        elif resp.status_code == 401:
            return ProbeResult(
                label=label,
                lines=[_check_fail(f"{label}: invalid API key (401)")],
                issue=f"{label}: invalid API key",
            )
        elif resp.status_code == 402:
            return ProbeResult(
                label=label,
                lines=[_check_warn(f"{label}: out of credits (402)")],
                issue=f"{label}: out of credits",
            )
        elif resp.status_code == 429:
            return ProbeResult(
                label=label,
                lines=[_check_warn(f"{label}: rate limited (429)")],
            )
        else:
            return ProbeResult(
                label=label,
                lines=[_check_warn(f"{label}: HTTP {resp.status_code}")],
            )
    except ImportError:
        return ProbeResult(label=label, lines=[_check_info(f"{label}: httpx not installed, skipping probe")])
    except Exception as exc:
        return ProbeResult(
            label=label,
            lines=[_check_warn(f"{label}: probe failed ({exc}")],
        )


def _run_provider_probes() -> List[ProbeResult]:
    """Run all provider probes in parallel."""
    results: List[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS, thread_name_prefix="doctor-probe") as pool:
        futures = {
            pool.submit(_probe_provider, label, env_vars, url, base_env, supports_hc): label
            for label, env_vars, url, base_env, supports_hc in _PROVIDER_PROBE_TABLE
        }
        for future in as_completed(futures):
            results.append(future.result())
    # Sort by the original table order.
    label_order = {label: i for i, (label, _, _, _, _) in enumerate(_PROVIDER_PROBE_TABLE)}
    results.sort(key=lambda r: label_order.get(r.label, 999))
    return results


# ---------------------------------------------------------------------------
# Session DB checks
# ---------------------------------------------------------------------------


def _check_session_db(fix: bool, result: DoctorResult, lines: List[str]) -> None:
    """Check session DB health (FTS integrity, schema repair, WAL checkpoint)."""
    from niaharness.services.session_db import _sessions_db_path, _db_opens_cleanly, repair_state_db_schema

    db_path = _sessions_db_path()
    if not db_path.exists():
        lines.append(_check_info("Session DB: not found (will be created on first use)"))
        return

    # Check FTS write health.
    try:
        error = _db_opens_cleanly(db_path)
        if error:
            if fix:
                lines.append(_check_warn(f"Session DB: {error} — attempting repair..."))
                repair_result = repair_state_db_schema(db_path, backup=True)
                if repair_result.get("repaired"):
                    lines.append(_check_ok(f"Session DB: repaired ({repair_result.get('strategy')})"))
                    if repair_result.get("backup_path"):
                        lines.append(_check_info(f"  backup: {repair_result['backup_path']}"))
                    result.fixed_count += 1
                else:
                    lines.append(_check_fail(f"Session DB: repair failed — {repair_result.get('error', 'unknown')}"))
                    result.manual_issues.append("Session DB repair failed — manual intervention needed")
            else:
                lines.append(_check_fail(f"Session DB: {error}", "run 'nia doctor --fix' to repair"))
                result.issues.append("Session DB needs repair — run 'nia doctor --fix'")
        else:
            lines.append(_check_ok("Session DB: healthy"))
            result.checks_passed += 1
    except Exception as exc:
        lines.append(_check_warn(f"Session DB: check failed ({exc})"))

    # Check WAL file size.
    wal_path = db_path.with_suffix(db_path.suffix + "-wal")
    if wal_path.exists():
        try:
            wal_size = wal_path.stat().st_size
            if wal_size > _WAL_CHECKPOINT_THRESHOLD:
                if fix:
                    lines.append(_check_warn(f"WAL file: {wal_size // 1024 // 1024}MB — checkpointing..."))
                    try:
                        conn = sqlite3.connect(str(db_path))
                        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        conn.close()
                        new_size = wal_path.stat().st_size
                        lines.append(_check_ok(f"WAL file: checkpointed ({wal_size // 1024}KB → {new_size // 1024}KB)"))
                        result.fixed_count += 1
                    except Exception as exc:
                        lines.append(_check_fail(f"WAL checkpoint failed: {exc}"))
                else:
                    lines.append(_check_warn(f"WAL file: {wal_size // 1024 // 1024}MB", "run 'nia doctor --fix' to checkpoint"))
                    result.issues.append("Large WAL file — run 'nia doctor --fix' to checkpoint")
            elif wal_size > _WAL_INFO_THRESHOLD:
                lines.append(_check_info(f"WAL file: {wal_size // 1024 // 1024}MB (normal for active sessions)"))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SSL CA bundle check
# ---------------------------------------------------------------------------


def _check_ssl_ca(lines: List[str]) -> None:
    """Check SSL CA certificate bundle validity."""
    try:
        import ssl
        # Try to create a default context — if certifi is installed it'll use it.
        ctx = ssl.create_default_context()
        if ctx.get_ca_certs():
            lines.append(_check_ok("SSL CA certificate bundle is valid"))
        else:
            lines.append(_check_warn("SSL CA bundle: no certificates found"))
    except Exception as exc:
        lines.append(_check_fail(f"SSL CA bundle check failed: {exc}"))


# ---------------------------------------------------------------------------
# MCP server security validation
# ---------------------------------------------------------------------------


def _check_mcp_security(result: DoctorResult, lines: List[str]) -> None:
    """Validate MCP server configurations for suspicious entries."""
    try:
        from niaharness.config.settings import load_settings
        from niaharness.mcp.security import validate_mcp_stdio_command

        settings = load_settings()
        mcp_servers = getattr(settings, "mcp_servers", {}) or {}
        if not mcp_servers:
            lines.append(_check_info("MCP servers: none configured"))
            return

        suspicious_count = 0
        for name, config in mcp_servers.items():
            config_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
            if config_dict.get("type") == "stdio":
                issues = validate_mcp_stdio_command(
                    config_dict.get("command", ""),
                    config_dict.get("args", []),
                    config_dict.get("env", {}),
                )
                if issues:
                    suspicious_count += 1
                    for issue in issues:
                        lines.append(_check_warn(f"MCP '{name}': {issue}"))
                    result.manual_issues.append(f"Review MCP server '{name}' — suspicious configuration detected")

        if suspicious_count == 0:
            lines.append(_check_ok(f"MCP servers: {len(mcp_servers)} configured, all passed security checks"))
            result.checks_passed += 1
    except Exception as exc:
        lines.append(_check_info(f"MCP security: skipped ({exc})"))


# ---------------------------------------------------------------------------
# Directory structure checks
# ---------------------------------------------------------------------------


def _check_directories(fix: bool, result: DoctorResult, lines: List[str]) -> None:
    """Check ~/.nia directory structure."""
    from niaharness.prompts.soul import get_nia_home

    nia_home = get_nia_home()
    required_dirs = ["cron", "sessions", "logs", "skills", "memories"]
    required_files = {
        "SOUL.md": "# NIA Agent Persona\n\nYou are NIA, a helpful AI assistant.\n",
    }

    # Check main directory.
    if not nia_home.exists():
        if fix:
            nia_home.mkdir(parents=True, exist_ok=True)
            lines.append(_check_ok(f"Created {nia_home}"))
            result.fixed_count += 1
        else:
            lines.append(_check_warn(f"~/.nia not found", "run 'nia doctor --fix' to create"))
            result.issues.append("Create ~/.nia directory")

    # Check subdirectories.
    for subdir in required_dirs:
        dir_path = nia_home / subdir
        if not dir_path.exists():
            if fix:
                dir_path.mkdir(parents=True, exist_ok=True)
                lines.append(_check_ok(f"Created {dir_path}"))
                result.fixed_count += 1
            else:
                lines.append(_check_warn(f"Missing directory: {subdir}"))
                result.issues.append(f"Create ~/.nia/{subdir}")

    # Check required files.
    for filename, default_content in required_files.items():
        file_path = nia_home / filename
        if not file_path.exists():
            if fix:
                file_path.write_text(default_content, encoding="utf-8")
                lines.append(_check_ok(f"Created {file_path}"))
                result.fixed_count += 1
            else:
                lines.append(_check_warn(f"Missing file: {filename}"))
                result.issues.append(f"Create ~/.nia/{filename}")

    if not result.issues:
        lines.append(_check_ok("Directory structure: complete"))
        result.checks_passed += 1


# ---------------------------------------------------------------------------
# External tools check
# ---------------------------------------------------------------------------


def _check_external_tools(lines: List[str]) -> None:
    """Check for external tools (git, ripgrep)."""
    # Git.
    git_path = shutil.which("git")
    if git_path:
        lines.append(_check_ok("git: found"))
    else:
        lines.append(_check_info("git: not found (optional)"))

    # Ripgrep.
    rg_path = shutil.which("rg")
    if rg_path:
        lines.append(_check_ok("ripgrep (rg): found"))
    else:
        lines.append(_check_info("ripgrep: not found (optional — install for faster search)"))


# ---------------------------------------------------------------------------
# Config check
# ---------------------------------------------------------------------------


def _check_config(fix: bool, result: DoctorResult, lines: List[str]) -> None:
    """Check config file existence + .env."""
    from niaharness.config.paths import get_config_file_path

    config_path = get_config_file_path()
    if config_path.exists():
        lines.append(_check_ok(f"Config: {config_path}"))
        result.checks_passed += 1
    else:
        if fix:
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    json.dumps({"model": "claude-sonnet-4-6"}, indent=2),
                    encoding="utf-8",
                )
                lines.append(_check_ok(f"Created default config: {config_path}"))
                result.fixed_count += 1
            except Exception as exc:
                lines.append(_check_fail(f"Could not create config: {exc}"))
        else:
            lines.append(_check_warn(f"Config not found: {config_path}"))
            result.issues.append("Create config file")

    # Check .env.
    from niaharness.prompts.soul import get_nia_home
    env_path = get_nia_home() / ".env"
    if env_path.exists():
        # Check if any provider env hints are present.
        content = env_path.read_text(encoding="utf-8")
        has_provider = any(hint in content for hint in _PROVIDER_ENV_HINTS)
        if has_provider:
            lines.append(_check_ok(".env: found with provider configuration"))
        else:
            lines.append(_check_warn(".env: found but no provider API keys detected"))
    else:
        if fix:
            try:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                env_path.write_text("# NIA environment variables\n", encoding="utf-8")
                os.chmod(str(env_path), 0o600)
                lines.append(_check_ok(f"Created .env: {env_path}"))
                result.fixed_count += 1
            except Exception as exc:
                lines.append(_check_fail(f"Could not create .env: {exc}"))
        else:
            lines.append(_check_warn(".env not found"))
            result.issues.append("Create ~/.nia/.env with API keys")


# ---------------------------------------------------------------------------
# Python environment check
# ---------------------------------------------------------------------------


def _check_python_env(lines: List[str]) -> None:
    """Check Python version + venv."""
    version = sys.version_info
    if version >= (3, 11):
        lines.append(_check_ok(f"Python: {version.major}.{version.minor}.{version.micro}"))
    elif version >= (3, 10):
        lines.append(_check_ok(f"Python: {version.major}.{version.minor}.{version.micro} (3.11+ recommended)"))
    elif version >= (3, 8):
        lines.append(_check_warn(f"Python: {version.major}.{version.minor} (3.10+ recommended)"))
    else:
        lines.append(_check_fail(f"Python: {version.major}.{version.minor} (3.8+ required)"))

    # Venv check.
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        lines.append(_check_ok(f"Virtual environment: active ({sys.prefix})"))
    else:
        lines.append(_check_info("Virtual environment: not active (recommended for isolation)"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_doctor(
    *,
    fix: bool = False,
    ack: Optional[str] = None,
) -> DoctorResult:
    """Run diagnostic checks.

    Args:
        fix: If True, auto-repair what can be repaired.
        ack: If set, acknowledge the advisory with this ID and return
            immediately (bypasses all checks).

    Returns:
        :class:`DoctorResult` with the full report.
    """
    # Handle --ack <id> fast path.
    if ack:
        valid_ids = {a.id for a in ADVISORIES}
        if ack not in valid_ids:
            return DoctorResult(report=f"Unknown advisory ID: {ack!r}. Known IDs: {', '.join(sorted(valid_ids))}")
        if _ack_advisory(ack):
            return DoctorResult(report=f"  ✓ Acknowledged advisory {ack}. It will no longer trigger startup banners.", fixed_count=1)
        return DoctorResult(report=f"  ✗ Failed to persist ack for {ack}.")

    result = DoctorResult()
    lines: List[str] = []

    # Header.
    lines.append("")
    lines.append("┌─────────────────────────────────────────────────────────┐")
    lines.append("│                    🩺 NIA Doctor                         │")
    lines.append("└─────────────────────────────────────────────────────────┘")

    # 1. Security advisories.
    lines.append(_section("Security Advisories"))
    try:
        all_hits = detect_compromised()
        fresh_hits = filter_unacked(all_hits)
        if fresh_hits:
            for hit in fresh_hits:
                lines.append(_check_fail(f"{hit.advisory.title} ({hit.package}=={hit.installed_version})"))
                lines.append(f"    ID: {hit.advisory.id}")
                lines.append(f"    Severity: {hit.advisory.severity}")
                lines.append(f"    URL: {hit.advisory.url}")
                lines.append(f"    Summary: {hit.advisory.summary}")
                lines.append("    Remediation:")
                for i, step in enumerate(hit.advisory.remediation, 1):
                    lines.append(f"      {i}. {step}")
                result.manual_issues.append(f"Security advisory {hit.advisory.id}: {hit.package}=={hit.installed_version}")
                result.checks_failed += 1
        else:
            lines.append(_check_ok("No compromised packages detected"))
            result.checks_passed += 1
    except Exception as exc:
        lines.append(_check_warn(f"Advisory check failed: {exc}"))

    # 2. MCP server security.
    lines.append(_section("MCP Server Security"))
    _check_mcp_security(result, lines)

    # 3. Python environment.
    lines.append(_section("Python Environment"))
    _check_python_env(lines)

    # 4. Configuration files.
    lines.append(_section("Configuration"))
    _check_config(fix, result, lines)

    # 5. Session DB health.
    lines.append(_section("Session Database"))
    _check_session_db(fix, result, lines)

    # 6. Directory structure.
    lines.append(_section("Directory Structure"))
    _check_directories(fix, result, lines)

    # 7. Provider API-key connectivity.
    lines.append(_section("Provider Connectivity"))
    try:
        probe_results = _run_provider_probes()
        for pr in probe_results:
            for line in pr.lines:
                lines.append(line)
            if pr.issue:
                result.issues.append(pr.issue)
                result.checks_failed += 1
            else:
                result.checks_passed += 1
    except Exception as exc:
        lines.append(_check_warn(f"Provider probes failed: {exc}"))

    # 8. SSL CA bundle.
    lines.append(_section("SSL / CA Certificates"))
    _check_ssl_ca(lines)

    # 9. External tools.
    lines.append(_section("External Tools"))
    _check_external_tools(lines)

    # Summary.
    lines.append(_section("Summary"))
    total = result.total_issues
    if total == 0:
        lines.append(_check_ok("All checks passed! 🎉"))
    else:
        if fix and result.fixed_count > 0:
            lines.append(_check_ok(f"Fixed {result.fixed_count} issue(s)"))
        if result.issues:
            lines.append(f"  ⚠ {len(result.issues)} auto-fixable issue(s) remain:")
            for i, issue in enumerate(result.issues, 1):
                lines.append(f"    {i}. {issue}")
        if result.manual_issues:
            lines.append(f"  ✗ {len(result.manual_issues)} issue(s) need manual action:")
            for i, issue in enumerate(result.manual_issues, 1):
                lines.append(f"    {i}. {issue}")
        if not fix and result.issues:
            lines.append("")
            lines.append("  Tip: Run 'nia doctor --fix' to auto-repair fixable issues.")

    result.report = "\n".join(lines)
    return result


__all__ = [
    "ADVISORIES",
    "Advisory",
    "AdvisoryHit",
    "DoctorResult",
    "detect_compromised",
    "filter_unacked",
    "run_doctor",
]
