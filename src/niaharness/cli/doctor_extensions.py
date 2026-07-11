"""P1 Doctor extensions — 10 missing diagnostic sections.

Ported from Hermes Agent's ``hermes_cli/doctor.py`` (2412 LOC), scoped
to NIA's architecture. Provides the 10 missing doctor sections identified
in AUDIT.md:

  1. Version consistency check — pyproject.toml vs installed version.
  2. Gateway service linger check — systemd user service survival after logout.
  3. Tool availability check — verify registered tools are importable.
  4. Skills hub check — verify skill hub directories + lock file exist.
  5. Memory provider check — verify memory store + write gate are functional.
  6. Profiles check — list profiles + verify active profile is valid.
  7. Required packages check — verify all required Python packages are installed.
  8. Command installation check — verify `nia` command is on PATH.
  9. Config structure validation — verify config.yaml has required sections.
  10. xAI model retirement warning — warn if using retired xAI models.

Each function returns a list of (status, message, detail) tuples that the
main ``run_doctor`` function renders. status is "ok", "warn", "fail", or
"info".
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Status constants.
OK = "ok"
WARN = "warn"
FAIL = "fail"
INFO = "info"

CheckResult = Tuple[str, str, str]  # (status, message, detail)


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


def _read_pyproject_version() -> Optional[str]:
    """Read the version from pyproject.toml."""
    try:
        # Find pyproject.toml relative to the niaharness package.
        import niaharness
        pkg_dir = Path(niaharness.__file__).resolve().parent
        pyproject = pkg_dir.parent / "pyproject.toml"
        if not pyproject.exists():
            return None
        text = pyproject.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("version") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1. Version consistency check
# ---------------------------------------------------------------------------


def check_version_consistency() -> List[CheckResult]:
    """Verify pyproject.toml version matches the installed niaharness version."""
    results: List[CheckResult] = []
    try:
        from niaharness import __version__ as installed_version
    except Exception:
        return [(WARN, "Could not determine installed version", "")]

    pyproject_version = _read_pyproject_version()
    if pyproject_version is None:
        return [(INFO, "pyproject.toml not found (installed wheel)", "")]

    if pyproject_version == installed_version:
        results.append((OK, "Version files consistent", f"({installed_version})"))
    else:
        results.append((FAIL,
            "Version mismatch between source files",
            f"(pyproject.toml {pyproject_version} != installed {installed_version})",
        ))
    return results


# ---------------------------------------------------------------------------
# 2. Gateway service linger check
# ---------------------------------------------------------------------------


def check_gateway_service_linger() -> List[CheckResult]:
    """Warn when a systemd user gateway service will stop after logout."""
    results: List[CheckResult] = []
    # Check if we're on Linux with systemd.
    if sys.platform != "linux":
        results.append((INFO, "Gateway linger check skipped (non-Linux)", ""))
        return results

    # Check if systemd is available.
    if not shutil.which("systemctl"):
        results.append((INFO, "systemctl not found — linger check skipped", ""))
        return results

    # Check if a nia-gateway service exists.
    import subprocess
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-enabled", "nia-gateway"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            results.append((INFO, "No nia-gateway systemd service configured", ""))
            return results
    except Exception:
        results.append((INFO, "Could not check systemd service", ""))
        return results

    # Check linger status.
    try:
        user = os.environ.get("USER", "")
        loginctl = shutil.which("loginctl")
        if loginctl:
            result = subprocess.run(
                [loginctl, "show-user", user, "--property=Linger"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                linger_line = result.stdout.strip()
                if "yes" in linger_line:
                    results.append((OK, "Systemd linger enabled (gateway survives logout)", ""))
                else:
                    results.append((WARN,
                        "Systemd linger NOT enabled",
                        "Gateway service will stop after you log out. "
                        "Run: sudo loginctl enable-linger $USER",
                    ))
            else:
                results.append((INFO, "Could not check linger status", ""))
        else:
            results.append((INFO, "loginctl not found", ""))
    except Exception:
        results.append((INFO, "Linger check failed", ""))
    return results


# ---------------------------------------------------------------------------
# 3. Tool availability check
# ---------------------------------------------------------------------------


def check_tool_availability() -> List[CheckResult]:
    """Verify that the registered tools are importable."""
    results: List[CheckResult] = []
    try:
        from niaharness.tools import create_default_tool_registry
        registry = create_default_tool_registry()
        tools = registry.list_tools()
        if not tools:
            results.append((WARN, "No tools registered", ""))
            return results
        results.append((OK, f"{len(tools)} tools registered", ""))
        # Check a few critical tools.
        critical = ["bash", "read_file", "write_file", "web_search", "skill"]
        for name in critical:
            if registry.get(name) is not None:
                results.append((OK, f"Tool '{name}' available", ""))
            else:
                results.append((WARN, f"Tool '{name}' not found", ""))
    except Exception as exc:
        results.append((FAIL, "Tool registry check failed", str(exc)))
    return results


# ---------------------------------------------------------------------------
# 4. Skills hub check
# ---------------------------------------------------------------------------


def check_skills_hub() -> List[CheckResult]:
    """Verify skill hub directories + lock file exist."""
    results: List[CheckResult] = []
    nia_home = _get_nia_home()

    # Check skills directory.
    skills_dir = nia_home / "skills"
    if skills_dir.exists():
        skill_count = len(list(skills_dir.rglob("SKILL.md")))
        results.append((OK, f"Skills directory exists ({skill_count} skills)", ""))
    else:
        results.append((INFO, "Skills directory not created yet", "Run 'nia' to initialize"))

    # Check hub directory.
    hub_dir = nia_home / "skills" / ".hub"
    if hub_dir.exists():
        results.append((OK, "Skill hub directory exists", ""))
    else:
        results.append((INFO, "Skill hub not initialized", "Install a skill to create it"))

    # Check lock file.
    lock_file = nia_home / "skills" / ".hub" / "lock.json"
    if lock_file.exists():
        try:
            import json
            data = json.loads(lock_file.read_text())
            installed = data.get("installed", {})
            results.append((OK, f"Lock file exists ({len(installed)} installed skills)", ""))
        except Exception:
            results.append((WARN, "Lock file exists but is corrupt", ""))
    # Don't warn if lock file doesn't exist — it's created on first install.

    # Check bundled skills.
    try:
        from niaharness.skills.bundled import get_bundled_skills_dir
        bundled_dir = get_bundled_skills_dir()
        if bundled_dir.exists():
            bundled_count = len(list(bundled_dir.rglob("SKILL.md")))
            results.append((OK, f"Bundled skills: {bundled_count}", ""))
        else:
            results.append((WARN, "Bundled skills directory not found", str(bundled_dir)))
    except Exception:
        results.append((INFO, "Could not check bundled skills", ""))
    return results


# ---------------------------------------------------------------------------
# 5. Memory provider check
# ---------------------------------------------------------------------------


def check_memory_provider() -> List[CheckResult]:
    """Verify memory store + write gate are functional."""
    results: List[CheckResult] = []
    nia_home = _get_nia_home()

    # Check memory directory.
    memory_dir = nia_home / "memory"
    if memory_dir.exists():
        results.append((OK, "Memory directory exists", ""))
    else:
        results.append((INFO, "Memory directory not created yet", ""))

    # Check if MemoryStore is importable.
    try:
        from niaharness.memory.store import MemoryStore, WriteGate
        results.append((OK, "MemoryStore module importable", ""))
        # Check WriteGate stats.
        gate = WriteGate()
        results.append((OK, f"WriteGate functional (blocked={gate.blocked_count})", ""))
    except Exception as exc:
        results.append((FAIL, "MemoryStore import failed", str(exc)))

    # Check if MemoryManager is initialized.
    try:
        from niaharness.memory import get_memory_manager
        manager = get_memory_manager()
        providers = manager.providers
        if providers:
            names = [p.name for p in providers]
            results.append((OK, f"MemoryManager active ({len(providers)} providers: {', '.join(names)})", ""))
        else:
            results.append((INFO, "MemoryManager has no providers registered", ""))
    except Exception:
        results.append((INFO, "MemoryManager not initialized", ""))
    return results


# ---------------------------------------------------------------------------
# 6. Profiles check
# ---------------------------------------------------------------------------


def check_profiles() -> List[CheckResult]:
    """List profiles + verify active profile is valid."""
    results: List[CheckResult] = []
    try:
        from niaharness.profiles import list_profiles, get_active_profile_name
        profiles = list_profiles()
        if not profiles:
            results.append((INFO, "No profiles configured (using default)", ""))
            return results
        results.append((OK, f"{len(profiles)} profile(s) found", ""))
        active = get_active_profile_name()
        results.append((OK, f"Active profile: {active}", ""))
        for p in profiles:
            marker = " ← active" if p.name == active else ""
            results.append((INFO, f"  {p.name}{marker}", ""))
    except Exception as exc:
        results.append((WARN, "Profile check failed", str(exc)))
    return results


# ---------------------------------------------------------------------------
# 7. Required packages check
# ---------------------------------------------------------------------------


# (module_name, display_name, is_required)
_PACKAGES = [
    ("anthropic", "Anthropic SDK", True),
    ("openai", "OpenAI SDK", True),
    ("httpx", "HTTPX", True),
    ("pydantic", "Pydantic", True),
    ("yaml", "PyYAML", True),
    ("rich", "Rich (terminal UI)", False),
    ("dotenv", "python-dotenv", True),
    ("typer", "Typer (CLI)", True),
    ("croniter", "Croniter (cron expressions)", False),
    ("playwright", "Playwright (browser tool)", False),
    ("edge_tts", "edge-tts (TTS fallback)", False),
    ("kittentts", "kittentts (neural TTS)", False),
]


def check_required_packages() -> List[CheckResult]:
    """Verify all required Python packages are installed."""
    results: List[CheckResult] = []
    for module, name, required in _PACKAGES:
        try:
            importlib.import_module(module)
            label = "" if required else " (optional)"
            results.append((OK, f"{name}{label}", ""))
        except ImportError:
            if required:
                results.append((FAIL, f"{name} (missing)", f"pip install {module}"))
            else:
                results.append((WARN, f"{name} (optional, not installed)", ""))
    return results


# ---------------------------------------------------------------------------
# 8. Command installation check
# ---------------------------------------------------------------------------


def check_command_installation() -> List[CheckResult]:
    """Verify `nia` command is on PATH."""
    results: List[CheckResult] = []
    nia_path = shutil.which("nia")
    if nia_path:
        results.append((OK, f"`nia` command on PATH ({nia_path})", ""))
    else:
        results.append((WARN,
            "`nia` command not on PATH",
            "Run: pip install -e . (or add ~/.local/bin to PATH)",
        ))

    # Also check niaharness.
    niaharness_path = shutil.which("niaharness")
    if niaharness_path:
        results.append((OK, f"`niaharness` command on PATH", ""))
    else:
        results.append((INFO, "`niaharness` command not on PATH (optional)", ""))
    return results


# ---------------------------------------------------------------------------
# 9. Config structure validation
# ---------------------------------------------------------------------------


_REQUIRED_CONFIG_SECTIONS = {"model", "permissions"}
_RECOMMENDED_CONFIG_SECTIONS = {"memory", "cron", "gateway"}


def check_config_structure() -> List[CheckResult]:
    """Verify config.yaml has required + recommended sections."""
    results: List[CheckResult] = []
    nia_home = _get_nia_home()
    config_path = nia_home / "config.yaml"

    if not config_path.exists():
        results.append((INFO, "No config.yaml found", "Run 'nia setup' to create one"))
        return results

    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        results.append((FAIL, "config.yaml is invalid YAML", str(exc)))
        return results

    if not isinstance(cfg, dict):
        results.append((FAIL, "config.yaml root is not a dict", ""))
        return results

    # Check required sections.
    for section in _REQUIRED_CONFIG_SECTIONS:
        if section in cfg:
            results.append((OK, f"Config section '{section}' present", ""))
        else:
            results.append((WARN, f"Config section '{section}' missing", ""))

    # Check recommended sections.
    for section in _RECOMMENDED_CONFIG_SECTIONS:
        if section in cfg:
            results.append((OK, f"Config section '{section}' present", ""))
        else:
            results.append((INFO, f"Config section '{section}' not set (optional)", ""))

    # Check model section has required keys.
    model = cfg.get("model", {})
    if isinstance(model, dict):
        if model.get("provider"):
            results.append((OK, f"Model provider: {model['provider']}", ""))
        else:
            results.append((WARN, "model.provider not set", ""))
        if model.get("default"):
            results.append((OK, f"Model: {model['default']}", ""))
        else:
            results.append((WARN, "model.default not set", ""))
    return results


# ---------------------------------------------------------------------------
# 10. xAI model retirement warning
# ---------------------------------------------------------------------------


_RETIRED_XAI_MODELS = frozenset({
    "grok-1",
    "grok-1-preview",
    "grok-1-heavy",
    "grok-2",
    "grok-2-mini",
    "grok-2-vision",
    "grok-2-vision-1212",
    "grok-beta",
    "grok-vision-beta",
})
_CURRENT_XAI_MODELS = frozenset({
    "grok-3",
    "grok-3-mini",
    "grok-3-fast",
    "grok-3-mini-fast",
    "grok-4",
    "grok-4-fast",
})


def check_xai_model_retirement() -> List[CheckResult]:
    """Warn if using retired xAI models."""
    results: List[CheckResult] = []
    nia_home = _get_nia_home()
    config_path = nia_home / "config.yaml"
    if not config_path.exists():
        return results

    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return results

    model = cfg.get("model", {})
    if not isinstance(model, dict):
        return results

    provider = str(model.get("provider", "")).lower()
    default_model = str(model.get("default", "") or model.get("model", ""))

    # Check env vars too.
    env_model = os.environ.get("NIA_MODEL", "").lower()
    env_provider = os.environ.get("NIA_PROVIDER", "").lower()

    models_to_check = [default_model.lower(), env_model]
    providers_to_check = [provider, env_provider]

    # Only warn if xAI is the provider.
    if "xai" not in providers_to_check and "grok" not in " ".join(models_to_check):
        return results

    for model_name in models_to_check:
        if not model_name:
            continue
        if model_name in _RETIRED_XAI_MODELS:
            results.append((WARN,
                f"xAI model '{model_name}' is retired",
                f"Switch to a current model: {', '.join(sorted(_CURRENT_XAI_MODELS))}",
            ))
        elif model_name in _CURRENT_XAI_MODELS:
            results.append((OK, f"xAI model '{model_name}' is current", ""))
    return results


# ---------------------------------------------------------------------------
# Runner — run all extension checks
# ---------------------------------------------------------------------------


def run_extension_checks() -> List[CheckResult]:
    """Run all 10 extension checks and return the combined results."""
    results: List[CheckResult] = []
    for check_fn in [
        check_version_consistency,
        check_gateway_service_linger,
        check_tool_availability,
        check_skills_hub,
        check_memory_provider,
        check_profiles,
        check_required_packages,
        check_command_installation,
        check_config_structure,
        check_xai_model_retirement,
    ]:
        try:
            results.extend(check_fn())
        except Exception as exc:
            results.append((FAIL, f"{check_fn.__name__} crashed", str(exc)))
    return results


# Section metadata for the doctor renderer.
EXTENSION_SECTIONS = [
    ("Version Consistency", check_version_consistency),
    ("Gateway Service Linger", check_gateway_service_linger),
    ("Tool Availability", check_tool_availability),
    ("Skills Hub", check_skills_hub),
    ("Memory Provider", check_memory_provider),
    ("Profiles", check_profiles),
    ("Required Packages", check_required_packages),
    ("Command Installation", check_command_installation),
    ("Config Structure", check_config_structure),
    ("xAI Model Retirement", check_xai_model_retirement),
]


__all__ = [
    "CheckResult",
    "EXTENSION_SECTIONS",
    "FAIL",
    "INFO",
    "OK",
    "WARN",
    "check_command_installation",
    "check_config_structure",
    "check_gateway_service_linger",
    "check_memory_provider",
    "check_profiles",
    "check_required_packages",
    "check_skills_hub",
    "check_tool_availability",
    "check_version_consistency",
    "check_xai_model_retirement",
    "run_extension_checks",
]
