"""NIA Update — detect install method, backup, upgrade, restart.

Ported from Hermes Agent's ``hermes_cli/subcommands/update.py`` +
``hermes_cli/main.py:cmd_update``, scoped to NIA's architecture.

Detects the install method (uv-tool / pipx / venv-pip / editable-source /
docker), creates a pre-update backup of ``~/.nia``, executes the
appropriate upgrade command, runs config migration, and restarts by
re-execing ``sys.executable``.

Usage::

    from niaharness.cli.update import run_update

    # Check for update availability.
    result = run_update(check=True)
    print(result.report)

    # Execute update.
    result = run_update()
    if result.success:
        print("Update complete — restarting...")
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class UpdateResult:
    """Result of running the update check or execution."""
    success: bool = False
    current_version: str = ""
    latest_version: str = ""
    install_method: str = ""
    update_available: bool = False
    backup_path: Optional[str] = None
    report: str = ""
    errors: List[str] = field(default_factory=list)
    needs_restart: bool = False


# ---------------------------------------------------------------------------
# Install method detection
# ---------------------------------------------------------------------------


def detect_install_method() -> str:
    """Detect how NIA was installed.

    Returns one of: ``"uv-tool"``, ``"pipx"``, ``"venv-pip"``,
    ``"editable"``, ``"docker"``, ``"pip"``.

    Resolution order:
      1. ``/uv/tools/`` in ``sys.prefix`` → ``"uv-tool"``
      2. ``pipx`` in ``sys.prefix`` path → ``"pipx"``
      3. ``.git`` directory in the project root → ``"editable"``
      4. Docker container detected (``/.dockerenv`` or ``NIA_DOCKER`` env) → ``"docker"``
      5. Venv active (``sys.prefix != sys.base_prefix``) → ``"venv-pip"``
      6. Fallback → ``"pip"``
    """
    # 1. uv-tool.
    if "/uv/tools/" in sys.prefix or "/uv/tools/" in sys.executable:
        return "uv-tool"

    # 2. pipx.
    if "pipx" in sys.prefix.split(os.sep):
        return "pipx"

    # 3. Editable source (git checkout).
    try:
        # Look for pyproject.toml with a [tool.setuptools] or [project] section
        # + a .git dir in the parent.
        import niaharness
        pkg_dir = Path(niaharness.__file__).resolve().parent
        # Walk up to find pyproject.toml + .git.
        for parent in [pkg_dir, *pkg_dir.parents]:
            if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
                return "editable"
    except Exception:
        pass

    # 4. Docker.
    if Path("/.dockerenv").exists() or os.environ.get("NIA_DOCKER"):
        return "docker"

    # 5. Venv-pip.
    if sys.prefix != sys.base_prefix:
        return "venv-pip"

    # 6. Fallback.
    return "pip"


def get_current_version() -> str:
    """Get the currently installed NIA version."""
    try:
        return importlib.metadata.version("niaharness")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"
    except Exception:
        return "unknown"


def get_install_path() -> Optional[Path]:
    """Return the project root path for editable installs, or None."""
    try:
        import niaharness
        pkg_dir = Path(niaharness.__file__).resolve().parent
        for parent in [pkg_dir, *pkg_dir.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------


def check_for_update() -> tuple[str, str, bool]:
    """Check if an update is available.

    Returns ``(current_version, latest_version, update_available)``.

    Uses ``pip index versions`` or ``uv tool upgrade --check`` depending
    on the install method. Falls back to PyPI JSON API.
    """
    current = get_current_version()
    method = detect_install_method()

    # Try pip/uv to check for the latest version.
    latest = ""
    try:
        if method == "uv-tool":
            result = subprocess.run(
                ["uv", "tool", "list", "--outdated"],
                capture_output=True, text=True, timeout=15,
            )
            # Parse output for niaharness.
            for line in result.stdout.split("\n"):
                if "niaharness" in line.lower():
                    # uv tool list --outdated shows: niaharness  v1.0.0 → v1.1.0
                    parts = line.split("→")
                    if len(parts) >= 2:
                        latest = parts[-1].strip().split()[0].lstrip("v")
                    break
        elif method == "pipx":
            result = subprocess.run(
                ["pipx", "list", "--short"],
                capture_output=True, text=True, timeout=15,
            )
            for line in result.stdout.split("\n"):
                if "niaharness" in line.lower():
                    # pipx list --short: niaharness 1.0.0
                    parts = line.split()
                    if len(parts) >= 2:
                        latest = parts[1]
                    break

        if not latest:
            # Fallback: PyPI JSON API.
            import json
            import urllib.request
            url = "https://pypi.org/pypi/niaharness/json"
            req = urllib.request.Request(url, headers={"User-Agent": "nia-update-check"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                latest = data.get("info", {}).get("version", "")
    except Exception as exc:
        logger.debug("Version check failed: %s", exc)
        return current, "", False

    update_available = bool(latest) and latest != current and latest != "unknown"
    return current, latest, update_available


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def create_pre_update_backup() -> Optional[str]:
    """Create a ZIP backup of ~/.nia before updating.

    Best-effort — never blocks the update. Returns the backup path or None.
    """
    try:
        import zipfile

        from niaharness.prompts.soul import get_nia_home

        nia_home = get_nia_home()
        if not nia_home.exists():
            return None

        backup_dir = nia_home / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        backup_path = backup_dir / f"pre-update-{timestamp}.zip"

        with zipfile.ZipFile(str(backup_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(str(nia_home)):
                # Skip the backups dir itself + large binary files.
                if "backups" in dirs:
                    dirs.remove("backups")
                if "sessions.db-wal" in files:
                    files.remove("sessions.db-wal")
                if "sessions.db-shm" in files:
                    files.remove("sessions.db-shm")
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(nia_home)
                    try:
                        zf.write(str(file_path), str(arcname))
                    except (OSError, PermissionError):
                        pass  # Best-effort.

        logger.info("Pre-update backup created: %s", backup_path)
        return str(backup_path)
    except Exception as exc:
        logger.warning("Pre-update backup failed: %s", exc)
        return None


def prune_old_backups(keep: int = 5) -> None:
    """Keep only the N most recent pre-update backups."""
    try:
        from niaharness.prompts.soul import get_nia_home

        backup_dir = get_nia_home() / "backups"
        if not backup_dir.exists():
            return

        backups = sorted(
            backup_dir.glob("pre-update-*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_backup in backups[keep:]:
            try:
                old_backup.unlink()
            except OSError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Execute update
# ---------------------------------------------------------------------------


def _run_upgrade_command(method: str) -> tuple[bool, str]:
    """Execute the appropriate upgrade command for the install method.

    Returns ``(success, output)``.
    """
    try:
        if method == "uv-tool":
            result = subprocess.run(
                ["uv", "tool", "upgrade", "niaharness"],
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0, result.stdout + result.stderr

        elif method == "pipx":
            result = subprocess.run(
                ["pipx", "upgrade", "niaharness"],
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0, result.stdout + result.stderr

        elif method == "editable":
            project_path = get_install_path()
            if project_path is None:
                return False, "Could not find project path for editable install"
            # git pull + uv sync.
            git_result = subprocess.run(
                ["git", "pull"],
                cwd=str(project_path),
                capture_output=True, text=True, timeout=60,
            )
            if git_result.returncode != 0:
                return False, f"git pull failed: {git_result.stderr}"
            sync_result = subprocess.run(
                ["uv", "sync"],
                cwd=str(project_path),
                capture_output=True, text=True, timeout=120,
            )
            return sync_result.returncode == 0, git_result.stdout + sync_result.stdout + sync_result.stderr

        elif method == "docker":
            # Docker: can't self-update; user needs to pull the new image.
            return False, "Docker installs must be updated by pulling the new image. Run: docker pull <image>"

        else:
            # venv-pip or pip.
            pip_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "niaharness"]
            result = subprocess.run(
                pip_cmd,
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0, result.stdout + result.stderr

    except subprocess.TimeoutExpired:
        return False, "Upgrade command timed out"
    except FileNotFoundError as exc:
        return False, f"Command not found: {exc}"
    except Exception as exc:
        return False, str(exc)


def run_update(
    *,
    check: bool = False,
    no_backup: bool = False,
    force_backup: bool = False,
) -> UpdateResult:
    """Run the NIA update process.

    Args:
        check: If True, only check for update availability (don't install).
        no_backup: If True, skip the pre-update backup.
        force_backup: If True, force a backup even if disabled by config.

    Returns:
        :class:`UpdateResult` with the outcome.
    """
    result = UpdateResult()
    result.current_version = get_current_version()
    result.install_method = detect_install_method()

    lines: List[str] = []
    lines.append("")
    lines.append("┌─────────────────────────────────────────────────────────┐")
    lines.append("│                   🔄 NIA Update                          │")
    lines.append("└─────────────────────────────────────────────────────────┘")
    lines.append(f"  Current version: {result.current_version}")
    lines.append(f"  Install method:  {result.install_method}")

    # Check for update.
    current, latest, update_available = check_for_update()
    result.latest_version = latest
    result.update_available = update_available

    if not latest:
        lines.append("  Latest version:  (could not determine)")
        lines.append("  ⚠ Could not check for updates — network error or package not on PyPI.")
        result.report = "\n".join(lines)
        return result

    lines.append(f"  Latest version:  {latest}")

    if check:
        if update_available:
            lines.append(f"\n  ✅ Update available: {current} → {latest}")
            lines.append(f"  Run 'nia update' to install.")
        else:
            lines.append("\n  ✅ Already up to date.")
        result.success = True
        result.report = "\n".join(lines)
        return result

    if not update_available:
        lines.append("\n  ✅ Already up to date.")
        result.success = True
        result.report = "\n".join(lines)
        return result

    lines.append(f"\n  Updating from {current} to {latest}...")

    # Create backup (unless --no-backup).
    if not no_backup or force_backup:
        backup_path = create_pre_update_backup()
        if backup_path:
            result.backup_path = backup_path
            lines.append(f"  ✓ Backup created: {backup_path}")
            prune_old_backups()
        else:
            lines.append("  ⚠ Backup failed — continuing with update.")

    # Execute the upgrade.
    success, output = _run_upgrade_command(result.install_method)
    if not success:
        lines.append(f"  ✗ Update failed: {output}")
        result.errors.append(output)
        result.report = "\n".join(lines)
        return result

    lines.append("  ✓ Upgrade complete.")

    # Run config migration (best-effort).
    try:
        from niaharness.cli.doctor import run_doctor
        doctor_result = run_doctor(fix=True)
        if doctor_result.fixed_count > 0:
            lines.append(f"  ✓ Config migration: fixed {doctor_result.fixed_count} issue(s)")
    except Exception:
        pass  # Best-effort.

    # Verify the new version.
    new_version = get_current_version()
    if new_version != result.current_version:
        lines.append(f"  ✓ Version updated: {result.current_version} → {new_version}")
        result.needs_restart = True
    else:
        lines.append("  ⚠ Version unchanged — you may need to restart manually.")

    lines.append("\n  ✅ Update complete!")
    lines.append("  Tip: Restart NIA to use the new version.")

    result.success = True
    result.report = "\n".join(lines)
    return result


def restart_process() -> None:
    """Re-exec the current process with the same arguments.

    Uses ``os.execvpe`` to replace the current process. The caller should
    call this only after all cleanup is done — it never returns.
    """
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


__all__ = [
    "UpdateResult",
    "check_for_update",
    "create_pre_update_backup",
    "detect_install_method",
    "get_current_version",
    "get_install_path",
    "prune_old_backups",
    "restart_process",
    "run_update",
]
