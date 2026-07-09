"""Profile aliases — wrapper scripts + collision detection + ProfileInfo.

Ported from Hermes Agent's ``hermes_cli/profiles.py`` (2,225 LOC),
scoped to the wrapper-script + alias subsystem. Provides:

  - :func:`create_wrapper_script` — installs ``~/.local/bin/<name>`` shell
    wrapper so ``coder`` ↔ ``nia -p coder``.
  - :func:`remove_wrapper_script` — removes the wrapper (verifies it's ours).
  - :func:`check_alias_collision` — checks for reserved names, existing
    commands, and NIA subcommands.
  - :func:`validate_alias_name` — regex guard against path traversal.
  - :class:`ProfileInfo` — summary dataclass for profile listing.

Wrapper format (POSIX)::

    #!/bin/sh
    exec /path/to/nia -p <profile> "$@"

Wrapper format (Windows)::

    @echo off
    nia -p <profile> %*
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for valid alias/profile names: [a-z0-9][a-z0-9_-]{0,63}
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Files copied during --clone (at profile root).
_CLONE_CONFIG_FILES = ["config.yaml", ".env", "SOUL.md"]

# Subdirectory files copied during --clone.
_CLONE_SUBDIR_FILES = ["memories/MEMORY.md", "memories/USER.md"]

# Reserved names that cannot be used as aliases.
_RESERVED_NAMES = frozenset({"nia", "default", "test", "tmp", "root", "sudo"})

# NIA subcommands that cannot be reused as aliases.
_NIA_SUBCOMMANDS = frozenset({
    "chat", "model", "gateway", "setup", "login", "logout", "status",
    "cron", "doctor", "config", "skills", "tools", "mcp", "sessions",
    "insights", "version", "update", "profile", "memory",
})

# The marker string that identifies a wrapper as ours.
_WRAPPER_MARKER = "nia -p"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_wrapper_dir() -> Path:
    """Return the wrapper script directory (~/.local/bin)."""
    return Path.home() / ".local" / "bin"


def _is_wrapper_dir_in_path() -> bool:
    """Check if ~/.local/bin is on PATH."""
    wrapper_dir = str(_get_wrapper_dir())
    return wrapper_dir in os.environ.get("PATH", "").split(os.pathsep)


# ---------------------------------------------------------------------------
# Alias validation
# ---------------------------------------------------------------------------


def validate_alias_name(name: str) -> None:
    """Raise ValueError if *name* is not a safe wrapper-alias identifier.

    The alias is used verbatim as a filename under :func:`_get_wrapper_dir`,
    so it must be a single safe command name with no path separators or
    traversal segments.
    """
    if not _PROFILE_ID_RE.match(name):
        raise ValueError(
            f"Invalid alias name {name!r}. Must match [a-z0-9][a-z0-9_-]{{0,63}}"
        )


def check_alias_collision(name: str) -> Optional[str]:
    """Return a human-readable collision message, or None if the name is safe.

    Checks: alias-name validity, reserved names, NIA subcommands, existing
    binaries in PATH.
    """
    try:
        validate_alias_name(name)
    except ValueError as exc:
        return str(exc)

    if name in _RESERVED_NAMES:
        return f"'{name}' is a reserved name"
    if name in _NIA_SUBCOMMANDS:
        return f"'{name}' conflicts with a NIA subcommand"

    # Check existing commands in PATH.
    wrapper_dir = _get_wrapper_dir()
    is_windows = sys.platform == "win32"
    try:
        result = subprocess.run(
            ["where" if is_windows else "which", name],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            existing_path = result.stdout.strip().splitlines()[0]
            # Allow overwriting our own wrappers.
            expected = wrapper_dir / (f"{name}.bat" if is_windows else name)
            if existing_path == str(expected):
                try:
                    content = expected.read_text()
                    if _WRAPPER_MARKER in content:
                        return None  # Our wrapper — safe to overwrite.
                except Exception:
                    pass
            return f"'{name}' conflicts with an existing command ({existing_path})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None  # Safe.


# ---------------------------------------------------------------------------
# Wrapper script management
# ---------------------------------------------------------------------------


def create_wrapper_script(name: str, target: Optional[str] = None) -> Optional[Path]:
    """Create a shell wrapper script at ~/.local/bin/<name>.

    The wrapper invokes ``nia -p <target or name>`` so typing the alias
    name launches NIA under the target profile.

    Args:
        name: The alias name (used as the wrapper filename).
        target: The profile to activate. Defaults to *name*.

    Returns:
        Path to the created wrapper, or None on failure.
    """
    validate_alias_name(name)
    profile = target or name
    wrapper_dir = _get_wrapper_dir()
    try:
        wrapper_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create %s: %s", wrapper_dir, exc)
        return None

    is_windows = sys.platform == "win32"
    if is_windows:
        wrapper_path = wrapper_dir / f"{name}.bat"
        try:
            wrapper_path.write_text(f"@echo off\r\nnia -p {profile} %*\r\n")
            return wrapper_path
        except OSError as exc:
            logger.warning("Could not create wrapper at %s: %s", wrapper_path, exc)
            return None
    else:
        wrapper_path = wrapper_dir / name
        try:
            nia_exe = shutil.which("nia") or "nia"
            wrapper_path.write_text(
                f'#!/bin/sh\nexec {shlex.quote(nia_exe)} -p {profile} "$@"\n'
            )
            wrapper_path.chmod(
                wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
            )
            return wrapper_path
        except OSError as exc:
            logger.warning("Could not create wrapper at %s: %s", wrapper_path, exc)
            return None


def remove_wrapper_script(name: str) -> bool:
    """Remove the wrapper script for a profile. Returns True if removed."""
    wrapper_dir = _get_wrapper_dir()
    try:
        validate_alias_name(name)
    except ValueError:
        return False

    is_windows = sys.platform == "win32"
    candidates = [wrapper_dir / name]
    if is_windows:
        candidates.insert(0, wrapper_dir / f"{name}.bat")

    for wrapper_path in candidates:
        if wrapper_path.exists():
            try:
                content = wrapper_path.read_text()
                if _WRAPPER_MARKER in content:
                    wrapper_path.unlink()
                    return True
            except Exception:
                pass
    return False


def find_alias_for_profile(profile_name: str) -> Optional[str]:
    """Find the alias name that points at *profile_name*.

    Scans wrapper scripts in ~/.local/bin/ for ``nia -p <profile_name>``.
    Returns the alias name, or None if no wrapper points at this profile.
    """
    wrapper_dir = _get_wrapper_dir()
    if not wrapper_dir.exists():
        return None

    is_windows = sys.platform == "win32"
    for wrapper_path in wrapper_dir.iterdir():
        if not wrapper_path.is_file():
            continue
        try:
            content = wrapper_path.read_text(errors="replace")
            if _WRAPPER_MARKER not in content:
                continue
            # Check if this wrapper points at our profile.
            if f"nia -p {profile_name}" in content or f"nia -p {profile_name} " in content:
                return wrapper_path.stem if is_windows else wrapper_path.name
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Clone helpers
# ---------------------------------------------------------------------------


def clone_profile_files(
    source_dir: Path,
    target_dir: Path,
    *,
    clone_all: bool = False,
) -> int:
    """Copy files from source profile to target profile.

    Args:
        source_dir: Source profile directory.
        target_dir: Target profile directory.
        clone_all: If True, copy the entire directory (with exclusions).
            If False, copy only config files + memory files.

    Returns:
        Number of files copied.
    """
    import shutil

    count = 0

    if clone_all:
        # Full copy with exclusions.
        exclude = {"__pycache__", "*.pyc", "sessions.db", "sessions.db-wal",
                    "sessions.db-shm", "gateway.pid", "backups"}
        exclude_dirs = {"sessions", "backups", "logs", "__pycache__"}

        def ignore_fn(directory: str, names: list[str]) -> list[str]:
            ignored = []
            for name in names:
                if name in exclude_dirs:
                    ignored.append(name)
                for pat in exclude:
                    if pat in name:
                        ignored.append(name)
                        break
            return ignored

        try:
            shutil.copytree(str(source_dir), str(target_dir), ignore=ignore_fn, dirs_exist_ok=True)
            # Count files (approximate).
            for _ in target_dir.rglob("*"):
                if _.is_file():
                    count += 1
        except Exception as exc:
            logger.warning("Clone-all failed: %s", exc)
    else:
        # Light clone: config files + memory files.
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in _CLONE_CONFIG_FILES:
            src = source_dir / filename
            if src.exists():
                try:
                    shutil.copy2(str(src), str(target_dir / filename))
                    count += 1
                except Exception as exc:
                    logger.debug("Could not copy %s: %s", filename, exc)
        for subdir_file in _CLONE_SUBDIR_FILES:
            src = source_dir / subdir_file
            if src.exists():
                dst = target_dir / subdir_file
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src), str(dst))
                    count += 1
                except Exception as exc:
                    logger.debug("Could not copy %s: %s", subdir_file, exc)

        # Set .env permissions.
        env_path = target_dir / ".env"
        if env_path.exists():
            try:
                os.chmod(str(env_path), 0o600)
            except OSError:
                pass

    return count


# ---------------------------------------------------------------------------
# ProfileInfo
# ---------------------------------------------------------------------------


@dataclass
class ProfileInfo:
    """Summary information about a profile."""
    name: str
    path: Path
    is_default: bool
    gateway_running: bool = False
    model: Optional[str] = None
    provider: Optional[str] = None
    has_env: bool = False
    skill_count: int = 0
    alias_path: Optional[Path] = None
    alias_name: Optional[str] = None
    description: str = ""


def get_profile_info(name: str, path: Path) -> ProfileInfo:
    """Build a ProfileInfo for a profile."""
    is_default = name == "default"
    has_env = (path / ".env").exists()

    # Count skills.
    skill_count = 0
    skills_dir = path / "skills"
    if skills_dir.exists():
        skill_count = sum(1 for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    # Read model/provider from config.
    model = None
    provider = None
    try:
        import json
        config_path = path / "settings.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("model")
            provider = config.get("base_url") or None
    except Exception:
        pass

    # Find alias.
    alias_name = find_alias_for_profile(name)
    alias_path = None
    if alias_name:
        alias_path = _get_wrapper_dir() / alias_name

    return ProfileInfo(
        name=name,
        path=path,
        is_default=is_default,
        gateway_running=False,  # Caller can set this.
        model=model,
        provider=provider,
        has_env=has_env,
        skill_count=skill_count,
        alias_path=alias_path,
        alias_name=alias_name,
    )


__all__ = [
    "ProfileInfo",
    "check_alias_collision",
    "clone_profile_files",
    "create_wrapper_script",
    "find_alias_for_profile",
    "get_profile_info",
    "remove_wrapper_script",
    "validate_alias_name",
]
