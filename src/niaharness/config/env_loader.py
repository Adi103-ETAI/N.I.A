"""Environment file loader — loads ~/.nia/.env at startup.

Reads the user's ``~/.nia/.env`` file (or ``$NIA_HOME/.env``) and injects
its values into ``os.environ`` with ``override=True`` so the file takes
precedence over stale shell-exported values. This is the single source
of truth for API keys and secrets — ``nia setup`` writes here, and every
NIA entry point loads it before doing anything else.

Called from:
  - ``niaharness.cli:main()`` (the ``nia`` command)
  - ``niaharness.__main__`` (``python -m niaharness``)

Behavior:
  1. Resolves the env path via ``get_nia_home() / ".env"``
  2. Pre-sanitizes corrupted files (strips null bytes, fixes bare ``=`` lines)
  3. Loads with ``override=True`` (user env beats stale shell vars)
  4. Falls back to UTF-8-sig → latin-1 encoding if needed
  5. Also loads ``~/.nia/.env.local`` if present (override=False, for dev overrides)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Regex for valid env var names.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$", re.IGNORECASE)


def get_env_path() -> Path:
    """Return the path to ~/.nia/.env."""
    try:
        from niaharness.prompts.soul import get_nia_home
        return get_nia_home() / ".env"
    except Exception:
        return Path(os.path.expanduser("~/.nia/.env"))


def _sanitize_env_file(path: Path) -> None:
    """Fix common .env corruption issues in-place.

    - Strips null bytes that can appear from interrupted writes
    - Removes lines that are just ``=`` with no key
    - Ensures the file ends with a newline
    """
    try:
        content = path.read_bytes()
        if b"\x00" in content:
            content = content.replace(b"\x00", b"")
            path.write_bytes(content)
            logger.debug("Stripped null bytes from %s", path)
    except Exception:
        pass

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()
        fixed: List[str] = []
        for line in lines:
            stripped = line.strip()
            # Skip bare = lines (corrupted entries).
            if stripped == "=" or (stripped.startswith("=") and not stripped[1:2].strip()):
                continue
            fixed.append(line)
        if not fixed:
            return
        if fixed[-1] != "":
            fixed.append("")
        new_text = "\n".join(fixed)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            logger.debug("Sanitized %s (%d → %d lines)", path, len(lines), len(fixed))
    except Exception as exc:
        logger.debug("Could not sanitize %s: %s", path, exc)


def _load_dotenv_safe(path: Path, *, override: bool) -> bool:
    """Load a .env file with encoding fallback. Returns True if loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Fallback: manual parser if python-dotenv isn't installed.
        return _load_dotenv_manual(path, override=override)

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            load_dotenv(dotenv_path=path, override=override, encoding=encoding)
            return True
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.debug("load_dotenv failed for %s with %s: %s", path, encoding, exc)
            continue
    return False


def _load_dotenv_manual(path: Path, *, override: bool) -> bool:
    """Manual .env parser (fallback when python-dotenv isn't installed)."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return False

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match KEY=VALUE or KEY="VALUE" or KEY='VALUE'.
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not _ENV_VAR_NAME_RE.match(key):
            continue
        value = value.strip()
        # Strip surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Unescape common sequences.
        value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def load_nia_env(
    *,
    nia_home: Optional[Path] = None,
    project_env: Optional[Path] = None,
) -> List[Path]:
    """Load NIA environment files.

    Loads in order (later files can fill gaps but not override user env):
      1. ``~/.nia/.env`` — user env (override=True, beats stale shell vars)
      2. ``~/.nia/.env.local`` — dev overrides (override=False, fills gaps)
      3. *project_env* — project-level .env (override=False, fills gaps)

    Args:
        nia_home: Override the NIA home directory. Defaults to ``get_nia_home()``.
        project_env: Optional project-level .env path.

    Returns:
        List of paths that were loaded.
    """
    loaded: List[Path] = []

    if nia_home is None:
        try:
            from niaharness.prompts.soul import get_nia_home
            nia_home = get_nia_home()
        except Exception:
            nia_home = Path(os.path.expanduser("~/.nia"))

    user_env = nia_home / ".env"

    # Sanitize before loading.
    if user_env.exists():
        _sanitize_env_file(user_env)

    # 1. User env — override=True (beats stale shell vars).
    if user_env.exists():
        if _load_dotenv_safe(user_env, override=True):
            loaded.append(user_env)
            logger.debug("Loaded user env: %s", user_env)

    # 2. .env.local — dev overrides (fills gaps only).
    local_env = nia_home / ".env.local"
    if local_env.exists():
        if _load_dotenv_safe(local_env, override=False):
            loaded.append(local_env)
            logger.debug("Loaded local env: %s", local_env)

    # 3. Project .env — fills gaps.
    if project_env and project_env.exists():
        if _load_dotenv_safe(project_env, override=not loaded):
            loaded.append(project_env)
            logger.debug("Loaded project env: %s", project_env)

    return loaded


def get_env_value(key: str) -> Optional[str]:
    """Read a value from the environment (after load_nia_env has been called)."""
    return os.environ.get(key) or None


def save_env_value(key: str, value: str) -> None:
    """Save or update a value in ~/.nia/.env.

    Also sets it in the current process's ``os.environ`` immediately.
    """
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid env var name: {key!r}")

    value = value.replace("\n", "").replace("\r", "")

    env_path = get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing lines.
    lines: List[str] = []
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()

    # Find and update or append.
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(str(env_path), 0o600)
    except OSError:
        pass

    # Set in current process immediately.
    os.environ[key] = value
    logger.debug("Saved %s to %s", key, env_path)


def remove_env_value(key: str) -> bool:
    """Remove a key from ~/.nia/.env. Returns True if it was present."""
    env_path = get_env_path()
    if not env_path.exists():
        return False

    text = env_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    new_lines = [l for l in lines if not l.strip().startswith(f"{key}=")]

    if len(new_lines) == len(lines):
        return False

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ.pop(key, None)
    return True


__all__ = [
    "get_env_path",
    "get_env_value",
    "load_nia_env",
    "remove_env_value",
    "save_env_value",
]
