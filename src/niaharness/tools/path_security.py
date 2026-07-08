"""Skill install path security — rmtree-escape defense in depth.

Ported from the reference project's tools/skills_hub.py:
  - ``_normalize_lock_install_path`` — validate lock-file install_path strings
  - ``_resolve_lock_install_path`` — resolve a lock-file path without allowing
    escapes from ``SKILLS_DIR`` (symlink/junction walk + final containment check)
  - ``_is_path_redirect`` — detect symlinks and Windows junctions

Why this matters
----------------
Lock-file ``install_path`` entries are the source-of-truth for where
``uninstall_skill`` will call ``shutil.rmtree``. A poisoned or buggy entry —
empty string, ``"."``, an absolute path, ``../..`` traversal, or anything whose
final component doesn't match the skill name — would let ``rmtree`` wipe either
the entire ``skills/`` tree or content outside it.

Three layers of defence
-----------------------
1. **Shape validation** (``_normalize_lock_install_path``)
   Rejects empty/``"."``/``/abs``/``../..`` paths and enforces that the final
   component matches the skill name. Nested official optional skills may
   legitimately install below paths such as ``mlops/training/<skill_name>``;
   traversal, absolute paths, empty paths, and mismatched final components are
   still rejected.

2. **Component walk** (``_resolve_lock_install_path``)
   Walks the path component-by-component and refuses if any intermediate
   component is a symlink/junction. A path resolution that follows a symlink
   to outside skills/ would otherwise be hidden by ``Path.resolve()``.

3. **Final containment check** (``_resolve_lock_install_path``)
   After ``resolve()``, reject not just escape-out but also
   ``resolved == SKILLS_DIR`` — an empty/``"."``/``""`` install_path resolves
   to the skills root itself, and ``rmtree(SKILLS_DIR)`` would wipe every
   installed skill.

The check is used at **write** time (``HubLockFile.record_install``) and at
**read** time (``uninstall_skill``), so a poisoned lock file written by a
previous version or a hand-edit cannot escalate to a destructive rmtree.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath
from typing import Final

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Skill name pattern: lowercase letters, digits, hyphens, dots, underscores.
# Must start with a letter or digit. Max 64 chars.
_SKILL_NAME_RE: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
MAX_SKILL_NAME_LENGTH: Final = 64

# Allowed trust levels in the audit log.
_VALID_TRUST_LEVELS: Final = frozenset({"builtin", "trusted", "community"})


# ---------------------------------------------------------------------------
# Path redirect detection
# ---------------------------------------------------------------------------


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (on Windows) a directory junction.

    Either form lets an attacker who can write into the ``skills/`` tree
    redirect a subsequent ``rmtree`` to content outside it. ``is_junction``
    only exists on Python 3.12+ Windows; gate with ``hasattr``.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        # Broken symlink or permission error — treat conservatively as a redirect.
        return True
    if hasattr(path, "is_junction"):
        try:
            if path.is_junction():
                return True
        except OSError:
            return True
    return False


# ---------------------------------------------------------------------------
# Skill name + bundle path validation
# ---------------------------------------------------------------------------


def _validate_skill_name(name: str) -> str:
    """Validate a skill name. Returns the validated name or raises ValueError.

    A skill name is the load-bearing identifier used in lock-file entries,
    directory names, and the rmtree-final-component check. It must:

    - Be a non-empty string
    - Start with an ASCII letter or digit
    - Contain only ASCII letters, digits, hyphens, dots, underscores
    - Be at most 64 characters long
    - Not contain path separators or ``..``
    """
    if not isinstance(name, str):
        raise ValueError(f"Unsafe skill name (not a string): {name!r}")
    raw = name.strip()
    if not raw:
        raise ValueError("Empty skill name")
    if len(raw) > MAX_SKILL_NAME_LENGTH:
        raise ValueError(
            f"Skill name exceeds {MAX_SKILL_NAME_LENGTH} characters: {name!r}"
        )
    if not _SKILL_NAME_RE.match(raw):
        raise ValueError(f"Unsafe skill name: {name!r}")
    # Belt-and-suspenders: the regex already excludes '/' and '..', but
    # future regex tweaks could regress this. Keep the explicit check.
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"Skill name contains forbidden path characters: {name!r}")
    return raw


def _normalize_bundle_path(
    path: str,
    *,
    field_name: str,
    allow_nested: bool,
) -> str:
    """Normalize and validate a relative bundle/install path.

    Parameters
    ----------
    path : str
        The path to validate.
    field_name : str
        Used in error messages (e.g. "skill name", "install path").
    allow_nested : bool
        If True, multi-component paths like ``mlops/training/<skill>`` are
        allowed. If False, only a single component is allowed (used for
        skill names and category names).

    Returns
    -------
    str
        The normalized POSIX-style relative path.

    Raises
    ------
    ValueError
        If the path is absolute, contains ``..``, is empty, or has too many
        components for the ``allow_nested`` setting.
    """
    if not isinstance(path, str):
        raise ValueError(f"Unsafe {field_name} (not a string): {path!r}")
    raw = path.strip().replace("\\", "/")
    if not raw:
        raise ValueError(f"Empty {field_name}")

    p = PurePosixPath(raw)
    parts = [seg for seg in p.parts if seg not in {"", "."}]
    if raw.startswith("/") or p.is_absolute():
        raise ValueError(f"Unsafe {field_name} (absolute): {path!r}")
    if not parts or any(seg == ".." for seg in parts):
        raise ValueError(f"Unsafe {field_name} (traversal): {path!r}")
    if not allow_nested and len(parts) > 1:
        raise ValueError(f"Unsafe {field_name} (nested): {path!r}")

    return "/".join(parts)


def _validate_install_parent_path(category: str) -> str:
    """Validate a category/install-parent path (allows nesting like 'mlops/training')."""
    return _normalize_bundle_path(category, field_name="install parent path", allow_nested=True)


# ---------------------------------------------------------------------------
# Lock-file install path validation (the destructive boundary)
# ---------------------------------------------------------------------------


def _normalize_lock_install_path(install_path: str, skill_name: str) -> str:
    """Validate a skill install path before it touches the lock file or disk.

    Lock-file ``install_path`` entries are the source-of-truth for where
    ``uninstall_skill`` will call ``shutil.rmtree``. A poisoned or buggy
    entry — empty string, ``"."``, an absolute path, ``../..`` traversal,
    or anything whose final component doesn't match the skill name — would
    let ``rmtree`` wipe either the entire ``skills/`` tree or content
    outside it.

    Enforce that ``install_path`` ends with ``<skill_name>``. Nested
    official optional skills may legitimately install below paths such as
    ``mlops/training/<skill_name>``; traversal, absolute paths, empty paths,
    and mismatched final components are still rejected.
    """
    safe_skill_name = _validate_skill_name(skill_name)
    normalized = _normalize_bundle_path(
        install_path,
        field_name="install path",
        allow_nested=True,
    )
    parts = normalized.split("/")
    if not parts or parts[-1] != safe_skill_name:
        raise ValueError(f"Unsafe install path: {install_path}")
    return normalized


def _resolve_lock_install_path(install_path: str, skill_name: str, skills_dir: Path) -> Path:
    """Resolve a lock-file install path without allowing escapes from ``skills_dir``.

    Two layers of defence on top of the existing ``is_relative_to`` check:

    1. Walk the path component-by-component and refuse if any intermediate
       component is a symlink/junction (a path resolution that follows a
       symlink to outside skills/ would otherwise be hidden by ``Path.resolve``).
    2. After ``resolve()``, reject not just escape-out but also
       ``resolved == skills_dir`` — an empty/``"."``/``""`` install_path
       resolves to the skills root itself, and ``rmtree(skills_dir)`` would
       wipe every installed skill.
    """
    normalized = _normalize_lock_install_path(install_path, skill_name)
    skills_root = skills_dir.resolve()

    target = skills_dir
    for part in normalized.split("/"):
        target = target / part
        if _is_path_redirect(target):
            raise ValueError(f"Unsafe install path (symlink redirect): {install_path}")

    target = target.resolve()
    if target == skills_root or not target.is_relative_to(skills_root):
        raise ValueError(f"Unsafe install path (escapes skills dir): {install_path}")
    return target


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def append_skill_audit_log(
    audit_path: Path,
    *,
    action: str,
    skill_name: str,
    source: str,
    trust_level: str,
    verdict: str,
    extra: str = "",
) -> None:
    """Append a structured line to the skill hub audit log.

    Format (one JSON line per event, easy to grep and parse):
        2026-07-08T10:30:00Z INSTALL github-pr-workflow official:builtin allowed sha256=...

    Parameters
    ----------
    audit_path : Path
        Path to the audit log file (typically ``~/.nia/skills/.hub/audit.log``).
    action : str
        One of: INSTALL, UNINSTALL, SCAN_BLOCK, SCAN_WARN, FETCH, FETCH_BLOCK.
    skill_name : str
        Validated skill name (use ``_validate_skill_name`` first if unsure).
    source : str
        Source identifier (e.g. "official", "github:user/repo").
    trust_level : str
        One of: builtin, trusted, community.
    verdict : str
        Scan verdict (e.g. "allowed", "blocked", "warn", "n/a").
    extra : str
        Optional free-form context (hash, error message, etc.).
    """
    from datetime import datetime, timezone

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        # Truncate extra to 500 chars to avoid log bloat from large error messages.
        parts.append(extra[:500])
    line = " ".join(parts) + "\n"
    try:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except OSError as exc:
        logger.debug("Could not write skill audit log: %s", exc)


__all__ = [
    "MAX_SKILL_NAME_LENGTH",
    "_is_path_redirect",
    "_validate_skill_name",
    "_normalize_bundle_path",
    "_validate_install_parent_path",
    "_normalize_lock_install_path",
    "_resolve_lock_install_path",
    "append_skill_audit_log",
]
