"""P1 Profile extensions — 8 missing profile features.

Ported from Hermes Agent's ``hermes_cli/profiles.py`` (2225 LOC) +
``hermes_cli/profile_distribution.py`` (726 LOC), scoped to NIA's
architecture. Provides the 8 missing profile features identified in
AUDIT.md:

  1. :func:`export_profile` — export a profile to a tar.gz archive
     (excludes credentials).
  2. :func:`import_profile` — import a profile from a tar.gz archive
     (with path-traversal protection).
  3. :func:`rename_profile` — rename a profile (directory + active + aliases).
  4. :func:`seed_profile_skills` — seed bundled skills into a new profile.
  5. :func:`backfill_profile_envs` — give every named profile a .env file.
  6. :func:`profiles_to_serve` — return (name, home) pairs for gateway multiplex.
  7. :func:`read_profile_meta` / :func:`write_profile_meta` — profile.yaml
     metadata (description, description_auto).
  8. :func:`get_distribution_meta` / :func:`set_distribution_meta` —
     distribution metadata for profile sharing.

All functions are safe to call on the default profile (which lives at
~/.nia itself, not under profiles/). Named profiles live at
~/.nia/profiles/<name>/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


def _profiles_root() -> Path:
    """Return the profiles root directory (~/.nia/profiles/)."""
    return _get_nia_home() / "profiles"


def _profile_dir(name: str) -> Path:
    """Return the directory for a named profile."""
    if name == "default":
        return _get_nia_home()
    return _profiles_root() / name


# ---------------------------------------------------------------------------
# 1. Export profile
# ---------------------------------------------------------------------------


# Files to exclude from exports (credentials + runtime state).
_CREDENTIAL_FILES = {"auth.json", ".env", "credentials", ".credentials"}
_RUNTIME_FILES = {".hub", "sessions.db", "sessions.db-wal", "sessions.db-shm"}


def _default_export_ignore(source_dir: Path):
    """Build an ignore function for shutil.copytree that excludes credentials + runtime."""
    def _ignore(directory: Path, contents: List[str]) -> List[str]:
        excluded: List[str] = []
        for item in contents:
            if item in _CREDENTIAL_FILES or item in _RUNTIME_FILES:
                excluded.append(item)
        return excluded
    return _ignore


def export_profile(name: str, output_path: str) -> Path:
    """Export a profile to a tar.gz archive.

    Excludes credentials (.env, auth.json, credentials/) and runtime state
    (sessions.db, .hub/). Safe to share the resulting archive.

    Args:
        name: The profile name (or "default").
        output_path: The output file path (.tar.gz or .tgz).

    Returns:
        The path to the created archive.
    """
    import tempfile

    source = _profile_dir(name)
    if not source.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist at {source}")

    output = Path(output_path)
    base = str(output).removesuffix(".tar.gz").removesuffix(".tgz")

    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / name
        shutil.copytree(
            source,
            staged,
            symlinks=True,
            ignore=_default_export_ignore(source),
        )
        result = shutil.make_archive(base, "gztar", tmpdir, name)
        return Path(result)


# ---------------------------------------------------------------------------
# 2. Import profile
# ---------------------------------------------------------------------------


def _normalize_archive_member(member_name: str) -> List[str]:
    """Return safe path parts for a profile archive member.

    Rejects absolute paths, path traversal, and Windows drive letters.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    normalized = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(member_name)

    if (
        not normalized
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"Unsafe archive member path: {member_name}")

    parts = [part for part in posix_path.parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return parts


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract a profile archive without allowing path escapes or symlinks."""
    import tarfile

    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _normalize_archive_member(member.name)
            target = destination.joinpath(*parts)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise ValueError(
                    f"Unsupported archive member type: {member.name}"
                )

            # Reject symlinks/hardlinks (path traversal vector).
            if member.issym() or member.islnk():
                raise ValueError(f"Archive contains a link: {member.name}")

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read archive member: {member.name}")

            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)

            try:
                os.chmod(target, member.mode & 0o777)
            except OSError:
                pass


def _inspect_archive_roots(archive: Path) -> set[str]:
    """Return the archive's top-level directory names."""
    import tarfile

    roots: set[str] = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _normalize_archive_member(member.name)
            if parts:
                roots.add(parts[0])
    return roots


def import_profile(
    archive_path: str,
    name: Optional[str] = None,
) -> Path:
    """Import a profile from a tar.gz archive.

    Args:
        archive_path: Path to the .tar.gz archive.
        name: Optional profile name. If not given, infers from the archive's
            top-level directory.

    Returns:
        The imported profile directory.

    Raises:
        FileNotFoundError: Archive not found.
        ValueError: Cannot determine name, or name is "default" (reserved).
        FileExistsError: Profile already exists.
    """
    import tempfile

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    top_dirs = _inspect_archive_roots(archive)
    archive_root = top_dirs.pop() if len(top_dirs) == 1 else None
    inferred_name = name or archive_root

    if not inferred_name:
        raise ValueError(
            "Cannot determine profile name from archive. "
            "Specify it explicitly: nia profile import <archive> --name <name>"
        )
    if archive_root is None:
        raise ValueError(
            "Profile archive must contain exactly one top-level directory."
        )

    if inferred_name == "default":
        raise ValueError(
            "Cannot import as 'default' — that is the built-in root profile. "
            "Specify a different name."
        )

    # Validate name.
    from niaharness.profiles import create_profile, list_profiles
    existing = {p.name for p in list_profiles()}
    if inferred_name in existing:
        raise FileExistsError(f"Profile '{inferred_name}' already exists")

    profile_dir = _profile_dir(inferred_name)
    _profiles_root().mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nia_profile_import_") as tmpdir:
        staging = Path(tmpdir)
        _safe_extract_archive(archive, staging)

        extracted = staging / archive_root
        if not extracted.is_dir():
            raise ValueError(f"Archive root is missing or invalid: {archive_root}")

        # Rename to the target name if different.
        final_source = extracted
        if archive_root != inferred_name:
            final_source = staging / inferred_name
            extracted.rename(final_source)

        shutil.move(str(final_source), str(profile_dir))

    logger.info("Imported profile '%s' from %s", inferred_name, archive)
    return profile_dir


# ---------------------------------------------------------------------------
# 3. Rename profile
# ---------------------------------------------------------------------------


def rename_profile(old_name: str, new_name: str) -> Path:
    """Rename a profile: directory, active_profile, aliases.

    Args:
        old_name: The current profile name.
        new_name: The new profile name.

    Returns:
        The new profile directory.

    Raises:
        ValueError: old_name or new_name is "default" (reserved).
        FileNotFoundError: old_name doesn't exist.
        FileExistsError: new_name already exists.
    """
    if old_name == "default":
        raise ValueError("Cannot rename the default profile.")
    if new_name == "default":
        raise ValueError("Cannot rename to 'default' — it is reserved.")

    old_dir = _profile_dir(old_name)
    new_dir = _profile_dir(new_name)

    if not old_dir.is_dir():
        raise FileNotFoundError(f"Profile '{old_name}' does not exist.")
    if new_dir.exists():
        raise FileExistsError(f"Profile '{new_name}' already exists.")

    # 1. Rename directory.
    old_dir.rename(new_dir)

    # 2. Update active_profile if it pointed to old name.
    try:
        from niaharness.profiles import get_active_profile_name, switch_profile
        if get_active_profile_name() == old_name:
            switch_profile(new_name)
    except Exception:
        pass

    # 3. Update aliases if they exist.
    try:
        from niaharness.profiles.aliases import remove_alias, set_alias
        remove_alias(old_name)
        set_alias(new_name, new_name)
    except Exception:
        pass

    logger.info("Renamed profile '%s' → '%s'", old_name, new_name)
    return new_dir


# ---------------------------------------------------------------------------
# 4. Seed profile skills
# ---------------------------------------------------------------------------


def has_bundled_skills_opt_out(profile_dir: Path) -> bool:
    """Check if a profile has opted out of bundled skills."""
    return (profile_dir / ".no-bundled-skills").exists()


def seed_profile_skills(profile_dir: Path, quiet: bool = False) -> Optional[dict]:
    """Seed bundled skills into a profile.

    Uses subprocess so the skill sync runs with the profile's NIA_HOME.
    Profiles that opted out of bundled skills (via .no-bundled-skills
    marker) are skipped.

    Returns the sync result dict, or None on failure.
    """
    if has_bundled_skills_opt_out(profile_dir):
        return {
            "copied": [],
            "updated": [],
            "user_modified": [],
            "skipped_opt_out": True,
        }

    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; from niaharness.skills.bundled import sync_bundled_skills; "
             "r = sync_bundled_skills(quiet=True); print(json.dumps(r))"],
            env={**os.environ, "NIA_HOME": str(profile_dir)},
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        if not quiet:
            logger.warning("Skill seeding returned exit code %d", result.returncode)
        return None
    except subprocess.TimeoutExpired:
        if not quiet:
            logger.warning("Skill seeding timed out (60s)")
        return None
    except Exception as exc:
        if not quiet:
            logger.warning("Skill seeding failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 5. Backfill profile envs
# ---------------------------------------------------------------------------


def backfill_profile_envs(quiet: bool = False) -> List[str]:
    """Give every named profile that lacks a .env file one.

    Copies the default install's .env into each named profile that doesn't
    have one. Never overwrites an existing .env. This is a migration helper
    for profiles created before per-profile .env files were standard.

    Returns the list of profile names that received a backfilled .env.
    """
    import re

    backfilled: List[str] = []
    root = _profiles_root()
    if not root.is_dir():
        return backfilled

    default_env = _get_nia_home() / ".env"
    _PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not _PROFILE_ID_RE.match(entry.name):
            continue
        if entry.name == "default":
            continue
        env_path = entry / ".env"
        if env_path.exists():
            continue
        try:
            if default_env.is_file():
                shutil.copy2(default_env, env_path)
            else:
                env_path.write_text(
                    "# Per-profile secrets for this NIA profile.\n"
                    "# API keys and tokens set here override the shell environment.\n",
                    encoding="utf-8",
                )
            os.chmod(str(env_path), 0o600)
            backfilled.append(entry.name)
        except OSError as exc:
            if not quiet:
                logger.warning("Could not seed .env for profile '%s': %s", entry.name, exc)

    return backfilled


# ---------------------------------------------------------------------------
# 6. profiles_to_serve — gateway multiplex support
# ---------------------------------------------------------------------------


def profiles_to_serve(multiplex: bool) -> List[Tuple[str, Path]]:
    """Return (profile_name, nia_home) pairs a gateway should serve.

    - ``multiplex=False``: returns exactly one entry for the active profile.
    - ``multiplex=True``: returns the default profile + every valid named profile.

    This is the single chokepoint for "which profiles does the gateway handle".
    """
    import re

    try:
        from niaharness.profiles import get_active_profile_name
        active = get_active_profile_name() or "default"
    except Exception:
        active = "default"

    if not multiplex:
        return [(active, _profile_dir(active))]

    serve: List[Tuple[str, Path]] = [("default", _get_nia_home())]

    root = _profiles_root()
    if root.is_dir():
        _PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name == "default":
                continue
            if not _PROFILE_ID_RE.match(name):
                continue
            serve.append((name, entry))

    return serve


# ---------------------------------------------------------------------------
# 7. Profile metadata YAML (profile.yaml)
# ---------------------------------------------------------------------------


def _profile_yaml_path(profile_dir: Path) -> Path:
    """Return the path to profile.yaml."""
    return profile_dir / "profile.yaml"


def read_profile_meta(profile_dir: Path) -> dict:
    """Read profile.yaml and return a dict.

    Returns ``{"description": "", "description_auto": False}`` when the
    file is missing or unreadable. Never raises.
    """
    path = _profile_yaml_path(profile_dir)
    if not path.is_file():
        return {"description": "", "description_auto": False}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {"description": "", "description_auto": False}
    if not isinstance(data, dict):
        return {"description": "", "description_auto": False}
    return {
        "description": str(data.get("description") or "").strip(),
        "description_auto": bool(data.get("description_auto", False)),
    }


def write_profile_meta(
    profile_dir: Path,
    *,
    description: Optional[str] = None,
    description_auto: Optional[bool] = None,
) -> None:
    """Update profile.yaml in place.

    Only the explicitly passed fields are overwritten; unspecified fields
    preserve existing values. Creates the file if missing.
    """
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory does not exist: {profile_dir}")

    import yaml
    path = _profile_yaml_path(profile_dir)
    existing: dict = {}
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    if description is not None:
        existing["description"] = description.strip()
    if description_auto is not None:
        existing["description_auto"] = bool(description_auto)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# 8. Distribution metadata
# ---------------------------------------------------------------------------


def _distribution_meta_path(profile_dir: Path) -> Path:
    """Return the path to the distribution metadata file."""
    return profile_dir / ".distribution.json"


def get_distribution_meta(profile_dir: Path) -> Optional[dict]:
    """Read distribution metadata from a profile.

    Distribution metadata tracks how a profile was distributed (exported,
    imported, cloned). Used by the profile sharing system to trace provenance.

    Returns None if no distribution metadata exists.
    """
    path = _distribution_meta_path(profile_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def set_distribution_meta(
    profile_dir: Path,
    *,
    source: str = "",
    exported_at: Optional[float] = None,
    imported_at: Optional[float] = None,
    version: str = "1.0",
    extra: Optional[dict] = None,
) -> None:
    """Write distribution metadata for a profile.

    Args:
        profile_dir: The profile directory.
        source: Where the profile came from ("export", "import", "clone").
        exported_at: Unix timestamp when the profile was exported.
        imported_at: Unix timestamp when the profile was imported.
        version: Distribution format version.
        extra: Additional metadata.
    """
    import time as _time

    path = _distribution_meta_path(profile_dir)
    meta: dict[str, Any] = {
        "source": source,
        "version": version,
        "created_at": _time.time(),
    }
    if exported_at is not None:
        meta["exported_at"] = exported_at
    if imported_at is not None:
        meta["imported_at"] = imported_at
    if extra:
        meta.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Gateway service registration (stub for future use)
# ---------------------------------------------------------------------------


def maybe_register_gateway_service(profile_name: str) -> None:
    """Register a gateway systemd service for a profile (future use).

    Currently a no-op — NIA doesn't have systemd service management yet.
    When implemented, this will create a systemd user service unit for
    the gateway so it starts on boot.
    """
    pass


def maybe_unregister_gateway_service(profile_name: str) -> None:
    """Unregister a gateway systemd service for a profile (future use)."""
    pass


__all__ = [
    "backfill_profile_envs",
    "export_profile",
    "get_distribution_meta",
    "has_bundled_skills_opt_out",
    "import_profile",
    "maybe_register_gateway_service",
    "maybe_unregister_gateway_service",
    "profiles_to_serve",
    "read_profile_meta",
    "rename_profile",
    "seed_profile_skills",
    "set_distribution_meta",
    "write_profile_meta",
]
