"""Profiles — per-profile isolation of NIA's data, config, and identity.

Ported from the reference project's hermes_cli/profiles.py (3,249 lines),
providing a profile system that lets users maintain multiple NIA
configurations (e.g. "work" and "personal") with isolated:

  - **Identity** (SOUL.md) — different personality for work vs personal
  - **Memory** — separate MEMORY.md and USER.md per profile
  - **Skills** — per-profile user skills directory
  - **Sessions** — per-profile session DB
  - **Credentials** — per-profile credential pool
  - **Config** — per-profile config.yaml overrides
  - **Cron jobs** — per-profile cron schedule

Profile resolution
------------------
The active profile is determined by:

  1. ``--profile <name>`` CLI flag (highest priority)
  2. ``NIA_PROFILE`` environment variable
  3. ``~/.nia/active_profile`` file (set by ``niaharness profile switch``)
  4. ``"default"`` (lowest priority)

Each profile has its own directory at ``~/.nia/profiles/<name>/``. The
``default`` profile uses the root ``~/.nia/`` directory itself (backward
compat — existing files at ``~/.nia/SOUL.md`` continue to work without
migration).

Usage::

    from niaharness.profiles import get_active_profile, switch_profile

    profile = get_active_profile()
    print(f"Active profile: {profile.name}")
    print(f"Home: {profile.home}")

    switch_profile("work")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


DEFAULT_PROFILE = "default"
ACTIVE_PROFILE_FILE = "active_profile"
PROFILES_DIR = "profiles"


def _nia_home_root() -> Path:
    """Return the root NIA home directory (~/.nia/).

    This is the ROOT home — NOT profile-aware. It resolves env vars
    (NIA_HOME, NIAHARNESS_CONFIG_DIR) or defaults to ~/.nia/.
    Profile-aware path resolution is done by get_nia_home() in soul.py,
    which calls get_active_profile_name() from this module. To avoid
    a circular import, this function does NOT call get_nia_home().
    """
    for env_var in ("NIA_HOME", "NIAHARNESS_CONFIG_DIR"):
        value = os.environ.get(env_var)
        if value:
            return Path(value)
    return Path.home() / ".nia"


def _profiles_root() -> Path:
    """Return the profiles root directory (~/.nia/profiles/)."""
    return _nia_home_root() / PROFILES_DIR


def _active_profile_file() -> Path:
    """Return the path to the active-profile marker file."""
    return _nia_home_root() / ACTIVE_PROFILE_FILE


@dataclass(frozen=True)
class Profile:
    """A NIA profile — a named configuration with isolated data.

    Attributes:
        name: The profile name (e.g. "default", "work", "personal").
        home: The profile's home directory.
        is_default: True if this is the default profile.
        exists: True if the profile directory exists.
    """

    name: str
    home: Path
    is_default: bool = False
    exists: bool = False

    @property
    def soul_md_path(self) -> Path:
        return self.home / "SOUL.md"

    @property
    def memory_path(self) -> Path:
        return self.home / "MEMORY.md"

    @property
    def user_md_path(self) -> Path:
        return self.home / "USER.md"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def sessions_db_path(self) -> Path:
        return self.home / "sessions.db"

    @property
    def credentials_dir(self) -> Path:
        return self.home / "credentials"

    @property
    def config_path(self) -> Path:
        return self.home / "config.yaml"

    @property
    def cron_dir(self) -> Path:
        return self.home / "cron"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "home": str(self.home),
            "is_default": self.is_default,
            "exists": self.exists,
        }


def get_active_profile_name() -> str:
    """Return the active profile name."""
    env_profile = os.environ.get("NIA_PROFILE")
    if env_profile and env_profile.strip():
        return env_profile.strip()

    active_file = _active_profile_file()
    if active_file.exists():
        try:
            name = active_file.read_text(encoding="utf-8").strip()
            if name:
                return name
        except OSError:
            pass

    return DEFAULT_PROFILE


def get_profile(name: Optional[str] = None) -> Profile:
    """Return the Profile for ``name`` (or the active profile if None)."""
    if name is None:
        name = get_active_profile_name()

    is_default = name == DEFAULT_PROFILE
    if is_default:
        home = _nia_home_root()
    else:
        home = _profiles_root() / name
        home.mkdir(parents=True, exist_ok=True)

    return Profile(
        name=name,
        home=home,
        is_default=is_default,
        exists=home.exists(),
    )


def get_active_profile() -> Profile:
    """Return the currently-active Profile."""
    return get_profile(get_active_profile_name())


def switch_profile(name: str) -> Profile:
    """Switch the active profile to ``name``."""
    if not name or not name.strip():
        raise ValueError("Profile name cannot be empty")
    name = name.strip()
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Profile name contains forbidden characters: {name!r}")

    profile = get_profile(name)
    active_file = _active_profile_file()
    active_file.parent.mkdir(parents=True, exist_ok=True)
    active_file.write_text(name, encoding="utf-8")

    logger.info("Switched to profile '%s' (home: %s)", name, profile.home)
    return profile


def list_profiles() -> List[Profile]:
    """List all profiles (default + any in ~/.nia/profiles/)."""
    profiles: List[Profile] = [get_profile(DEFAULT_PROFILE)]

    profiles_root = _profiles_root()
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                profiles.append(Profile(
                    name=entry.name,
                    home=entry,
                    is_default=False,
                    exists=True,
                ))

    return profiles


def create_profile(name: str, *, seed_from_default: bool = False) -> Profile:
    """Create a new profile."""
    if not name or not name.strip():
        raise ValueError("Profile name cannot be empty")
    name = name.strip()
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Profile name contains forbidden characters: {name!r}")
    if name == DEFAULT_PROFILE:
        raise ValueError(f"Cannot create profile '{name}' — it's the default")

    profile = get_profile(name)
    profile.home.mkdir(parents=True, exist_ok=True)

    if seed_from_default:
        default = get_profile(DEFAULT_PROFILE)
        if default.soul_md_path.exists():
            profile.soul_md_path.write_text(
                default.soul_md_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if default.config_path.exists():
            profile.config_path.write_text(
                default.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    logger.info("Created profile '%s' (home: %s)", name, profile.home)
    return profile


def delete_profile(name: str) -> bool:
    """Delete a profile. Returns True if found and deleted."""
    if name == DEFAULT_PROFILE:
        raise ValueError("Cannot delete the default profile")
    if name == get_active_profile_name():
        raise ValueError(
            f"Cannot delete the active profile '{name}'. "
            "Switch to another profile first."
        )

    profile = get_profile(name)
    if not profile.exists:
        return False

    import shutil

    shutil.rmtree(profile.home, ignore_errors=True)
    logger.info("Deleted profile '%s'", name)
    return True


def get_profile_home(name: Optional[str] = None) -> Path:
    return get_profile(name).home


def get_profile_soul_md_path(name: Optional[str] = None) -> Path:
    return get_profile(name).soul_md_path


def get_profile_skills_dir(name: Optional[str] = None) -> Path:
    d = get_profile(name).skills_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_profile_sessions_db_path(name: Optional[str] = None) -> Path:
    return get_profile(name).sessions_db_path


def get_profile_credentials_dir(name: Optional[str] = None) -> Path:
    d = get_profile(name).credentials_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_profile_cron_dir(name: Optional[str] = None) -> Path:
    d = get_profile(name).cron_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "ACTIVE_PROFILE_FILE",
    "DEFAULT_PROFILE",
    "PROFILES_DIR",
    "Profile",
    "create_profile",
    "delete_profile",
    "get_active_profile",
    "get_active_profile_name",
    "get_profile",
    "get_profile_credentials_dir",
    "get_profile_cron_dir",
    "get_profile_home",
    "get_profile_sessions_db_path",
    "get_profile_skills_dir",
    "get_profile_soul_md_path",
    "list_profiles",
    "switch_profile",
]
