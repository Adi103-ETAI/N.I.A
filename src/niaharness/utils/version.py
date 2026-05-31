"""Version info utilities.

Provides version normalization and release URL generation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


NIAHARNESS_RELEASES_URL = "https://github.com/niaharness/niaharness/releases"

_FALLBACK_VERSION = "0.0.0"


def normalize_public_version(version: str) -> str:
    """Normalize a version string to semver format.

    Strips leading 'v' and attempts to coerce to a valid semver string.
    """
    trimmed = version.strip()

    # Try to extract semver-like pattern
    semver_match = re.match(r"v?(\d+(?:\.\d+(?:\.\d+)?)?)", trimmed, re.IGNORECASE)
    if semver_match:
        return semver_match.group(1)

    # Strip leading 'v' as fallback
    return re.sub(r"^v", "", trimmed, flags=re.IGNORECASE)


def _read_package_version_from_disk() -> Optional[str]:
    """Walk up directories to find package.json with a version field."""
    current = Path(__file__).resolve().parent

    while True:
        pyproject = current / "pyproject.toml"
        setup = current / "setup.py"

        # Try pyproject.toml
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                match = re.search(r'^version\s*=\s*["\'](.+?)["\']', content, re.MULTILINE)
                if match:
                    return match.group(1)
            except (OSError, IOError):
                pass

        # Try setup.py
        if setup.exists():
            try:
                content = setup.read_text(encoding="utf-8")
                match = re.search(r'version\s*=\s*["\'](.+?)["\']', content)
                if match:
                    return match.group(1)
            except (OSError, IOError):
                pass

        # Try package.json (for hybrid projects)
        package_json = current / "package.json"
        if package_json.exists():
            try:
                import json

                data = json.loads(package_json.read_text(encoding="utf-8"))
                ver = data.get("version")
                if isinstance(ver, str) and ver.strip():
                    return ver.strip()
            except (OSError, IOError, ValueError):
                pass

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _get_build_version() -> str:
    """Get the build version from environment or package metadata."""
    # Environment variable override
    env_version = os.environ.get("NIAHARNESS_VERSION")
    if env_version:
        return env_version

    # Try to read from package metadata
    disk_version = _read_package_version_from_disk()
    if disk_version:
        return disk_version

    return _FALLBACK_VERSION


PUBLIC_BUILD_VERSION: str = normalize_public_version(_get_build_version())


def get_release_tag_url(version: Optional[str] = None) -> str:
    """Get the URL for a release tag."""
    v = normalize_public_version(version or PUBLIC_BUILD_VERSION)
    return f"{NIAHARNESS_RELEASES_URL}/tag/v{v}"


def get_public_build_version() -> str:
    """Return the public build version string."""
    return PUBLIC_BUILD_VERSION
