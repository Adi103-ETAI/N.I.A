"""Platform detection utilities.

Provides functions to detect the current OS, WSL version, Linux distro info,
and version control systems in use.
"""

from __future__ import annotations

import functools
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


Platform = str  # Literal["macos", "windows", "wsl", "linux", "unknown"]

SUPPORTED_PLATFORMS: list[str] = ["macos", "wsl"]

_VCS_MARKERS: list[tuple[str, str]] = [
    (".git", "git"),
    (".hg", "mercurial"),
    (".svn", "svn"),
    (".p4config", "perforce"),
    ("$tf", "tfs"),
    (".tfvc", "tfs"),
    (".jj", "jujutsu"),
    (".sl", "sapling"),
]


@functools.lru_cache(maxsize=1)
def get_platform() -> Platform:
    """Detect the current platform.

    Returns one of 'macos', 'windows', 'wsl', 'linux', or 'unknown'.
    """
    try:
        sys = platform.system().lower()
        if sys == "darwin":
            return "macos"
        if sys == "windows":
            return "windows"
        if sys == "linux":
            try:
                proc_version = Path("/proc/version").read_text(encoding="utf-8").lower()
                if "microsoft" in proc_version or "wsl" in proc_version:
                    return "wsl"
            except (OSError, IOError):
                pass
            return "linux"
        return "unknown"
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=1)
def get_wsl_version() -> Optional[str]:
    """Return the WSL version string (e.g. '1', '2') or None if not WSL."""
    if platform.system().lower() != "linux":
        return None
    try:
        proc_version = Path("/proc/version").read_text(encoding="utf-8")
        import re

        wsl_match = re.search(r"WSL(\d+)", proc_version, re.IGNORECASE)
        if wsl_match:
            return wsl_match.group(1)
        if "microsoft" in proc_version.lower():
            return "1"
    except (OSError, IOError):
        pass
    return None


@dataclass(frozen=True)
class LinuxDistroInfo:
    """Linux distribution metadata."""

    distro_id: Optional[str] = None
    distro_version: Optional[str] = None
    kernel: Optional[str] = None


@functools.lru_cache(maxsize=1)
async def get_linux_distro_info() -> Optional[LinuxDistroInfo]:
    """Return Linux distro info by reading /etc/os-release."""
    if platform.system().lower() != "linux":
        return None

    kernel = platform.release()
    result = LinuxDistroInfo(kernel=kernel)

    try:
        content = Path("/etc/os-release").read_text(encoding="utf-8")
        d_id: Optional[str] = None
        d_version: Optional[str] = None
        for line in content.splitlines():
            import re

            match = re.match(r'^(ID|VERSION_ID)=(.*)$', line)
            if match:
                value = match.group(2).strip('"')
                if match.group(1) == "ID":
                    d_id = value
                else:
                    d_version = value
        result = LinuxDistroInfo(distro_id=d_id, distro_version=d_version, kernel=kernel)
    except (OSError, IOError):
        pass

    return result


async def detect_vcs(directory: Optional[str] = None) -> List[str]:
    """Detect version control systems in the given directory.

    Checks for well-known VCS marker files/directories and environment
    variables to determine which VCS systems are in use.
    """
    detected: set[str] = set()

    if os.environ.get("P4PORT"):
        detected.add("perforce")

    try:
        target = Path(directory) if directory else Path.cwd()
        entries = {item.name for item in target.iterdir()}
        for marker, vcs_name in _VCS_MARKERS:
            if marker in entries:
                detected.add(vcs_name)
    except (OSError, IOError):
        pass

    return sorted(detected)
