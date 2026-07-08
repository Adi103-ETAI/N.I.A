"""Resolve NIA_HOME for standalone skill scripts.

Skill scripts may run outside the N.I.A process (e.g. system Python,
nix env, CI) where ``niaharness_constants`` is not importable.  This module
provides the same ``get_niaharness_home()`` and ``display_niaharness_home()``
contracts as ``niaharness_constants`` without requiring it on ``sys.path``.

When ``niaharness_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``niaharness_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``NIA_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from niaharness_constants import display_niaharness_home as display_niaharness_home
    from niaharness_constants import get_niaharness_home as get_niaharness_home
except (ModuleNotFoundError, ImportError):

    def get_niaharness_home() -> Path:
        """Return the N.I.A home directory (default: ~/.niaharness).

        Mirrors ``niaharness_constants.get_niaharness_home()``."""
        val = os.environ.get("NIA_HOME", "").strip()
        return Path(val) if val else Path.home() / ".niaharness"

    def display_niaharness_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``niaharness_constants.display_niaharness_home()``."""
        home = get_niaharness_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)
