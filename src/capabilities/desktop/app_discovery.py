"""
MODULE: Application Index (Dynamic Discovery Engine)
VERSION: 1.0.0
SCOPE: Index ALL installed applications via Windows Start Menu and launch them.
RUNS ON: Host OS (NOT Docker).

Uses PowerShell `Get-StartApps | ConvertTo-Json` to build a complete index of
every launchable app on the system — Win32 .exe, UWP Store apps, and Shell apps.
Supports 3-tier search: Exact -> Token -> Fuzzy.
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.logger import setup_logger

logger = setup_logger("TARA.AppIndex")

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AppEntry:
    """A launchable application discovered from the Start Menu index."""
    name: str           # Human-friendly name, e.g. "Visual Studio Code"
    app_id: str         # Raw AppID from Get-StartApps
    app_type: str       # "win32" | "uwp" | "shell"

    def display(self) -> str:
        return f"{self.name} [{self.app_type}]"


# =============================================================================
# AppIndex
# =============================================================================

# Cache TTL: 10 minutes (apps rarely install mid-session)
_CACHE_TTL_SECONDS = 600


class AppIndex:
    """
    Dynamic Application Discovery Engine.

    Builds an in-memory index of ALL installed applications by calling
    PowerShell's `Get-StartApps`. Provides 3-tier search (exact, token,
    fuzzy) and universal launch via `os.startfile`.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, AppEntry] = {}       # name_lower -> AppEntry
        self._all_entries: List[AppEntry] = []       # ordered list
        self._last_refresh: float = 0.0
        self._refresh_count: int = 0

    # =========================================================================
    # Index Builder
    # =========================================================================

    def refresh(self) -> int:
        """
        Rebuild the app index by calling Get-StartApps.

        Returns:
            Number of apps indexed.
        """
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-StartApps | ConvertTo-Json"
                ],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if result.returncode != 0:
                logger.error(f"Get-StartApps failed: {result.stderr.strip()}")
                return 0

            raw = result.stdout.strip()
            if not raw:
                logger.warning("Get-StartApps returned empty output")
                return 0

            data = json.loads(raw)

            # Handle single-item (returns dict instead of list)
            if isinstance(data, dict):
                data = [data]

        except subprocess.TimeoutExpired:
            logger.error("Get-StartApps timed out after 10s")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Get-StartApps JSON: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during index refresh: {e}")
            return 0

        # Build the index
        new_cache: Dict[str, AppEntry] = {}
        new_entries: List[AppEntry] = []

        for item in data:
            name = item.get("Name", "").strip()
            app_id = item.get("AppID", "").strip()

            if not name or not app_id:
                continue

            # Classify app type
            if app_id.endswith(".exe") or os.sep in app_id or "/" in app_id:
                app_type = "win32"
            elif "!" in app_id:
                app_type = "uwp"
            else:
                app_type = "shell"

            entry = AppEntry(name=name, app_id=app_id, app_type=app_type)
            name_lower = name.lower()

            # Store by lowercase name (first occurrence wins for duplicates)
            if name_lower not in new_cache:
                new_cache[name_lower] = entry

            new_entries.append(entry)

        self._cache = new_cache
        self._all_entries = new_entries
        self._last_refresh = time.time()
        self._refresh_count += 1

        logger.info(
            f"AppIndex refreshed: {len(new_entries)} apps indexed "
            f"(refresh #{self._refresh_count})"
        )
        return len(new_entries)

    def _ensure_index(self) -> None:
        """Lazy-load: refresh index if stale or empty."""
        age = time.time() - self._last_refresh
        if not self._cache or age > _CACHE_TTL_SECONDS:
            self.refresh()

    # =========================================================================
    # 3-Tier Search
    # =========================================================================

    def search(self, query: str) -> Optional[AppEntry]:
        """
        Find an application by name using 3-tier matching.

        Tier 1: Exact match (case-insensitive).
        Tier 2: Token match (query is a substring of app name).
        Tier 3: Fuzzy match (difflib, ratio > 0.6).

        Args:
            query: Human-friendly search query (e.g., "code", "whatsapp").

        Returns:
            Best matching AppEntry, or None.
        """
        self._ensure_index()

        if not query or not query.strip():
            return None

        q = query.lower().strip()

        # Tier 1: Exact match
        if q in self._cache:
            entry = self._cache[q]
            logger.debug(f"Tier 1 (exact): '{query}' -> {entry.display()}")
            return entry

        # Tier 2: Token match (query is contained in app name)
        token_matches: List[AppEntry] = []
        for name_lower, entry in self._cache.items():
            if q in name_lower:
                token_matches.append(entry)

        if token_matches:
            # Prefer shortest name (closest match)
            best = min(token_matches, key=lambda e: len(e.name))
            logger.debug(
                f"Tier 2 (token): '{query}' -> {best.display()} "
                f"(from {len(token_matches)} candidates)"
            )
            return best

        # Tier 3: Fuzzy match (difflib)
        all_names = list(self._cache.keys())
        close = difflib.get_close_matches(q, all_names, n=1, cutoff=0.5)

        if close:
            entry = self._cache[close[0]]
            logger.debug(f"Tier 3 (fuzzy): '{query}' -> {entry.display()}")
            return entry

        logger.debug(f"No match found for '{query}'")
        return None

    def search_all(self, query: str, limit: int = 5) -> List[AppEntry]:
        """
        Find all matching apps (for disambiguation).

        Returns:
            List of matching AppEntry objects, up to `limit`.
        """
        self._ensure_index()

        if not query or not query.strip():
            return []

        q = query.lower().strip()
        results: List[AppEntry] = []

        # Exact
        if q in self._cache:
            results.append(self._cache[q])

        # Token matches
        for name_lower, entry in self._cache.items():
            if q in name_lower and entry not in results:
                results.append(entry)

        # Fuzzy matches
        if len(results) < limit:
            all_names = list(self._cache.keys())
            close = difflib.get_close_matches(q, all_names, n=limit, cutoff=0.5)
            for match_name in close:
                entry = self._cache[match_name]
                if entry not in results:
                    results.append(entry)

        return results[:limit]

    # =========================================================================
    # Launcher
    # =========================================================================

    def launch(self, entry: AppEntry) -> str:
        """
        Launch an application by its AppEntry.

        Uses `os.startfile` with the shell:AppsFolder protocol, which is
        the universal launcher for Win32, UWP, and Shell apps.

        Args:
            entry: The AppEntry to launch.

        Returns:
            Status message.
        """
        try:
            if entry.app_type == "win32" and os.path.isfile(entry.app_id):
                # Direct .exe path — launch directly for PID tracking
                proc = subprocess.Popen(
                    [entry.app_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                logger.info(f"Launched {entry.name} (Win32, PID={proc.pid})")
                return f"launched:win32:pid={proc.pid}"
            else:
                # UWP / Shell / non-path Win32 — use shell:AppsFolder
                os.startfile(f"shell:AppsFolder\\{entry.app_id}")
                logger.info(f"Launched {entry.name} ({entry.app_type}, via shell:AppsFolder)")
                return f"launched:{entry.app_type}:shell"

        except OSError as e:
            logger.error(f"Failed to launch {entry.name}: {e}")
            return f"error:{e}"
        except Exception as e:
            logger.error(f"Unexpected error launching {entry.name}: {e}")
            return f"error:{e}"

    # =========================================================================
    # Utilities
    # =========================================================================

    def list_apps(self, filter_query: Optional[str] = None, limit: int = 30) -> str:
        """
        List indexed apps, optionally filtered.

        Args:
            filter_query: Optional substring filter.
            limit: Max results.

        Returns:
            Formatted app list.
        """
        self._ensure_index()

        entries = self._all_entries
        if filter_query:
            q = filter_query.lower()
            entries = [e for e in entries if q in e.name.lower()]

        if not entries:
            return f"No apps found matching '{filter_query}'" if filter_query else "No apps indexed"

        lines = [f"Installed Apps ({min(len(entries), limit)}/{len(entries)}):"]
        for entry in entries[:limit]:
            lines.append(f"  {entry.name:<40} [{entry.app_type}]")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Number of indexed apps."""
        self._ensure_index()
        return len(self._all_entries)


# =============================================================================
# Singleton
# =============================================================================

_index: Optional[AppIndex] = None


def get_app_index() -> AppIndex:
    """Get or create the global AppIndex singleton."""
    global _index
    if _index is None:
        _index = AppIndex()
    return _index


__all__ = [
    "AppIndex",
    "AppEntry",
    "get_app_index",
]
