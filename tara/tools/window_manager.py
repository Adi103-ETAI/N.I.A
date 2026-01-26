"""
Singleton Registry to map Window Aliases -> Handles/PIDs.
Shared between Launcher and WindowOps.

TARA 2.0 Persistence Layer.

This module provides state persistence across tool calls, enabling:
- Multi-instance window tracking (notepad_1, notepad_2)
- PID-to-HWND mapping for launch verification
- Alias-based window operations

Usage:
    from tara.tools.window_manager import get_registry
    
    registry = get_registry()
    alias = registry.register("notepad", hwnd=12345, pid=6789, title="Untitled")
    info = registry.get(alias)
    registry.deregister(alias)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import setup_logger
from core.config import settings

logger = setup_logger("TARA.Tools.WindowManager")
REGISTRY_FILE = settings.WINDOW_REGISTRY_FILE


# =============================================================================
# WindowInfo Data Structure
# =============================================================================

@dataclass
class WindowInfo:
    """Information about a tracked window."""
    alias: str
    app_name: str
    hwnd: Optional[int] = None
    pid: Optional[int] = None
    title: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alias": self.alias,
            "app": self.app_name,
            "hwnd": self.hwnd,
            "pid": self.pid,
            "title": self.title,
        }


# =============================================================================
# WindowRegistry Singleton
# =============================================================================

class WindowRegistry:
    """
    Registry for tracking opened windows by alias.
    
    Use get_registry() or container.window_registry to access.
    
    Thread-safe state persistence for multi-instance window control.
    
    Attributes:
        _windows: Dict mapping alias -> WindowInfo
        _counters: Dict mapping app_name -> count (for auto-numbering)
        _lock: Threading lock for thread-safe access
    """
    
    def __init__(self) -> None:
        """Initialize a new WindowRegistry instance."""
        self._windows: Dict[str, WindowInfo] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        # Ensure data directory exists
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-load on startup
        self.load_registry()
        
        logger.debug("WindowRegistry initialized (Persistence: Enabled)")
    
    def register(
        self,
        app_name: str,
        hwnd: Optional[int] = None,
        pid: Optional[int] = None,
        title: str = "",
    ) -> str:
        """
        Register a window with auto-generated alias.
        
        Args:
            app_name: Base app name (e.g., "notepad").
            hwnd: Window handle (HWND).
            pid: Process ID.
            title: Window title.
            
        Returns:
            Generated alias (e.g., "notepad_1").
        """
        # 🌊 RIPPLE CHECK: Locking thread access
        with self._lock:
            # Normalize app name for alias generation
            base = app_name.lower().replace(" ", "_").replace(".exe", "")
            
            # Auto-increment counter for this app
            count = self._counters.get(base, 0) + 1
            self._counters[base] = count
            alias = f"{base}_{count}"
            
            # Create and store WindowInfo
            info = WindowInfo(
                alias=alias,
                app_name=app_name,
                hwnd=hwnd,
                pid=pid,
                title=title or app_name,
            )
            self._windows[alias] = info
            
            # Auto-save
            self.save_registry()
            
            logger.info(f"📝 Registered: {alias} (HWND={hwnd}, PID={pid})")
            return alias
    
    def deregister(self, alias: str) -> bool:
        """
        Remove a window from the registry.
        
        Args:
            alias: Window alias to remove.
            
        Returns:
            True if removed, False if not found.
        """
        # 🌊 RIPPLE CHECK: Locking thread access
        with self._lock:
            if alias in self._windows:
                del self._windows[alias]
                self.save_registry()
                logger.info(f"🗑️ Deregistered: {alias}")
                return True
            logger.warning(f"Alias '{alias}' not found for deregistration")
            return False
    
    def get(self, alias: str) -> Optional[WindowInfo]:
        """
        Get window info by alias.
        
        Args:
            alias: Window alias.
            
        Returns:
            WindowInfo or None if not found.
        """
        with self._lock:
            return self._windows.get(alias)
    
    def get_handle(self, alias: str) -> Optional[int]:
        """
        Get window handle by alias.
        
        Args:
            alias: Window alias.
            
        Returns:
            HWND or None if not found.
        """
        with self._lock:
            info = self._windows.get(alias)
            return info.hwnd if info else None
    
    def get_pid(self, alias: str) -> Optional[int]:
        """
        Get process ID by alias.
        
        Args:
            alias: Window alias.
            
        Returns:
            PID or None if not found.
        """
        with self._lock:
            info = self._windows.get(alias)
            return info.pid if info else None
    
    def find_by_pid(self, pid: int) -> Optional[str]:
        """
        Find alias by process ID.
        
        Args:
            pid: Process ID to search for.
            
        Returns:
            Alias or None if not found.
        """
        with self._lock:
            for alias, info in self._windows.items():
                if info.pid == pid:
                    return alias
            return None
    
    def find_by_title(self, title: str, partial: bool = True) -> Optional[str]:
        """
        Find alias by window title.
        
        Args:
            title: Window title (or partial).
            partial: If True, match substring.
            
        Returns:
            First matching alias or None.
        """
        title_lower = title.lower()
        with self._lock:
            for alias, info in self._windows.items():
                if partial:
                    if title_lower in info.title.lower():
                        return alias
                else:
                    if title_lower == info.title.lower():
                        return alias
            return None
    
    def list_windows(self) -> List[Dict[str, Any]]:
        """
        List all registered windows.
        
        Returns:
            List of window info dicts.
        """
        with self._lock:
            return [info.to_dict() for info in self._windows.values()]
    
    def list_aliases(self) -> List[str]:
        """Get list of all registered aliases."""
        with self._lock:
            return list(self._windows.keys())
    
    def clear(self) -> None:
        """Clear all registered windows."""
        with self._lock:
            count = len(self._windows)
            self._windows.clear()
            self._counters.clear()
            logger.info(f"🧹 Registry cleared ({count} windows removed)")
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._windows)
    
    def __contains__(self, alias: str) -> bool:
        with self._lock:
            return alias in self._windows

    # =========================================================================
    # Persistence (Ripple Fix)
    # =========================================================================

    def save_registry(self) -> bool:
        """Save registry state to JSON file."""
        try:
            data = {alias: info.to_dict() for alias, info in self._windows.items()}
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False

    def load_registry(self) -> bool:
        """Load registry state from JSON file."""
        if not REGISTRY_FILE.exists():
            return False
        
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            with self._lock:
                self._windows.clear()
                for alias, item in data.items():
                    self._windows[alias] = WindowInfo(
                        alias=item["alias"],
                        app_name=item["app"],
                        hwnd=item.get("hwnd"),
                        pid=item.get("pid"),
                        title=item.get("title", "")
                    )
                self._restore_counters()
            
            # logger.debug(f"Loaded {len(self._windows)} windows from registry")
            return True
            
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load registry (starting fresh): {e}")
            return False

    def _restore_counters(self) -> None:
        """Restore auto-increment counters based on loaded aliases."""
        self._counters.clear()
        for alias in self._windows:
            if "_" in alias:
                try:
                    name, num = alias.rsplit("_", 1)
                    count = int(num)
                    if count > self._counters.get(name, 0):
                        self._counters[name] = count
                except ValueError:
                    pass


# =============================================================================
# Module-level Singleton Accessor
# =============================================================================

_registry: Optional[WindowRegistry] = None


def get_registry() -> WindowRegistry:
    """Get or create the global WindowRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = WindowRegistry()
    return _registry


__all__ = [
    "WindowRegistry",
    "WindowInfo",
    "get_registry",
]
