"""TARA Plugin System - File System Watcher.

VERSION: 3.0.0

This module provides hot-reload capabilities by monitoring the plugins/
directory for file changes using the watchdog library.

Features:
    - Automatic plugin reload on file create/modify
    - Automatic plugin unload on file delete
    - Debouncing to prevent duplicate events (0.5s delay)
    - Daemon thread (doesn't block shutdown)

Usage:
    from tara.plugin_system.watcher import start_plugin_watcher, stop_plugin_watcher
    
    observer = start_plugin_watcher(plugins_dir)
    # ... application runs ...
    stop_plugin_watcher(observer)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from core.logger import setup_logger

if TYPE_CHECKING:
    from watchdog.observers import Observer

logger = setup_logger("TARA.Watcher")

import json

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "tara" / "plugins.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_PLUGIN_CONFIG = _load_config()

# Debounce delay in seconds (prevents double-reload on save)
DEBOUNCE_DELAY = _PLUGIN_CONFIG.get("DEBOUNCE_DELAY", 0.5)


class PluginEventHandler:
    """File system event handler for plugin hot-reloading.
    
    Monitors the plugins directory for .py file changes and triggers
    appropriate loader methods with debouncing to prevent duplicate events.
    
    Attributes:
        loader: The PluginLoader instance to notify of changes.
        _debounce_timers: Active debounce timers keyed by file path.
    """
    
    def __init__(self) -> None:
        """Initialize the event handler."""
        from tara.plugin_system.loader import get_plugin_loader
        
        self._loader = get_plugin_loader()
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._timer_lock = threading.Lock()
    
    def _is_valid_plugin_file(self, path: str) -> bool:
        """Check if path is a valid plugin file (not __pycache__, etc)."""
        p = Path(path)
        
        # Must be a .py file
        if p.suffix != ".py":
            return False
        
        # Skip __pycache__, __init__, etc.
        if "__pycache__" in path or p.stem.startswith("__"):
            return False
        
        return True
    
    def _cancel_timer(self, path: str) -> None:
        """Cancel any existing debounce timer for a path."""
        with self._timer_lock:
            if path in self._debounce_timers:
                self._debounce_timers[path].cancel()
                del self._debounce_timers[path]
    
    def _schedule_reload(self, path: str) -> None:
        """Schedule a debounced reload for a file."""
        self._cancel_timer(path)
        
        def do_reload():
            with self._timer_lock:
                if path in self._debounce_timers:
                    del self._debounce_timers[path]
            
            file_path = Path(path)
            logger.debug(f"Debounce triggered for: {file_path.name}")
            self._loader.reload_plugin_by_path(file_path)
        
        timer = threading.Timer(DEBOUNCE_DELAY, do_reload)
        timer.daemon = True
        
        with self._timer_lock:
            self._debounce_timers[path] = timer
        
        timer.start()
    
    def _schedule_unload(self, path: str) -> None:
        """Schedule a debounced unload for a deleted file."""
        self._cancel_timer(path)
        
        def do_unload():
            with self._timer_lock:
                if path in self._debounce_timers:
                    del self._debounce_timers[path]
            
            file_path = Path(path)
            logger.debug(f"Unload triggered for: {file_path.name}")
            self._loader.unload_plugin(file_path)
        
        timer = threading.Timer(DEBOUNCE_DELAY, do_unload)
        timer.daemon = True
        
        with self._timer_lock:
            self._debounce_timers[path] = timer
        
        timer.start()
    
    # =========================================================================
    # Watchdog Event Callbacks
    # =========================================================================
    
    def dispatch(self, event) -> None:
        """Dispatch file system events to appropriate handlers.
        
        This is the main entry point called by watchdog's Observer.
        """
        # Ignore directory events
        if getattr(event, 'is_directory', False):
            return
        
        src_path = getattr(event, 'src_path', None)
        if not src_path:
            return
        
        if not self._is_valid_plugin_file(src_path):
            return
        
        event_type = getattr(event, 'event_type', None)
        
        if event_type == 'created':
            logger.debug(f"File created: {Path(src_path).name}")
            self._schedule_reload(src_path)
        
        elif event_type == 'modified':
            logger.debug(f"File modified: {Path(src_path).name}")
            self._schedule_reload(src_path)
        
        elif event_type == 'deleted':
            logger.debug(f"File deleted: {Path(src_path).name}")
            self._schedule_unload(src_path)
    
    def cleanup(self) -> None:
        """Cancel all pending timers on shutdown."""
        with self._timer_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()


# =============================================================================
# Factory Functions
# =============================================================================

_observer: Optional["Observer"] = None


def start_plugin_watcher(plugins_dir: Optional[Path] = None) -> "Observer":
    """Start the file system watcher for plugin hot-reloading.
    
    Creates an Observer that monitors the plugins directory for changes.
    The observer runs in a daemon thread and will not block shutdown.
    
    Args:
        plugins_dir: Path to watch. If None, uses loader's default.
        
    Returns:
        The started Observer instance.
        
    Example:
        observer = start_plugin_watcher()
        # ... later ...
        stop_plugin_watcher(observer)
    """
    global _observer
    
    try:
        from watchdog.observers import Observer
    except ImportError:
        logger.error("❌ watchdog library not installed. Run: pip install watchdog")
        raise ImportError("watchdog library required for hot-reload. Install with: pip install watchdog")
    
    # Get plugins directory
    if plugins_dir is None:
        from tara.plugin_system.loader import get_plugin_loader
        plugins_dir = get_plugin_loader().plugins_dir
    
    plugins_dir = Path(plugins_dir)
    
    if not plugins_dir.exists():
        logger.warning(f"Plugins directory does not exist: {plugins_dir}")
        plugins_dir.mkdir(parents=True, exist_ok=True)
    
    # Create handler and observer
    handler = PluginEventHandler()
    observer = Observer()
    
    # Schedule watching (recursive=False - only watch top level)
    observer.schedule(handler, str(plugins_dir), recursive=False)
    
    # Start as daemon thread
    observer.daemon = True
    observer.start()
    
    _observer = observer
    
    logger.debug(f"🔄 Plugin watcher started (monitoring: {plugins_dir})")
    
    return observer


def stop_plugin_watcher(observer: Optional["Observer"] = None) -> None:
    """Stop the file system watcher gracefully.
    
    Args:
        observer: The Observer to stop. If None, stops the global observer.
    """
    global _observer
    
    target = observer or _observer
    
    if target is None:
        logger.debug("No plugin watcher to stop")
        return
    
    try:
        target.stop()
        target.join(timeout=2.0)
        logger.info("🔄 Plugin watcher stopped")
    except Exception as e:
        logger.warning(f"Error stopping plugin watcher: {e}")
    
    if target is _observer:
        _observer = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PluginEventHandler",
    "start_plugin_watcher",
    "stop_plugin_watcher",
]
