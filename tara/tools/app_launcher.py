"""
MODULE: Process Management
STRICT SCOPE: Start, Kill, List Processes.
CONSTRAINTS: No Window manipulation. No generic 'app_control'.
RETURNS: PIDs and Process Objects only.

TARA 2.0 Atomic Tool Module.

Exports:
    - launch_app(app_name: str) -> str
    - kill_app(app_name: str) -> str
    - list_processes(filter_name: str = None) -> str
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from core.logger import setup_logger
from core.config import get_settings

from .registry import get_registry

logger = setup_logger("TARA.Tools.AppLauncher")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import pygetwindow as gw
    _HAS_PYGETWINDOW = True
except ImportError:
    _HAS_PYGETWINDOW = False
    gw = None  # type: ignore

try:
    import win32gui
    import win32process
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False
    win32gui = None  # type: ignore
    win32process = None  # type: ignore


# =============================================================================
# App Configuration Loader
# =============================================================================

def _load_app_config() -> dict:
    """Load app configuration from external JSON file."""
    config_path = Path(__file__).parent.parent / "units" / "config" / "apps.json"
    
    if not config_path.exists():
        return {
            "system_apps": ["notepad", "calc", "cmd", "explorer"],
            "custom_aliases": {}
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


_APP_CONFIG = _load_app_config()
SYSTEM_APPS: List[str] = _APP_CONFIG.get("system_apps", [])
CUSTOM_ALIASES: dict = _APP_CONFIG.get("custom_aliases", {})


# =============================================================================
# PID-to-HWND Helpers
# =============================================================================

def _get_hwnds_from_pid(target_pid: int) -> List[int]:
    """Get all window handles belonging to a specific PID."""
    if not _HAS_WIN32:
        return []
    
    hwnds = []
    
    def enum_callback(hwnd, _):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid:
                hwnds.append(hwnd)
        except Exception:
            pass
        return True
    
    try:
        win32gui.EnumWindows(enum_callback, None)
    except Exception as e:
        logger.debug(f"EnumWindows error: {e}")
    
    return hwnds


def _is_window_visible(hwnd: int) -> bool:
    """Check if a window handle is visible."""
    if not _HAS_WIN32:
        return False
    try:
        return win32gui.IsWindowVisible(hwnd)
    except Exception:
        return False


def _get_window_title(hwnd: int) -> str:
    """Get window title from HWND."""
    if not _HAS_WIN32:
        return ""
    try:
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return ""


# =============================================================================
# Atomic Tool: launch_app
# =============================================================================

def launch_app(app_name: str) -> str:
    """
    Launch an application by name with verification.
    
    ONE ACTION: Start a process and verify its window appears.
    
    Args:
        app_name: Application name (e.g., "notepad", "chrome", "brave").
        
    Returns:
        Success message with alias, or error message.
        
    Example:
        >>> launch_app("notepad")
        "✅ Launched: notepad [alias: notepad_1]"
    """
    if not app_name:
        return "❌ Error: app_name is required"
    
    app_lower = app_name.lower().strip()
    registry = get_registry()
    settings = get_settings()
    
    # Determine launch command
    if app_lower in CUSTOM_ALIASES:
        launch_cmd = CUSTOM_ALIASES[app_lower]
        launch_method = "alias"
    elif app_lower in SYSTEM_APPS or app_lower.endswith(".exe"):
        launch_cmd = app_lower
        launch_method = "system"
    else:
        launch_cmd = app_lower
        launch_method = "start"
    
    # Snapshot existing windows (to detect NEW ones)
    pre_launch_windows: Set[Tuple[str, int]] = set()
    if _HAS_PYGETWINDOW:
        try:
            pre_launch_windows = {
                (w.title, w._hWnd) for w in gw.getAllWindows()
                if app_lower in w.title.lower()
            }
        except Exception:
            pass
    
    # Launch the application
    launched_pid: Optional[int] = None
    launched = False
    
    try:
        logger.info(f"🚀 Launching '{launch_cmd}'...")
        
        # Direct execution for system apps (get real PID)
        if app_lower in SYSTEM_APPS or app_lower.endswith(".exe"):
            try:
                exe_name = f"{app_lower}.exe" if not app_lower.endswith(".exe") else app_lower
                proc = subprocess.Popen(
                    [exe_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False
                )
                launched_pid = proc.pid
                launched = True
                launch_method = "direct"
                logger.debug(f"Direct launch PID: {launched_pid}")
            except FileNotFoundError:
                logger.debug("Direct launch failed, falling back to shell...")
        
        # Fallback: Shell execution
        if not launched:
            subprocess.Popen(
                f'start "" "{launch_cmd}"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            launched = True
            launch_method = "shell"
            
    except Exception as e1:
        logger.warning(f"Subprocess failed: {e1}, trying os.startfile...")
        try:
            os.startfile(launch_cmd)
            launched = True
            launch_method = "startfile"
        except FileNotFoundError:
            return f"❌ Application not found: '{app_name}'"
        except OSError as e2:
            return f"❌ Failed to launch '{app_name}': {e2}"
    
    if not launched:
        return f"❌ Launch failed: {app_name}"
    
    # Verification loop: Wait for visible window
    max_retries = settings.LAUNCH_MAX_RETRIES
    poll_interval = settings.LAUNCH_POLL_INTERVAL
    visible_hwnd: Optional[int] = None
    window_title = ""
    
    logger.debug(f"⏳ Verifying launch (PID={launched_pid}, retries={max_retries})...")
    
    for attempt in range(1, max_retries + 1):
        # Method 1: PID-based verification
        if launched_pid and _HAS_WIN32:
            hwnds = _get_hwnds_from_pid(launched_pid)
            for hwnd in hwnds:
                if _is_window_visible(hwnd):
                    visible_hwnd = hwnd
                    window_title = _get_window_title(hwnd)
                    logger.debug(f"✅ PID {launched_pid} -> HWND {hwnd} (attempt {attempt})")
                    break
            if visible_hwnd:
                break
        
        # Method 2: New window detection
        if not visible_hwnd and _HAS_PYGETWINDOW:
            try:
                current_windows = [
                    w for w in gw.getAllWindows()
                    if app_lower in w.title.lower() and w.title.strip()
                ]
                for win in current_windows:
                    win_key = (win.title, win._hWnd)
                    if win_key not in pre_launch_windows and win.visible:
                        visible_hwnd = win._hWnd
                        window_title = win.title
                        logger.debug(f"✅ NEW window detected: '{win.title}'")
                        break
            except Exception as e:
                logger.debug(f"Window check error: {e}")
        
        if visible_hwnd:
            break
        
        time.sleep(poll_interval)
    
    # Final result
    if visible_hwnd:
        # Register in window registry
        alias = registry.register(
            app_name=app_name,
            hwnd=visible_hwnd,
            pid=launched_pid,
            title=window_title,
        )
        return f"✅ Launched: {app_name} [alias: {alias}]"
    else:
        # Kill zombie process if we have PID
        if launched_pid:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(launched_pid)],
                    capture_output=True,
                    timeout=5
                )
                logger.warning(f"💀 Killed zombie PID {launched_pid}")
            except Exception:
                pass
        
        return f"❌ {app_name} started but window never appeared"


# =============================================================================
# Atomic Tool: kill_app
# =============================================================================

def kill_app(app_name: str) -> str:
    """
    Kill an application by name.
    
    ONE ACTION: Terminate process(es) matching the app name.
    
    Args:
        app_name: Application name or alias (e.g., "notepad", "notepad_1").
        
    Returns:
        Success or failure message.
        
    Example:
        >>> kill_app("notepad")
        "💀 Killed: notepad"
    """
    if not app_name:
        return "❌ Error: app_name is required"
    
    registry = get_registry()
    app_lower = app_name.lower().strip()
    
    # Check if it's a registry alias
    if app_lower in registry:
        info = registry.get(app_lower)
        if info and info.pid:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(info.pid)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                registry.deregister(app_lower)
                return f"💀 Killed: {app_lower} (PID {info.pid})"
            except Exception as e:
                return f"❌ Failed to kill PID {info.pid}: {e}"
        # Fallback: deregister anyway
        registry.deregister(app_lower)
    
    # Kill by exe name
    clean_name = app_lower.replace(".exe", "")
    exe_name = f"{clean_name}.exe"
    
    # Check if process is running
    try:
        check = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if exe_name.lower() not in check.stdout.lower():
            return f"⚠️ {clean_name} is not running"
    except Exception:
        pass
    
    # Kill the process
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", exe_name, "/T"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Verify it's dead
        time.sleep(0.5)
        verify = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if exe_name.lower() not in verify.stdout.lower():
            # Deregister any aliases for this app
            for alias in list(registry.list_aliases()):
                info = registry.get(alias)
                if info and info.app_name.lower() == app_lower:
                    registry.deregister(alias)
            
            return f"💀 Killed: {clean_name}"
        else:
            return f"❌ {clean_name} refused to die (try running as admin)"
            
    except subprocess.TimeoutExpired:
        return f"❌ Kill command timed out for {clean_name}"
    except Exception as e:
        return f"❌ Failed to kill {clean_name}: {e}"


# =============================================================================
# Atomic Tool: list_processes
# =============================================================================

def list_processes(filter_name: Optional[str] = None) -> str:
    """
    List running processes, optionally filtered by name.
    
    ONE ACTION: Read process list from OS.
    
    Args:
        filter_name: Optional filter string for process names.
        
    Returns:
        Formatted list of processes.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        lines = result.stdout.strip().split("\n")
        processes = []
        
        for line in lines[:50]:  # Limit output
            parts = line.replace('"', '').split(",")
            if len(parts) >= 2:
                name, pid = parts[0], parts[1]
                if filter_name is None or filter_name.lower() in name.lower():
                    processes.append(f"{name} (PID: {pid})")
        
        if not processes:
            return "No matching processes found."
        
        return f"Running processes ({len(processes)}):\n" + "\n".join(processes[:20])
        
    except Exception as e:
        return f"❌ Failed to list processes: {e}"


__all__ = [
    "launch_app",
    "kill_app",
    "list_processes",
]
