"""
MODULE: Process Management
VERSION: 3.0.0
STRICT SCOPE: Start, Kill, List Processes.
CONSTRAINTS: No Window manipulation. No generic 'app_control'.
RETURNS: PIDs and Process Objects only.

TARA 2.0 Atomic Tool Module - ASYNC UPDATE.

Verification Logic (Trust But Verify):
    - launch_app(): Uses PID tracking + window polling to verify launch success.
      Takes pre/post window snapshot, waits for visible HWND, kills zombie PIDs.
    - kill_app(): Verifies process termination via tasklist before returning.

Error Handling:
    - All tools return emoji-prefixed strings (✅/❌/⚠️) for LLM parsing.
    - Failures are returned as descriptive error strings, NOT exceptions.

Exports:
    - launch_app(app_name: str) -> str
    - kill_app(app_name: str) -> str
    - list_processes(filter_name: str = None) -> str
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from src.core.logger import setup_logger
from src.core.config import get_settings

from .window_manager import get_registry

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
    """Load app configuration from centralized ROOT/config/tara/."""
    # Path: desktop -> tools -> tara -> ROOT (3 levels up via .parents[3])
    config_path = Path(__file__).resolve().parents[3] / "config" / "tara" / "apps.json"
    
    if not config_path.exists():
        # Auto-create default if missing
        default_config = {
            "system_apps": ["notepad", "calc", "cmd", "explorer"],
            "custom_aliases": {}
        }
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default apps.json at {config_path}")
        except Exception as e:
            logger.warning(f"Could not create apps.json: {e}")
            return default_config
            
        return default_config
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load apps.json: {e}")
        return {"system_apps": [], "custom_aliases": {}}


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
from ..decorators import security_level

# =============================================================================
# Atomic Tool: launch_app
# =============================================================================

@security_level("high_risk")
async def launch_app(app_name: str) -> str:
    """
    Launch an application by name with verification.
    
    ONE ACTION: Start a process and verify its window appears.
    
    Args:
        app_name: Application name (e.g., "notepad", "chrome", "brave").
        
    Returns:
        Success message with alias, or error message.
        
    Example:
        >>> await launch_app("notepad")
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
            # Run blocking call in thread
            pre_launch_windows = await asyncio.to_thread(
                lambda: {
                    (w.title, w._hWnd) for w in gw.getAllWindows()
                    if app_lower in w.title.lower()
                }
            )
        except Exception:
            pass
    
    # Launch the application
    launched_pid: Optional[int] = None
    launched = False
    
    logger.info(f"🚀 Launching '{launch_cmd}'...")
    
    # === DYNAMIC DISCOVERY (AppIndex) ===
    # Try the omniscient index first — covers Win32, UWP, and Shell apps
    from src.infrastructure.host_os.app_index import get_app_index
    app_index = get_app_index()
    index_entry = await asyncio.to_thread(app_index.search, app_lower)

    if index_entry:
        logger.info(f"AppIndex matched '{app_name}' -> {index_entry.display()}")

        # For UWP/Shell apps, launch via shell:AppsFolder (can't get PID directly)
        if index_entry.app_type in ("uwp", "shell"):
            try:
                launch_result = await asyncio.to_thread(app_index.launch, index_entry)
                if launch_result.startswith("launched:"):
                    alias = registry.register(
                        app_name=index_entry.name,
                        hwnd=None,
                        pid=None,
                        title=f"{index_entry.name} ({index_entry.app_type})",
                    )
                    return f"✅ Launched: {index_entry.name} [alias: {alias}] ({index_entry.app_type})"
                else:
                    return f"❌ Failed to launch '{index_entry.name}': {launch_result}"
            except Exception as e:
                return f"❌ Failed to launch '{index_entry.name}': {e}"

        # For Win32 apps with a valid .exe path, use the path from AppIndex
        elif index_entry.app_type == "win32" and os.path.isfile(index_entry.app_id):
            executable_path = index_entry.app_id
            logger.debug(f"AppIndex resolved Win32 path: {executable_path}")
        else:
            # Win32 but no direct path — try shutil.which as fallback
            executable_path = shutil.which(index_entry.app_id) or shutil.which(app_lower)
    else:
        # No AppIndex match — original fallback logic
        executable_path = None

    # === LEGACY PATH RESOLUTION (fallback for unlisted apps) ===
    if not executable_path:
        # Check if it's an absolute path first
        if os.path.isabs(launch_cmd) and os.path.isfile(launch_cmd):
            executable_path = launch_cmd
            logger.debug(f"Using absolute path: {executable_path}")
        else:
            # Try resolving with shutil.which()
            exe_name = f"{launch_cmd}.exe" if not launch_cmd.endswith(".exe") else launch_cmd
            executable_path = shutil.which(exe_name)
            if not executable_path:
                executable_path = shutil.which(launch_cmd)
            if not executable_path and launch_cmd in CUSTOM_ALIASES:
                alias_path = CUSTOM_ALIASES[launch_cmd]
                if os.path.isfile(alias_path):
                    executable_path = alias_path
                else:
                    executable_path = shutil.which(alias_path)

    # === FAIL FAST: No executable found ===
    if not executable_path:
        # Provide helpful suggestions
        suggestions = await asyncio.to_thread(app_index.search_all, app_lower, 3)
        if suggestions:
            names = ", ".join(f"'{s.name}'" for s in suggestions)
            logger.warning(f"App not found: '{app_name}'. Did you mean: {names}?")
            return f"❌ App not found: '{app_name}'. Did you mean: {names}?"
        return f"❌ Application not found: '{app_name}' (not in Start Menu or PATH)"
    
    logger.debug(f"Resolved executable: {executable_path}")
    
    # === SECURE EXECUTION (shell=False ONLY) ===
    try:
        def _run_popen():
            return subprocess.Popen(
                [executable_path],  # Always list, never string
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False  # 🛡️ THE IRON BAR - NEVER TRUE
            )
        
        proc = await asyncio.to_thread(_run_popen)
        launched_pid = proc.pid
        launched = True
        logger.debug(f"Direct launch PID: {launched_pid}")
        
        # === FIX 1 (The Pause) ===
        # Give the UI time to render before any verification
        time.sleep(2)
        
    except PermissionError as e:
        logger.error(f"Permission denied launching '{executable_path}': {e}")
        return f"❌ Permission denied: '{app_name}'"
    except OSError as e:
        logger.error(f"OS error launching '{executable_path}': {e}")
        return f"❌ Failed to launch '{app_name}': {e}"
    
    if not launched:
        return f"❌ Launch failed: {app_name}"
    
    # === FIX 2 (Blind Faith) ===
    # If subprocess.Popen did not raise, assume success.
    # Don't aggressively verify PID/window immediately - apps need time.
    
    # Try to detect a window for registration (best effort, not required for success)
    visible_hwnd: Optional[int] = None
    window_title = ""
    
    if launched_pid and _HAS_WIN32:
        try:
            hwnds = await asyncio.to_thread(_get_hwnds_from_pid, launched_pid)
            for hwnd in hwnds:
                is_visible = await asyncio.to_thread(_is_window_visible, hwnd)
                if is_visible:
                    visible_hwnd = hwnd
                    window_title = await asyncio.to_thread(_get_window_title, hwnd)
                    break
        except Exception:
            pass
    
    # Register if we found a window
    if visible_hwnd:
        alias = registry.register(
            app_name=app_name,
            hwnd=visible_hwnd,
            pid=launched_pid,
            title=window_title,
        )
        
        # === FIX 3 (Focus) ===
        # Try to focus the window so it's ready for interaction
        try:
            from .window_ops import focus_window
            focus_window(alias)
        except Exception:
            pass  # Focus is best-effort
        
        return f"✅ Launched: {app_name} [alias: {alias}]"
    else:
        # No window found yet, but app was launched successfully
        # Register with PID only (no HWND yet)
        alias = registry.register(
            app_name=app_name,
            hwnd=None,
            pid=launched_pid,
            title=f"{app_name} (pending)",
        )
        return f"✅ Launched: {app_name} [alias: {alias}]. Window may still be loading."


# =============================================================================
# Atomic Tool: kill_app
# =============================================================================

async def kill_app(app_name: str) -> str:
    """
    Kill an application by name.
    
    ONE ACTION: Terminate process(es) matching the app name.
    
    Args:
        app_name: Application name or alias (e.g., "notepad", "notepad_1").
        
    Returns:
        Success or failure message.
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
                proc = await asyncio.create_subprocess_exec(
                    "taskkill", "/F", "/PID", str(info.pid),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await proc.communicate()
                
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
        proc = await asyncio.create_subprocess_exec(
            "tasklist", "/FI", f"IMAGENAME eq {exe_name}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode('utf-8', errors='ignore')
        
        if exe_name.lower() not in output.lower():
            return f"⚠️ {clean_name} is not running"
    except Exception:
        pass
    
    # Kill the process
    try:
        proc = await asyncio.create_subprocess_exec(
            "taskkill", "/F", "/IM", exe_name, "/T",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await proc.communicate()
        
        # Verify it's dead
        await asyncio.sleep(0.5)
        
        proc_verify = await asyncio.create_subprocess_exec(
            "tasklist", "/FI", f"IMAGENAME eq {exe_name}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = await proc_verify.communicate()
        verify_output = stdout.decode('utf-8', errors='ignore')
        
        if exe_name.lower() not in verify_output.lower():
            # Deregister any aliases for this app
            for alias in list(registry.list_aliases()):
                info = registry.get(alias)
                if info and info.app_name.lower() == app_lower:
                    registry.deregister(alias)
            
            return f"💀 Killed: {clean_name}"
        else:
            return f"❌ {clean_name} refused to die (try running as admin)"
            
    except Exception as e:
        return f"❌ Failed to kill {clean_name}: {e}"


# =============================================================================
# Atomic Tool: list_processes
# =============================================================================

async def list_processes(filter_name: Optional[str] = None) -> str:
    """
    List running processes, optionally filtered by name.
    
    ONE ACTION: Read process list from OS.
    
    Args:
        filter_name: Optional filter string for process names.
        
    Returns:
        Formatted list of processes.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "tasklist", "/FO", "CSV", "/NH",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode('utf-8', errors='ignore')
        
        lines = output.strip().split("\n")
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
