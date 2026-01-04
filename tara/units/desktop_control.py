"""TARA Desktop Control Unit - Universal Desktop Controller for N.I.A.

The "Hands" of the assistant - manipulates windows, apps, browser, and files.

Architecture:
    - "Smart Paths": Resolves aliases (Desktop, Downloads) and nested paths
    - "Launch & Wait": Solves race conditions by waiting for window to exist
    - "System-First": Prioritizes exact system executables over fuzzy matching
    - "Junk-Safe": Uses **kwargs to ignore hallucinated LLM arguments
    - "Forced Search": Bypasses browser default engine, forces Google

Features:
    - App Control (launch with validation, type, hotkey)
    - Browser Control (search via Google, tabs, navigation)
    - Window Manager (focus, snap, minimize, maximize, close)
    - File Manager (open, reveal, create, delete with path aliases)

Dependencies:
    pip install pyautogui pygetwindow AppOpener send2trash
"""
from __future__ import annotations

import os
import time
import subprocess
import urllib.parse
from typing import Optional

from tara.protocols import tara_tool

# =============================================================================
# Optional Dependencies (Graceful Imports)
# =============================================================================

try:
    import pyautogui
    pyautogui.FAILSAFE = True   # Move mouse to corner to abort
    pyautogui.PAUSE = 0.5       # Stability delay between actions
    _HAS_PYAUTOGUI = True
except ImportError:
    _HAS_PYAUTOGUI = False
    pyautogui = None  # type: ignore

try:
    import pygetwindow as gw
    _HAS_PYGETWINDOW = True
except ImportError:
    _HAS_PYGETWINDOW = False
    gw = None  # type: ignore

try:
    from AppOpener import open as app_open, close as app_close
    _HAS_APPOPENER = True
except ImportError:
    _HAS_APPOPENER = False

try:
    from send2trash import send2trash
    _HAS_SEND2TRASH = True
except ImportError:
    _HAS_SEND2TRASH = False


# =============================================================================
# Constants
# =============================================================================

# Windows system apps that should be launched directly via subprocess
# This prevents fuzzy matching (e.g., "notepad" -> "Notepad++")
SYSTEM_APPS = [
    "notepad",
    "calc",
    "mspaint",
    "cmd",
    "powershell",
    "explorer",
    "taskmgr",
    "control",
    "regedit",
    "snippingtool",
    "charmap",
    "magnify",
    "osk",
    "write",
    "mstsc",
    "winver",
    "msconfig",
]

# Path aliases mapping user-friendly names to actual paths
def _get_path_aliases() -> dict:
    """Get path aliases with user's home directory."""
    home = os.path.expanduser("~")
    return {
        "desktop": os.path.join(home, "Desktop"),
        "downloads": os.path.join(home, "Downloads"),
        "documents": os.path.join(home, "Documents"),
        "music": os.path.join(home, "Music"),
        "pictures": os.path.join(home, "Pictures"),
        "videos": os.path.join(home, "Videos"),
        "home": home,
        "~": home,
    }


# =============================================================================
# Helper: Smart Path Resolution (The Nested Fix)
# =============================================================================

def _resolve_path(path_str: str) -> str:
    """Resolve path aliases and nested paths to absolute paths.
    
    Handles:
        - "Desktop" -> C:/Users/user/Desktop
        - "Downloads/NewFolder" -> C:/Users/user/Downloads/NewFolder
        - "~/Documents/file.txt" -> C:/Users/user/Documents/file.txt
        - Absolute paths passed through unchanged
    
    Args:
        path_str: Path string (may contain aliases).
        
    Returns:
        Absolute resolved path.
    """
    if not path_str:
        return os.path.expanduser("~/Desktop")  # Default to Desktop
    
    # Normalize path separators
    path_str = os.path.normpath(path_str)
    
    # Split into parts
    parts = path_str.split(os.sep)
    
    # Get aliases
    aliases = _get_path_aliases()
    
    # Check if first part is an alias
    first_part_lower = parts[0].lower()
    
    if first_part_lower in aliases:
        # Replace alias with actual path
        base_path = aliases[first_part_lower]
        if len(parts) > 1:
            # Join with remaining path parts
            return os.path.join(base_path, *parts[1:])
        return base_path
    
    # Check if it's an absolute path
    if os.path.isabs(path_str):
        return path_str
    
    # Expand ~ if present
    if path_str.startswith("~"):
        return os.path.expanduser(path_str)
    
    # Default: treat as relative to Desktop
    return os.path.join(aliases["desktop"], path_str)


# =============================================================================
# Helper: Window Focus (Multi-Strategy Matching)
# =============================================================================

def _focus_window(title_keyword: str) -> bool:
    """Find and focus a window using multi-strategy matching.
    
    Strategies (in order):
        1. Exact match (case-insensitive)
        2. First word match (e.g., "Brave" matches "Brave Browser")
        3. Contains match (keyword anywhere in title)
    
    Args:
        title_keyword: Window title or keyword to search for.
        
    Returns:
        True if window was found and focused, False otherwise.
    """
    if not _HAS_PYGETWINDOW:
        return False
    
    try:
        # Get all windows
        all_windows = gw.getAllWindows()
        
        if not all_windows:
            return False
        
        keyword_lower = title_keyword.lower().strip()
        target_window = None
        
        # Strategy 1: Exact match (case-insensitive)
        for win in all_windows:
            if win.title and win.title.lower() == keyword_lower:
                target_window = win
                break
        
        # Strategy 2: First word match
        if not target_window:
            for win in all_windows:
                if win.title:
                    first_word = win.title.split()[0].lower() if win.title.split() else ""
                    if first_word == keyword_lower:
                        target_window = win
                        break
        
        # Strategy 3: Contains match (keyword anywhere in title)
        if not target_window:
            for win in all_windows:
                if win.title and keyword_lower in win.title.lower():
                    target_window = win
                    break
        
        if not target_window:
            return False
        
        # Restore if minimized
        if target_window.isMinimized:
            target_window.restore()
            time.sleep(0.2)
        
        # Bring to front
        target_window.activate()
        
        return True
        
    except Exception:
        return False


# =============================================================================
# Helper: Wait & Focus (Race Condition Fix)
# =============================================================================

def _wait_and_focus(app_name: str, timeout: float = 5.0) -> bool:
    """Wait for a window to appear and focus it (solves race conditions).
    
    Args:
        app_name: Window title keyword to search for.
        timeout: Maximum seconds to wait for window.
        
    Returns:
        True if window was found and focused, False if timeout.
    """
    if not _HAS_PYGETWINDOW:
        time.sleep(2.0)  # Fallback: just wait
        return True
    
    start_time = time.time()
    poll_interval = 0.3
    
    while (time.time() - start_time) < timeout:
        if _focus_window(app_name):
            time.sleep(0.5)  # Window settle time
            return True
        time.sleep(poll_interval)
    
    return False


# =============================================================================
# TOOL 1: App Control (Launch & Wait + Junk-Safe)
# =============================================================================

@tara_tool(
    name="app_control",
    category="desktop",
    description="Launch/kill applications or send keystrokes. Actions: 'launch'/'open', 'kill'/'close', 'type' (sends text), 'hotkey' (e.g., 'ctrl+s'). System apps launch directly; others use fuzzy match."
)
def app_control(action: str, app_name: str, keys: str = None, **kwargs) -> str:
    """Control applications - launch, kill, or send keystrokes.
    
    Args:
        action: 'launch'/'open', 'kill'/'close', 'type', 'hotkey'.
        app_name: Name of the application.
        keys: Text to type or hotkey combo.
        **kwargs: Ignored (catches LLM hallucinated args like runasadmin).
    """
    action = action.lower().strip()
    app_lower = app_name.lower().strip()
    
    try:
        # =================================================================
        # LAUNCH / OPEN
        # =================================================================
        if action in ("launch", "open"):
            launched = False
            launch_method = ""
            
            # System App Check (Exact Match)
            if app_lower in SYSTEM_APPS or app_lower.endswith(".exe"):
                try:
                    subprocess.Popen(
                        app_lower,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    launched = True
                    launch_method = "system"
                except Exception as e:
                    return f"❌ Failed to launch system app '{app_name}': {e}"
            
            # Fuzzy Fallback (AppOpener)
            else:
                if not _HAS_APPOPENER:
                    return "Error: AppOpener not installed. Run: pip install AppOpener"
                try:
                    app_open(app_name, match_closest=True, throw_error=True)
                    launched = True
                    launch_method = "fuzzy"
                except Exception as e:
                    return f"❌ Could not find or launch '{app_name}': {e}"
            
            # CRITICAL: Wait for window to exist
            if launched:
                if _wait_and_focus(app_name, timeout=5.0):
                    return f"✅ Launched and focused: {app_name} ({launch_method})"
                else:
                    return f"⚠️ Launched '{app_name}' but window focus failed"
            
            return f"❌ Launch failed: {app_name}"
        
        # =================================================================
        # KILL / CLOSE
        # =================================================================
        if action in ("kill", "close"):
            if not _HAS_APPOPENER:
                return "Error: AppOpener not installed."
            try:
                app_close(app_name, match_closest=True, throw_error=True)
                return f"💀 Killed: {app_name}"
            except Exception as e:
                return f"❌ Could not close '{app_name}': {e}"
        
        # =================================================================
        # TYPE
        # =================================================================
        if action == "type":
            if not _HAS_PYAUTOGUI:
                return "Error: pyautogui not installed."
            if not keys:
                return "Error: 'type' requires 'keys' parameter."
            
            focused = _focus_window(app_name) or _wait_and_focus(app_name, timeout=3.0)
            
            if focused:
                time.sleep(0.2)
                pyautogui.write(keys, interval=0.02)
                display = f"{keys[:50]}..." if len(keys) > 50 else keys
                return f"⌨️ Typed '{display}' in {app_name}"
            
            return f"❌ Could not focus '{app_name}'"
        
        # =================================================================
        # HOTKEY
        # =================================================================
        if action == "hotkey":
            if not _HAS_PYAUTOGUI:
                return "Error: pyautogui not installed."
            if not keys:
                return "Error: 'hotkey' requires 'keys' parameter."
            
            focused = _focus_window(app_name) or _wait_and_focus(app_name, timeout=3.0)
            
            if focused:
                time.sleep(0.2)
                key_parts = [k.strip().lower() for k in keys.split("+")]
                pyautogui.hotkey(*key_parts)
                return f"⌨️ Sent hotkey '{keys}' to {app_name}"
            
            return f"❌ Could not focus '{app_name}'"
        
        return f"Error: Unknown action '{action}'."
        
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TOOL 2: Browser Control (Forced Google + Junk-Safe)
# =============================================================================

@tara_tool(
    name="browser_general",
    category="browser",
    description="Universal browser actions. Actions: 'open', 'search' (forces Google), 'new_tab', 'close_tab', 'reopen_tab', 'refresh', 'back', 'forward', 'history', 'downloads', 'address_bar', 'fullscreen'."
)
def browser_general(action: str, url: str = None, query: str = None, **kwargs) -> str:
    """Control browser with universal shortcuts.
    
    Args:
        action: Browser action.
        url: URL for 'open'.
        query: Search query for 'search'.
        **kwargs: Ignored.
    """
    if not _HAS_PYAUTOGUI and action not in ("open", "search"):
        return "Error: pyautogui not installed."
    
    action = action.lower().strip()
    
    try:
        # SEARCH (Forced Google)
        if action == "search":
            if not query:
                return "Error: 'search' requires 'query'."
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            os.startfile(search_url)
            return f"🔍 Searched Google: {query}"
        
        # OPEN URL
        if action == "open":
            if not url:
                return "Error: 'open' requires 'url'."
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            os.startfile(url)
            return f"🌐 Opened: {url}"
        
        # TAB MANAGEMENT
        if action == "new_tab":
            pyautogui.hotkey("ctrl", "t")
            return "➕ New tab opened"
        
        if action == "close_tab":
            pyautogui.hotkey("ctrl", "w")
            return "❌ Tab closed"
        
        if action == "reopen_tab":
            pyautogui.hotkey("ctrl", "shift", "t")
            return "♻️ Tab reopened"
        
        # NAVIGATION
        if action == "refresh":
            pyautogui.hotkey("ctrl", "r")
            return "🔄 Refreshed"
        
        if action == "back":
            pyautogui.hotkey("alt", "left")
            return "⬅️ Back"
        
        if action == "forward":
            pyautogui.hotkey("alt", "right")
            return "➡️ Forward"
        
        if action == "history":
            pyautogui.hotkey("ctrl", "h")
            return "📜 History opened"
        
        if action == "downloads":
            pyautogui.hotkey("ctrl", "j")
            return "📥 Downloads opened"
        
        if action == "address_bar":
            pyautogui.hotkey("alt", "d")
            return "🔗 Address bar focused"
        
        if action == "fullscreen":
            pyautogui.press("f11")
            return "🖥️ Fullscreen toggled"
        
        if action == "zoom_in":
            pyautogui.hotkey("ctrl", "plus")
            return "🔍 Zoomed in"
        
        if action == "zoom_out":
            pyautogui.hotkey("ctrl", "minus")
            return "🔍 Zoomed out"
        
        if action == "zoom_reset":
            pyautogui.hotkey("ctrl", "0")
            return "🔍 Zoom reset"
        
        return f"Error: Unknown action '{action}'."
        
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TOOL 3: Window Manager (Junk-Safe)
# =============================================================================

@tara_tool(
    name="window_manager",
    category="desktop",
    description="Manage window focus and position. Actions: 'minimize_all', 'show_desktop', 'switch', 'task_view', 'focus', 'close', 'snap_left', 'snap_right', 'maximize', 'minimize'. Target is window name."
)
def window_manager(action: str, target: str = None, **kwargs) -> str:
    """Manage windows on the desktop.
    
    Args:
        action: Action to perform.
        target: Window title for targeted actions.
        **kwargs: Ignored.
    """
    if not _HAS_PYAUTOGUI:
        return "Error: pyautogui not installed."
    
    action = action.lower().strip()
    
    try:
        # GLOBAL ACTIONS
        if action in ("minimize_all", "show_desktop"):
            pyautogui.hotkey("win", "d")
            return "🖥️ Desktop shown"
        
        if action == "switch":
            pyautogui.hotkey("alt", "tab")
            return "🔄 Switched window"
        
        if action == "task_view":
            pyautogui.hotkey("win", "tab")
            return "📋 Task view opened"
        
        # TARGETED ACTIONS
        if not target:
            return f"Error: '{action}' requires target window name."
        
        if action == "focus":
            if _focus_window(target):
                return f"✅ Focused: {target}"
            return f"❌ Not found: {target}"
        
        if action in ("close", "close_active"):
            if _focus_window(target):
                time.sleep(0.2)
                pyautogui.hotkey("alt", "F4")
                return f"🚪 Closed: {target}"
            return f"❌ Not found: {target}"
        
        if action == "snap_left":
            if _focus_window(target):
                pyautogui.hotkey("win", "left")
                return f"⬅️ Snapped left: {target}"
            return f"❌ Not found: {target}"
        
        if action == "snap_right":
            if _focus_window(target):
                pyautogui.hotkey("win", "right")
                return f"➡️ Snapped right: {target}"
            return f"❌ Not found: {target}"
        
        if action == "maximize":
            if _focus_window(target):
                pyautogui.hotkey("win", "up")
                time.sleep(0.1)
                pyautogui.hotkey("win", "up")
                return f"⬆️ Maximized: {target}"
            return f"❌ Not found: {target}"
        
        if action == "minimize":
            if _focus_window(target):
                pyautogui.hotkey("win", "down")
                time.sleep(0.1)
                pyautogui.hotkey("win", "down")
                return f"⬇️ Minimized: {target}"
            return f"❌ Not found: {target}"
        
        return f"Error: Unknown action '{action}'."
        
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TOOL 4: File Manager (Smart Paths + Junk-Safe)
# =============================================================================

@tara_tool(
    name="file_manager",
    category="files",
    description="Manage files/folders with smart paths. Supports aliases: Desktop, Downloads, Documents, Pictures, Music, Videos, ~. Actions: 'open', 'reveal', 'create_folder', 'delete'."
)
def file_manager(action: str, path: str, **kwargs) -> str:
    """Manage files and folders with smart path resolution.
    
    Args:
        action: 'open', 'reveal', 'create_folder', 'delete'.
        path: Path (supports aliases like 'Desktop/NewFolder').
        **kwargs: Ignored.
    """
    if not path:
        return "Error: 'path' is required."
    
    # SMART PATH RESOLUTION
    path = _resolve_path(path)
    action = action.lower().strip()
    
    try:
        # OPEN
        if action == "open":
            if not os.path.exists(path):
                return f"❌ Not found: {path}"
            os.startfile(path)
            return f"📂 Opened: {path}"
        
        # REVEAL
        if action == "reveal":
            if not os.path.exists(path):
                return f"❌ Not found: {path}"
            subprocess.run(["explorer", "/select,", path], check=True)
            return f"📁 Revealed: {path}"
        
        # CREATE FOLDER
        if action == "create_folder":
            if os.path.exists(path):
                return f"⚠️ Already exists: {path}"
            os.makedirs(path, exist_ok=True)
            return f"📁 Created: {path}"
        
        # DELETE (Safe - Recycle Bin)
        if action == "delete":
            if not os.path.exists(path):
                return f"❌ Not found: {path}"
            if not _HAS_SEND2TRASH:
                return "Error: send2trash not installed."
            send2trash(path)
            return f"🗑️ Deleted: {path}"
        
        return f"Error: Unknown action '{action}'."
        
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"
