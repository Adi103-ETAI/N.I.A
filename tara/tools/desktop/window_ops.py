"""
MODULE: Window State Management
VERSION: 2.5.2
STRICT SCOPE: Focus, Minimize, Maximize, Snap, Close Window.
CONSTRAINTS: Uses WindowRegistry. Input is Alias or Handle.

TARA 2.0 Atomic Tool Module.

Verification Logic (Trust But Verify):
    - All operations use `_resolve_hwnd()` to validate alias -> HWND mapping.
    - focus_window(): Multi-step fallback for Windows 10/11 focus restrictions
      (IsIconic check -> SetForegroundWindow -> Alt-key trick -> BringWindowToTop).
    - Window state is verified before operations (IsIconic, IsZoomed checks).

Error Handling:
    - Returns descriptive error strings with available aliases on lookup failure.
    - Win32 exceptions caught and converted to LLM-readable messages.

Exports:
    - focus_window(alias: str) -> str
    - minimize_window(alias: str) -> str
    - maximize_window(alias: str) -> str
    - snap_window(alias: str, position: str) -> str
    - close_window(alias: str) -> str
    - list_open_windows() -> str
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

from core.logger import setup_logger

from .window_manager import get_registry

logger = setup_logger("TARA.Tools.WindowOps")

# =============================================================================
# Optional Dependencies (Win32 APIs)
# =============================================================================

try:
    import win32gui
    import win32con
    import win32api
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False
    win32gui = None  # type: ignore
    win32con = None  # type: ignore
    win32api = None  # type: ignore
    logger.warning("win32gui not available - window operations disabled")

try:
    import win32com.client
    import pythoncom
    _HAS_WIN32COM = True
except ImportError:
    _HAS_WIN32COM = False
    pythoncom = None  # type: ignore
    logger.warning("win32com not available - fallback focus disabled")


# =============================================================================
# Helper: Get HWND from Alias or Direct
# =============================================================================

def _resolve_hwnd(alias_or_hwnd: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Resolve an alias to HWND, or validate a direct HWND.
    
    Args:
        alias_or_hwnd: Window alias (e.g., "notepad_1") or HWND as string.
        
    Returns:
        Tuple of (hwnd, error_message). One will be None.
    """
    registry = get_registry()
    
    # Check if it's a registered alias
    if alias_or_hwnd in registry:
        hwnd = registry.get_handle(alias_or_hwnd)
        if hwnd:
            return hwnd, None
        else:
            return None, f"Alias '{alias_or_hwnd}' has no HWND registered"
    
    # Check if it's a direct HWND number
    try:
        hwnd = int(alias_or_hwnd)
        if _HAS_WIN32 and win32gui.IsWindow(hwnd):
            return hwnd, None
        else:
            return None, f"HWND {hwnd} is not a valid window"
    except ValueError:
        pass
    
    # Not found
    available = registry.list_aliases()
    avail_str = ", ".join(available[:5]) if available else "none"
    return None, f"Unknown alias '{alias_or_hwnd}'. Available: {avail_str}"


def _get_screen_size() -> Tuple[int, int]:
    """Get primary screen dimensions."""
    if _HAS_WIN32:
        width = win32api.GetSystemMetrics(0)  # SM_CXSCREEN
        height = win32api.GetSystemMetrics(1)  # SM_CYSCREEN
        return width, height
    return 1920, 1080  # Fallback


# =============================================================================
# Atomic Tool: focus_window (Smart Focus)
# =============================================================================

def focus_window(alias: str) -> str:
    """
    Bring a window to the foreground with Smart Focus.
    
    ONE ACTION: Focus a window by its alias.
    
    Handles edge cases:
    - Minimized windows are restored first
    - Uses Alt-key trick for Windows 10/11 focus restrictions
    
    Args:
        alias: Window alias from registry (e.g., "notepad_1").
        
    Returns:
        Success or failure message.
        
    Example:
        >>> focus_window("notepad_1")
        "✅ Brought notepad_1 to front"
    """
    if not _HAS_WIN32:
        return "❌ win32gui not available"
    
    if not alias:
        return "❌ Error: alias is required"
    
    # Resolve HWND
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"
    
    try:
        # Step 1: Check if minimized (Iconic) - restore first
        if win32gui.IsIconic(hwnd):
            logger.debug(f"Window {alias} is minimized, restoring...")
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.1)  # Brief settle time
        
        # Step 2: Try direct SetForegroundWindow
        try:
            win32gui.SetForegroundWindow(hwnd)
            logger.debug(f"SetForegroundWindow succeeded for {alias}")
            return f"✅ Brought {alias} to front"
        except Exception as e:
            logger.debug(f"SetForegroundWindow failed: {e}, trying Alt-key fallback...")
        
        # Step 3: Windows 10/11 Fallback - Alt-key simulation
        # Windows blocks background apps from stealing focus unless
        # the user "recently" pressed a key
        if _HAS_WIN32COM:
            try:
                # COM thread initialization for non-main threads
                pythoncom.CoInitialize()
                try:
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys('%')  # Simulate Alt key press
                    time.sleep(0.05)
                    win32gui.SetForegroundWindow(hwnd)
                    logger.debug(f"Alt-key fallback succeeded for {alias}")
                    return f"✅ Brought {alias} to front"
                finally:
                    pythoncom.CoUninitialize()
            except Exception as e2:
                logger.warning(f"Alt-key fallback failed: {e2}")
        
        # Step 4: Last resort - BringWindowToTop
        try:
            win32gui.BringWindowToTop(hwnd)
            return f"✅ Brought {alias} to front (partial)"
        except Exception:
            pass
        
        return f"⚠️ Could not fully focus {alias} (Windows restriction)"
        
    except Exception as e:
        logger.error(f"Focus error: {e}")
        return f"❌ Focus failed: {e}"


# =============================================================================
# Atomic Tool: minimize_window
# =============================================================================

def minimize_window(alias: str) -> str:
    """
    Minimize a window to the taskbar.
    
    ONE ACTION: Minimize a window by its alias.
    
    Args:
        alias: Window alias from registry.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_WIN32:
        return "❌ win32gui not available"
    
    if not alias:
        return "❌ Error: alias is required"
    
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"
    
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        logger.debug(f"Minimized window: {alias}")
        return f"⬇️ Minimized: {alias}"
    except Exception as e:
        return f"❌ Minimize failed: {e}"


# =============================================================================
# Atomic Tool: maximize_window
# =============================================================================

def maximize_window(alias: str) -> str:
    """
    Maximize a window to fill the screen.
    
    ONE ACTION: Maximize a window by its alias.
    
    Args:
        alias: Window alias from registry.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_WIN32:
        return "❌ win32gui not available"
    
    if not alias:
        return "❌ Error: alias is required"
    
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"
    
    try:
        # Restore first if minimized
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.05)
        
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        logger.debug(f"Maximized window: {alias}")
        return f"⬆️ Maximized: {alias}"
    except Exception as e:
        return f"❌ Maximize failed: {e}"


# =============================================================================
# Atomic Tool: snap_window
# =============================================================================

def snap_window(alias: str, position: str = "left") -> str:
    """
    Snap a window to left or right half of screen.
    
    ONE ACTION: Snap window to screen edge.
    
    Args:
        alias: Window alias from registry.
        position: "left" or "right" (default: "left").
        
    Returns:
        Success or failure message.
    """
    if not _HAS_WIN32:
        return "❌ win32gui not available"
    
    if not alias:
        return "❌ Error: alias is required"
    
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"
    
    position = position.lower().strip()
    if position not in ("left", "right"):
        return f"❌ Invalid position '{position}'. Use 'left' or 'right'"
    
    try:
        # Get screen dimensions
        screen_width, screen_height = _get_screen_size()
        half_width = screen_width // 2
        
        # Restore if minimized/maximized
        if win32gui.IsIconic(hwnd) or win32gui.IsZoomed(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.05)
        
        # Calculate position
        if position == "left":
            x, y = 0, 0
            width, height = half_width, screen_height
        else:  # right
            x, y = half_width, 0
            width, height = half_width, screen_height
        
        # Move and resize
        win32gui.MoveWindow(hwnd, x, y, width, height, True)
        
        direction = "⬅️" if position == "left" else "➡️"
        logger.debug(f"Snapped {alias} to {position}")
        return f"{direction} Snapped {alias} to {position}"
        
    except Exception as e:
        return f"❌ Snap failed: {e}"


# =============================================================================
# Atomic Tool: close_window
# =============================================================================

def close_window(alias: str) -> str:
    """
    Close a window gracefully.
    
    ONE ACTION: Send WM_CLOSE to window.
    
    Also deregisters the alias from the registry.
    
    Args:
        alias: Window alias from registry.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_WIN32:
        return "❌ win32gui not available"
    
    if not alias:
        return "❌ Error: alias is required"
    
    registry = get_registry()
    
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"
    
    try:
        # Send WM_CLOSE (graceful close request)
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        
        # Deregister from registry
        if alias in registry:
            registry.deregister(alias)
        
        logger.debug(f"Closed window: {alias}")
        return f"🚪 Closed: {alias}"
        
    except Exception as e:
        return f"❌ Close failed: {e}"


# =============================================================================
# Atomic Tool: list_open_windows
# =============================================================================

def list_open_windows() -> str:
    """
    List all windows tracked in the registry.
    
    ONE ACTION: Return registry snapshot.
    
    Returns:
        Formatted list of tracked windows.
    """
    registry = get_registry()
    windows = registry.list_windows()
    
    if not windows:
        return "📋 No windows are currently tracked. Launch apps using launch_app first."
    
    lines = [f"📋 Tracked Windows ({len(windows)}):"]
    for w in windows:
        hwnd_str = f"HWND={w.get('hwnd')}" if w.get('hwnd') else "no handle"
        lines.append(f"  • {w['alias']}: \"{w.get('title', 'Unknown')}\" ({hwnd_str})")
    
    return "\n".join(lines)


# =============================================================================
# Atomic Tool: show_desktop
# =============================================================================

def show_desktop() -> str:
    """
    Minimize all windows to show the desktop.
    
    ONE ACTION: Show desktop (Win+D equivalent).
    
    Returns:
        Success message.
    """
    if not _HAS_WIN32COM:
        return "❌ win32com not available"
    
    try:
        # COM thread initialization for non-main threads
        pythoncom.CoInitialize()
        try:
            shell = win32com.client.Dispatch("Shell.Application")
            shell.MinimizeAll()
            logger.debug("Minimized all windows (show desktop)")
            return "🖥️ Desktop shown (all windows minimized)"
        finally:
            pythoncom.CoUninitialize()
    except Exception as e:
        return f"❌ Show desktop failed: {e}"


__all__ = [
    "focus_window",
    "minimize_window",
    "maximize_window",
    "snap_window",
    "close_window",
    "list_open_windows",
    "show_desktop",
]
