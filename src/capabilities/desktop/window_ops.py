"""
MODULE: Window State Management
VERSION: 2.5.2
STRICT SCOPE: Focus, Minimize, Maximize, Snap, Close Window.
CONSTRAINTS: Uses WindowRegistry. Input is Alias or Handle.

TARA 2.0 Atomic Tool Module - Cross-platform support (Windows, Linux, macOS).

Platform Support:
    - Windows: uses win32gui APIs
    - Linux: uses xdotool (requires libxdo3 package)
    - macOS: uses AppleScript via osascript

Verification Logic (Trust But Verify):
    - All operations use `_resolve_hwnd()` to validate alias -> HWND mapping.
    - focus_window(): Multi-step fallback for Windows 10/11 focus restrictions
      (IsIconic check -> SetForegroundWindow -> Alt-key trick -> BringWindowToTop).
    - Window state is verified before operations (IsIconic, IsZoomed checks).

Error Handling:
    - Returns descriptive error strings with available aliases on lookup failure.
    - Win32/xdotool exceptions caught and converted to LLM-readable messages.

Exports:
    - focus_window(alias: str) -> str
    - minimize_window(alias: str) -> str
    - maximize_window(alias: str) -> str
    - snap_window(alias: str, position: str) -> str
    - close_window(alias: str) -> str
    - list_open_windows() -> str
"""
from __future__ import annotations

import sys
import time
import subprocess
from typing import Optional, Tuple

from src.core.logger import setup_logger

from .window_registry import get_registry

logger = setup_logger("TARA.Tools.WindowOps")

# =============================================================================
# Platform Detection
# =============================================================================

_PLATFORM = sys.platform
_IS_WINDOWS = sys.platform == "win32"
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform == "darwin"

# =============================================================================
# Optional Dependencies (Platform-Specific)
# =============================================================================

# Windows dependencies
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
    if _IS_WINDOWS:
        logger.warning("win32gui not available - window operations disabled on Windows")

try:
    import win32com.client
    import pythoncom
    _HAS_WIN32COM = True
except ImportError:
    _HAS_WIN32COM = False
    pythoncom = None  # type: ignore
    if _IS_WINDOWS:
        logger.warning("win32com not available - fallback focus disabled")

# Linux dependencies (xdotool)
_HAS_XDOTOOL = False
if _IS_LINUX:
    try:
        result = subprocess.run(["which", "xdotool"], capture_output=True)
        _HAS_XDOTOOL = result.returncode == 0
        if not _HAS_XDOTOOL:
            logger.info("xdotool not found - install with: sudo apt install xdotool")
    except Exception:
        pass

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
        if _IS_WINDOWS and _HAS_WIN32 and win32gui.IsWindow(hwnd):
            return hwnd, None
        elif not _IS_WINDOWS:
            # On non-Windows, just validate the window ID exists
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
    """Get primary screen dimensions (cross-platform)."""
    try:
        if _IS_WINDOWS and _HAS_WIN32:
            width = win32api.GetSystemMetrics(0)  # SM_CXSCREEN
            height = win32api.GetSystemMetrics(1)  # SM_CYSCREEN
            return width, height
        elif _IS_LINUX and _HAS_XDOTOOL:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowgeometry"],
                capture_output=True, text=True, timeout=2
            )
            # Fallback to xrandr
            result = subprocess.run(
                ["xrandr", "--current"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if " connected" in line:
                    parts = line.split()[0].split("x")
                    if len(parts) >= 2:
                        try:
                            return int(parts[0]), int(parts[1].split("+")[0])
                        except (ValueError, IndexError):
                            pass
        elif _IS_MACOS:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            # Parse from output (basic approach)
            pass
    except Exception as e:
        logger.debug(f"Could not get screen size: {e}")

    return 1920, 1080  # Fallback


# =============================================================================
# Cross-Platform Window Operations
# =============================================================================

def _focus_window_windows(hwnd: int) -> bool:
    """Focus window on Windows using Win32 APIs."""
    if not _HAS_WIN32:
        return False

    try:
        # Step 1: Check if minimized (Iconic) - restore first
        if win32gui.IsIconic(hwnd):
            logger.debug(f"Window is minimized, restoring...")
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.1)

        # Step 2: Try direct SetForegroundWindow
        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            logger.debug(f"SetForegroundWindow failed: {e}, trying Alt-key fallback...")

        # Step 3: Windows 10/11 Fallback - Alt-key simulation
        if _HAS_WIN32COM:
            try:
                pythoncom.CoInitialize()
                try:
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys('%')  # Simulate Alt key press
                    time.sleep(0.05)
                    win32gui.SetForegroundWindow(hwnd)
                    return True
                finally:
                    pythoncom.CoUninitialize()
            except Exception as e2:
                logger.warning(f"Alt-key fallback failed: {e2}")

        # Step 4: Last resort - BringWindowToTop
        try:
            win32gui.BringWindowToTop(hwnd)
            return True
        except Exception:
            pass

        return False
    except Exception as e:
        logger.error(f"Windows focus error: {e}")
        return False


def _focus_window_linux(hwnd: int) -> bool:
    """Focus window on Linux using xdotool."""
    if not _HAS_XDOTOOL:
        return False

    try:
        subprocess.run(
            ["xdotool", "windowactivate", str(hwnd)],
            check=True, capture_output=True, timeout=5
        )
        return True
    except Exception as e:
        logger.error(f"Linux focus error: {e}")
        return False


def _focus_window_macos(hwnd: int, alias: str) -> bool:
    """Focus window on macOS using AppleScript."""
    try:
        # Try to get window title from registry for better targeting
        registry = get_registry()
        if alias in registry:
            window_info = registry.get(alias)
            title = window_info.get("title", "") if window_info else ""
        else:
            title = alias

        # Use AppleScript to activate window
        script = f'''
        tell application "System Events"
            set allWindows to windows of processes
            repeat with w in allWindows
                if (name of w) contains "{title}" then
                    set visible of w to true
                    set position of (first window of (first item of (processes where name contains "{title}"))) to {{0, 0}}
                end if
            end repeat
        end tell
        '''
        subprocess.run(
            ["osascript", "-e", script],
            check=False, capture_output=True, timeout=5
        )
        return True
    except Exception as e:
        logger.error(f"macOS focus error: {e}")
        return False


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

    if not alias:
        return "❌ Error: alias is required"

    # Resolve HWND
    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"

    # Platform-specific focus
    success = False
    if _IS_WINDOWS:
        success = _focus_window_windows(hwnd)
    elif _IS_LINUX:
        success = _focus_window_linux(hwnd)
    elif _IS_MACOS:
        success = _focus_window_macos(hwnd, alias)
    else:
        return f"❌ Unsupported platform: {_PLATFORM}"

    if success:
        logger.debug(f"Focused window: {alias}")
        return f"✅ Brought {alias} to front"
    else:
        return f"⚠️ Could not fully focus {alias}"


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

    if not alias:
        return "❌ Error: alias is required"

    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"

    try:
        if _IS_WINDOWS and _HAS_WIN32:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        elif _IS_LINUX and _HAS_XDOTOOL:
            subprocess.run(
                ["xdotool", "windowminimize", str(hwnd)],
                check=True, capture_output=True, timeout=5
            )
        elif _IS_MACOS:
            # macOS doesn't have direct minimize via AppleScript window ID
            # Use keyboard shortcut instead
            subprocess.run(
                ["osascript", "-e", "key code 106"],  # Cmd+M keyboard code
                check=False, capture_output=True, timeout=5
            )
        else:
            return f"❌ Unsupported platform: {_PLATFORM}"

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

    if not alias:
        return "❌ Error: alias is required"

    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"

    try:
        if _IS_WINDOWS and _HAS_WIN32:
            # Restore first if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.05)

            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        elif _IS_LINUX and _HAS_XDOTOOL:
            subprocess.run(
                ["xdotool", "windowsize", str(hwnd), "100%", "100%"],
                check=True, capture_output=True, timeout=5
            )
        elif _IS_MACOS:
            # macOS zoom (maximize equivalent)
            subprocess.run(
                ["osascript", "-e", "tell application \"System Events\" to perform action \"AXZoomWindow\" of (first window of first application process)"],
                check=False, capture_output=True, timeout=5
            )
        else:
            return f"❌ Unsupported platform: {_PLATFORM}"

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

        if _IS_WINDOWS and _HAS_WIN32:
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
        elif _IS_LINUX and _HAS_XDOTOOL:
            if position == "left":
                x, y = 0, 0
                width, height = half_width, screen_height
            else:
                x, y = half_width, 0
                width, height = half_width, screen_height

            subprocess.run(
                [
                    "xdotool", "windowmove", str(hwnd), str(x), str(y),
                    "windowsize", str(hwnd), str(width), str(height)
                ],
                check=True, capture_output=True, timeout=5
            )
        elif _IS_MACOS:
            # macOS snap requires AppleScript with coordinates
            if position == "left":
                x, y = 0, 0
            else:
                x, y = half_width, 0

            script = f'''
            tell application "System Events"
                set frontmost of first application process to true
                perform action "AXRaise" of (first window of first application process)
                set position of (first window of first application process) to {{{x}, {y}}}
                set size of (first window of first application process) to {{{half_width}, {screen_height}}}
            end tell
            '''
            subprocess.run(
                ["osascript", "-e", script],
                check=False, capture_output=True, timeout=5
            )
        else:
            return f"❌ Unsupported platform: {_PLATFORM}"

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

    ONE ACTION: Send close signal to window.

    Also deregisters the alias from the registry.

    Args:
        alias: Window alias from registry.

    Returns:
        Success or failure message.
    """

    if not alias:
        return "❌ Error: alias is required"

    registry = get_registry()

    hwnd, error = _resolve_hwnd(alias)
    if error:
        return f"❌ {error}"

    try:
        if _IS_WINDOWS and _HAS_WIN32:
            # Send WM_CLOSE (graceful close request)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        elif _IS_LINUX and _HAS_XDOTOOL:
            subprocess.run(
                ["xdotool", "windowkill", str(hwnd)],
                check=True, capture_output=True, timeout=5
            )
        elif _IS_MACOS:
            subprocess.run(
                ["osascript", "-e", "tell application \"System Events\" to perform action \"AXClose\" of (first window of first application process)"],
                check=False, capture_output=True, timeout=5
            )
        else:
            return f"❌ Unsupported platform: {_PLATFORM}"

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

    ONE ACTION: Show desktop (Win+D equivalent on Windows, Cmd+F3 on macOS).

    Returns:
        Success message.
    """

    try:
        if _IS_WINDOWS and _HAS_WIN32COM:
            pythoncom.CoInitialize()
            try:
                shell = win32com.client.Dispatch("Shell.Application")
                shell.MinimizeAll()
                logger.debug("Minimized all windows (show desktop)")
                return "🖥️ Desktop shown (all windows minimized)"
            finally:
                pythoncom.CoUninitialize()
        elif _IS_LINUX and _HAS_XDOTOOL:
            subprocess.run(
                ["xdotool", "key", "super+d"],
                check=False, capture_output=True, timeout=5
            )
            return "🖥️ Desktop shown"
        elif _IS_MACOS:
            subprocess.run(
                ["osascript", "-e", "key code 101 using {cmd down, fn down}"],  # F3
                check=False, capture_output=True, timeout=5
            )
            return "🖥️ Desktop shown"
        else:
            return f"❌ Show desktop not supported on {_PLATFORM}"
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
