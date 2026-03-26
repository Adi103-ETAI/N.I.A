"""
MODULE: Screen Operations (The Camera Lens)
STRICT SCOPE: Raw pixel capture only. No OCR/intelligence.
CONSTRAINTS: The 'Eyes' of the agent. Read-only operations.

TARA 2.0 Atomic Tool Module.
Cross-platform support: Windows, Linux, macOS

Exports:
    - take_screenshot(filename: str = None) -> str
    - get_screen_resolution() -> str
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.core.logger import setup_logger

logger = setup_logger("TARA.Tools.ScreenOps")

# =============================================================================
# Cross-Platform Screenshot Backends
# =============================================================================

_HAS_PYAUTOGUI = False
_HAS_PIL = False
_HAS_XLIB = False
_BACKEND = None

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
    logger.debug("pyautogui available")
except ImportError:
    logger.debug("pyautogui not available")

try:
    from PIL import ImageGrab
    _HAS_PIL = True
    logger.debug("PIL/Pillow available for screenshots")
except ImportError:
    logger.debug("PIL/Pillow not available")

# Try Xlib for Linux
if sys.platform.startswith("linux"):
    try:
        import Xlib.display
        _HAS_XLIB = True
        logger.debug("Xlib available for Linux")
    except ImportError:
        logger.debug("Xlib not available for Linux")

# Select best available backend
if sys.platform == "win32":
    _BACKEND = "pyautogui" if _HAS_PYAUTOGUI else ("pil" if _HAS_PIL else None)
elif sys.platform == "darwin":  # macOS
    _BACKEND = "pil" if _HAS_PIL else ("pyautogui" if _HAS_PYAUTOGUI else None)
elif sys.platform.startswith("linux"):
    _BACKEND = "xlib" if _HAS_XLIB else ("pyautogui" if _HAS_PYAUTOGUI else ("pil" if _HAS_PIL else None))
else:
    _BACKEND = "pyautogui" if _HAS_PYAUTOGUI else ("pil" if _HAS_PIL else None)

if _BACKEND:
    logger.info(f"Screen capture backend: {_BACKEND}")
else:
    logger.warning("No screenshot backend available")

# Screenshot output directory
SCREENSHOT_DIR = Path("data/screenshots")


# =============================================================================
# Cross-Platform Helpers
# =============================================================================

def _screenshot_pyautogui(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
    """Take screenshot using pyautogui."""
    try:
        if region:
            return pyautogui.screenshot(region=region)
        else:
            return pyautogui.screenshot()
    except Exception as e:
        logger.error(f"pyautogui screenshot failed: {e}")
        return None


def _screenshot_pil(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
    """Take screenshot using PIL (works on Windows, macOS, limited Linux)."""
    try:
        if region:
            x, y, w, h = region
            return ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            return ImageGrab.grab()
    except Exception as e:
        logger.error(f"PIL screenshot failed: {e}")
        return None


def _screenshot_xlib(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
    """Take screenshot using Xlib on Linux."""
    try:
        from PIL import Image
        display = Xlib.display.Display()
        screen = display.screen()
        window = screen.root

        if region:
            x, y, w, h = region
            geom = window.get_geometry()
            raw = window.get_image(x, y, w, h, Xlib.X.ZPixmap, ~0)
            image = Image.frombytes("RGB", (w, h), raw.data, "raw", "BGRX")
        else:
            geom = window.get_geometry()
            raw = window.get_image(0, 0, geom.width, geom.height, Xlib.X.ZPixmap, ~0)
            image = Image.frombytes("RGB", (geom.width, geom.height), raw.data, "raw", "BGRX")

        return image
    except Exception as e:
        logger.error(f"Xlib screenshot failed: {e}")
        return None


def _screenshot_linux_fallback(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
    """Fallback screenshot method for Linux using gnome-screenshot or import tool."""
    try:
        from PIL import Image
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        # Try different screenshot tools available on Linux
        tools = [
            ["gnome-screenshot", "-f", tmp_path],
            ["import", "-window", "root", tmp_path],
            ["scrot", tmp_path],
        ]

        for tool in tools:
            try:
                subprocess.run(tool, check=True, capture_output=True, timeout=5)
                if Path(tmp_path).exists():
                    return Image.open(tmp_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None
    except Exception as e:
        logger.error(f"Linux fallback screenshot failed: {e}")
        return None


def _get_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
    """Get screenshot using best available backend."""
    if _BACKEND == "pyautogui":
        return _screenshot_pyautogui(region)
    elif _BACKEND == "pil":
        return _screenshot_pil(region)
    elif _BACKEND == "xlib":
        img = _screenshot_xlib(region)
        if img is None and sys.platform.startswith("linux"):
            img = _screenshot_linux_fallback(region)
        return img
    return None


# =============================================================================
# Atomic Tool: take_screenshot
# =============================================================================

def take_screenshot(filename: Optional[str] = None, region: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Take a screenshot of the screen.

    ONE ACTION: Capture screen pixels to file.

    Args:
        filename: Optional filename (auto-generated if None).
        region: Optional (x, y, width, height) to capture region only.

    Returns:
        Absolute path to saved screenshot.

    Example:
        >>> take_screenshot()
        "📸 Screenshot saved: /home/user/.../screenshot_20260111_221500.png"
    """
    if not _BACKEND:
        return "❌ No screenshot backend available. Install: pip install pillow"

    try:
        # Ensure directory exists
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"

        # Ensure .png extension
        if not filename.lower().endswith(".png"):
            filename += ".png"

        # Full path
        filepath = SCREENSHOT_DIR / filename

        # Take screenshot
        screenshot = _get_screenshot(region)

        if screenshot is None:
            return "❌ Screenshot capture failed - all backends unavailable"

        # Save
        screenshot.save(str(filepath))

        abs_path = filepath.resolve()
        logger.info(f"Screenshot saved: {abs_path}")
        return f"📸 Screenshot saved: {abs_path}"

    except Exception as e:
        return f"❌ Screenshot failed: {e}"


# =============================================================================
# Atomic Tool: get_screen_resolution
# =============================================================================

def get_screen_resolution() -> str:
    """
    Get the screen resolution.

    ONE ACTION: Read screen dimensions.

    Returns:
        Formatted resolution string.

    Example:
        >>> get_screen_resolution()
        "🖥️ Screen Resolution: Width: 1920, Height: 1080"
    """
    try:
        if sys.platform == "win32" and _HAS_PYAUTOGUI:
            width, height = pyautogui.size()
            return f"🖥️ Screen Resolution: Width: {width}, Height: {height}"
        elif sys.platform == "darwin":
            # macOS
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            # Parse resolution from output (basic approach)
            if "Resolution:" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "Resolution:" in line:
                        return f"🖥️ Screen Resolution: {line.strip()}"
            return "🖥️ Could not determine screen resolution"
        elif sys.platform.startswith("linux"):
            # Linux - try different methods
            try:
                # Try xrandr first
                result = subprocess.run(
                    ["xrandr", "--current"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if " connected" in line or " disconnected" in line:
                            continue
                        if "x" in line.split()[0]:
                            res = line.split()[0]
                            return f"🖥️ Screen Resolution: {res}"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Try pyautogui as fallback
            if _HAS_PYAUTOGUI:
                width, height = pyautogui.size()
                return f"🖥️ Screen Resolution: Width: {width}, Height: {height}"

            return "🖥️ Could not determine screen resolution"
        else:
            return "🖥️ Unsupported platform for resolution detection"
    except Exception as e:
        return f"❌ Failed to get resolution: {e}"


# =============================================================================
# Atomic Tool: get_mouse_position
# =============================================================================

def get_mouse_position() -> str:
    """
    Get current mouse cursor position.

    ONE ACTION: Read cursor coordinates.

    Returns:
        Formatted position string.
    """
    try:
        if _HAS_PYAUTOGUI:
            x, y = pyautogui.position()
            return f"🖱️ Mouse Position: ({x}, {y})"
        else:
            return "❌ Mouse position tracking not available (requires pyautogui)"
    except Exception as e:
        return f"❌ Failed to get position: {e}"


__all__ = [
    "take_screenshot",
    "get_screen_resolution",
    "get_mouse_position",
]

