"""
MODULE: Screen Operations (The Camera Lens)
STRICT SCOPE: Raw pixel capture only. No OCR/intelligence.
CONSTRAINTS: The 'Eyes' of the agent. Read-only operations.

TARA 2.0 Atomic Tool Module.

Exports:
    - take_screenshot(filename: str = None) -> str
    - get_screen_resolution() -> str
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.core.logger import setup_logger

logger = setup_logger("TARA.Tools.ScreenOps")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
except ImportError:
    _HAS_PYAUTOGUI = False
    pyautogui = None  # type: ignore
    logger.warning("pyautogui not available - screenshot disabled")

# Screenshot output directory
SCREENSHOT_DIR = Path("data/screenshots")


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
        "📸 Screenshot saved: C:/Users/.../screenshot_20260111_221500.png"
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: uv add pyautogui"
    
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
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
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
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    try:
        width, height = pyautogui.size()
        return f"🖥️ Screen Resolution: Width: {width}, Height: {height}"
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
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    try:
        x, y = pyautogui.position()
        return f"🖱️ Mouse Position: ({x}, {y})"
    except Exception as e:
        return f"❌ Failed to get position: {e}"


__all__ = [
    "take_screenshot",
    "get_screen_resolution",
    "get_mouse_position",
]
