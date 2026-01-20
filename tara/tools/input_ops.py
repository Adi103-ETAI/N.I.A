"""
MODULE: Input Operations (The Blind Hands - Layer 3)
STRICT SCOPE: Mouse/Keyboard control using Coordinates (Fallback).
CONSTRAINTS: Coordinate-based ONLY. Use uia_ops for semantic interaction.

TARA 2.0 Atomic Tool Module.

This is the fallback layer when UI Automation fails.
Uses raw screen coordinates without element awareness.

Exports:
    - mouse_click(x, y, button, double) -> str
    - mouse_drag(start_x, start_y, end_x, end_y) -> str
    - mouse_scroll(clicks) -> str
    - keyboard_type(text, interval) -> str
    - keyboard_hotkey(*keys) -> str
    - keyboard_press(key) -> str
"""
from __future__ import annotations

import time
from typing import Tuple

from core.logger import setup_logger

logger = setup_logger("TARA.Tools.InputOps")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
    
    # SAFETY: Enable failsafe (move mouse to corner to abort)
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1  # Small pause between actions
    
except ImportError:
    _HAS_PYAUTOGUI = False
    pyautogui = None  # type: ignore
    logger.warning("pyautogui not available - input operations disabled")


# =============================================================================
# Atomic Tool: mouse_click
# =============================================================================

def mouse_click(
    x: int,
    y: int,
    button: str = "left",
    double: bool = False,
) -> str:
    """
    Click at screen coordinates.
    
    ONE ACTION: Mouse click at (x, y).
    
    Args:
        x: X coordinate on screen.
        y: Y coordinate on screen.
        button: "left", "right", or "middle" (default: "left").
        double: If True, double-click (default: False).
        
    Returns:
        Success or failure message.
        
    Example:
        >>> mouse_click(500, 300)
        "✅ Clicked at (500, 300)"
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    try:
        clicks = 2 if double else 1
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        
        click_type = "Double-clicked" if double else "Clicked"
        button_str = f" ({button})" if button != "left" else ""
        
        logger.debug(f"{click_type}{button_str} at ({x}, {y})")
        return f"✅ {click_type}{button_str} at ({x}, {y})"
        
    except Exception as e:
        return f"❌ Click failed: {e}"


# =============================================================================
# Atomic Tool: mouse_drag
# =============================================================================

def mouse_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
) -> str:
    """
    Drag mouse from one point to another.
    
    ONE ACTION: Click and drag from start to end.
    
    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        end_x: Ending X coordinate.
        end_y: Ending Y coordinate.
        duration: Drag duration in seconds (default: 0.5).
        
    Returns:
        Success or failure message.
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    try:
        # Move to start position
        pyautogui.moveTo(start_x, start_y)
        time.sleep(0.05)
        
        # Drag to end position
        pyautogui.drag(
            end_x - start_x,
            end_y - start_y,
            duration=duration,
            button="left"
        )
        
        logger.debug(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return f"✅ Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        
    except Exception as e:
        return f"❌ Drag failed: {e}"


# =============================================================================
# Atomic Tool: mouse_scroll
# =============================================================================

def mouse_scroll(clicks: int, x: int = None, y: int = None) -> str:
    """
    Scroll the mouse wheel.
    
    ONE ACTION: Scroll up (positive) or down (negative).
    
    Args:
        clicks: Number of scroll clicks (positive=up, negative=down).
        x: Optional X coordinate to scroll at.
        y: Optional Y coordinate to scroll at.
        
    Returns:
        Success message.
        
    Example:
        >>> mouse_scroll(-3)  # Scroll down 3 clicks
        "📜 Scrolled down by 3 clicks"
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    try:
        pyautogui.scroll(clicks, x=x, y=y)
        
        direction = "up" if clicks > 0 else "down"
        logger.debug(f"Scrolled {direction} by {abs(clicks)} clicks")
        return f"📜 Scrolled {direction} by {abs(clicks)} clicks"
        
    except Exception as e:
        return f"❌ Scroll failed: {e}"


# =============================================================================
# Atomic Tool: keyboard_type
# =============================================================================

def keyboard_type(text: str, interval: float = 0.05) -> str:
    """
    Type text using keyboard simulation.
    
    ONE ACTION: Type characters sequentially.
    
    Args:
        text: Text to type.
        interval: Delay between keystrokes in seconds.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not text:
        return "❌ No text provided"
    
    try:
        pyautogui.write(text, interval=interval)
        
        display_text = text[:30] + "..." if len(text) > 30 else text
        logger.debug(f"Typed: {display_text}")
        return f"⌨️ Typed: '{display_text}'"
        
    except Exception as e:
        return f"❌ Type failed: {e}"


# =============================================================================
# Atomic Tool: keyboard_hotkey
# =============================================================================

def keyboard_hotkey(*keys: str) -> str:
    """
    Press a keyboard shortcut combination.
    
    ONE ACTION: Press multiple keys simultaneously.
    
    Args:
        *keys: Keys to press together (e.g., "ctrl", "c").
        
    Returns:
        Success or failure message.
        
    Example:
        >>> keyboard_hotkey("ctrl", "s")
        "⌨️ Pressed: Ctrl+S"
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not keys:
        return "❌ No keys provided"
    
    try:
        pyautogui.hotkey(*keys)
        
        combo = "+".join(k.capitalize() for k in keys)
        logger.debug(f"Hotkey: {combo}")
        return f"⌨️ Pressed: {combo}"
        
    except Exception as e:
        return f"❌ Hotkey failed: {e}"


# =============================================================================
# Atomic Tool: keyboard_press
# =============================================================================

def keyboard_press(key: str, presses: int = 1) -> str:
    """
    Press a single key.
    
    ONE ACTION: Press a key one or more times.
    
    Args:
        key: Key to press (e.g., "enter", "tab", "escape").
        presses: Number of times to press (default: 1).
        
    Returns:
        Success or failure message.
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not key:
        return "❌ No key provided"
    
    try:
        pyautogui.press(key, presses=presses)
        
        times = f" x{presses}" if presses > 1 else ""
        logger.debug(f"Pressed: {key}{times}")
        return f"⌨️ Pressed: {key.capitalize()}{times}"
        
    except Exception as e:
        return f"❌ Key press failed: {e}"


__all__ = [
    "mouse_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_hotkey",
    "keyboard_press",
]
