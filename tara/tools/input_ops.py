"""
MODULE: Input Operations (The Blind Hands - Layer 3)
STRICT SCOPE: Mouse/Keyboard control using Coordinates (Fallback).
CONSTRAINTS: Coordinate-based ONLY. Use uia_ops for semantic interaction.

TARA 2.0 Atomic Tool Module - ASYNC UPDATE.

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

import asyncio
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

async def mouse_click(
    x: int,
    y: int,
    button: str = "left",
    double: bool = False,
) -> str:
    """
    Click at screen coordinates (Async).
    
    ONE ACTION: Mouse click at (x, y).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    def _do_click():
        try:
            clicks = 2 if double else 1
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            
            click_type = "Double-clicked" if double else "Clicked"
            button_str = f" ({button})" if button != "left" else ""
            
            logger.debug(f"{click_type}{button_str} at ({x}, {y})")
            return f"✅ {click_type}{button_str} at ({x}, {y})"
        except Exception as e:
            return f"❌ Click failed: {e}"

    return await asyncio.to_thread(_do_click)


# =============================================================================
# Atomic Tool: mouse_drag
# =============================================================================

async def mouse_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
) -> str:
    """
    Drag mouse from one point to another (Async).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    def _do_drag():
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

    return await asyncio.to_thread(_do_drag)


# =============================================================================
# Atomic Tool: mouse_scroll
# =============================================================================

async def mouse_scroll(clicks: int, x: int = None, y: int = None) -> str:
    """
    Scroll the mouse wheel (Async).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    def _do_scroll():
        try:
            pyautogui.scroll(clicks, x=x, y=y)
            
            direction = "up" if clicks > 0 else "down"
            logger.debug(f"Scrolled {direction} by {abs(clicks)} clicks")
            return f"📜 Scrolled {direction} by {abs(clicks)} clicks"
        except Exception as e:
            return f"❌ Scroll failed: {e}"

    return await asyncio.to_thread(_do_scroll)


# =============================================================================
# Atomic Tool: keyboard_type
# =============================================================================

async def keyboard_type(text: str, interval: float = 0.05) -> str:
    """
    Type text using keyboard simulation (Async).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not text:
        return "❌ No text provided"
    
    def _do_type():
        try:
            pyautogui.write(text, interval=interval)
            
            display_text = text[:30] + "..." if len(text) > 30 else text
            logger.debug(f"Typed: {display_text}")
            return f"⌨️ Typed: '{display_text}'"
        except Exception as e:
            return f"❌ Type failed: {e}"

    return await asyncio.to_thread(_do_type)


# =============================================================================
# Atomic Tool: keyboard_hotkey
# =============================================================================

async def keyboard_hotkey(*keys: str) -> str:
    """
    Press a keyboard shortcut combination (Async).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not keys:
        return "❌ No keys provided"
    
    def _do_hotkey():
        try:
            pyautogui.hotkey(*keys)
            
            combo = "+".join(k.capitalize() for k in keys)
            logger.debug(f"Hotkey: {combo}")
            return f"⌨️ Pressed: {combo}"
        except Exception as e:
            return f"❌ Hotkey failed: {e}"

    return await asyncio.to_thread(_do_hotkey)


# =============================================================================
# Atomic Tool: keyboard_press
# =============================================================================

async def keyboard_press(key: str, presses: int = 1) -> str:
    """
    Press a single key (Async).
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed"
    
    if not key:
        return "❌ No key provided"
    
    def _do_press():
        try:
            pyautogui.press(key, presses=presses)
            
            times = f" x{presses}" if presses > 1 else ""
            logger.debug(f"Pressed: {key}{times}")
            return f"⌨️ Pressed: {key.capitalize()}{times}"
        except Exception as e:
            return f"❌ Key press failed: {e}"

    return await asyncio.to_thread(_do_press)


__all__ = [
    "mouse_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_hotkey",
    "keyboard_press",
]
