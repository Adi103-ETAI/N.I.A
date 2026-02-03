"""
MODULE: Input Operations (The Blind Hands - Layer 3)
VERSION: 2.5.2
STRICT SCOPE: Mouse/Keyboard control using Raw Coordinates (Fallback Layer).
CONSTRAINTS: Coordinate-based ONLY. Use uia_ops for semantic interaction.

TARA 2.0 Atomic Tool Module - ASYNC UPDATE.

Architecture:
    This is the fallback layer when UI Automation fails. Uses raw screen 
    coordinates without element awareness. Should only be used when:
    - uia_ops cannot find the target element
    - The target application doesn't support UI Automation
    - Direct pixel coordinates are known (e.g., from screenshot analysis)

Safety Features:
    - pyautogui.FAILSAFE = True (move mouse to corner to abort)
    - pyautogui.PAUSE = 0.1 (small pause between actions)

Exports:
    - mouse_click(x, y, button, double) -> str
    - mouse_drag(start_x, start_y, end_x, end_y, duration) -> str
    - mouse_scroll(clicks, x, y) -> str
    - keyboard_type(text, interval) -> str
    - keyboard_hotkey(*keys) -> str
    - keyboard_press(key, presses) -> str

LLM Usage Tips:
    - Prefer uia_ops tools (click_element, type_in_element) over these
    - Use these only when you have exact pixel coordinates
    - Always call dump_ui_tree first to try semantic interaction
"""
from __future__ import annotations

import ast
import asyncio
import time
from typing import List, Optional, Tuple, Union

from src.core.logger import setup_logger

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
# Input Parsing Helpers (LLM Robustness)
# =============================================================================

def _parse_int(value: Union[int, str, float], name: str = "value") -> int:
    """Safely parse an integer from various input types.
    
    Handles stringified inputs like '500' or '500.0' from LLM.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            # Try to parse as float first (handles '500.0')
            return int(float(value.strip()))
        except (ValueError, AttributeError):
            raise ValueError(f"Cannot parse {name}: '{value}' is not a valid integer")
    raise ValueError(f"Cannot parse {name}: expected int, got {type(value).__name__}")


def _parse_keys(keys: Union[List[str], Tuple[str, ...], str]) -> Tuple[str, ...]:
    """Safely parse keys from various input types.
    
    Handles stringified lists like "['ctrl', 's']" from LLM.
    """
    if isinstance(keys, tuple):
        return keys
    if isinstance(keys, list):
        return tuple(keys)
    if isinstance(keys, str):
        try:
            parsed = ast.literal_eval(keys)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
            else:
                return (keys,)
        except (ValueError, SyntaxError):
            # Plain string key like "enter"
            return (keys,)
    return (str(keys),)


# =============================================================================
# Atomic Tool: mouse_click
# =============================================================================

async def mouse_click(
    x: Union[int, str],
    y: Union[int, str],
    button: str = "left",
    double: bool = False,
) -> str:
    """
    Click at specific screen coordinates (Async).
    
    ONE ACTION: Perform a mouse click at the specified (x, y) position.
    
    Args:
        x: Horizontal pixel coordinate from the left edge of the screen.
           Accepts string like '500' for LLM compatibility.
        y: Vertical pixel coordinate from the top edge of the screen.
           Accepts string like '300' for LLM compatibility.
        button: Mouse button to click. Options: "left", "right", "middle".
                Default is "left".
        double: If True, performs a double-click instead of single-click.
                Default is False.
    
    Returns:
        Success message with click details, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await mouse_click(500, 300)
        "✅ Clicked at (500, 300)"
        
        >>> await mouse_click("100", "200", button="right")
        "✅ Clicked (right) at (100, 200)"
        
        >>> await mouse_click(400, 400, double=True)
        "✅ Double-clicked at (400, 400)"
    
    LLM Usage Tip:
        Only use when you have exact pixel coordinates (e.g., from screenshot
        analysis with IRIS). Prefer click_element() from uia_ops when possible
        as it uses semantic element names instead of coordinates.
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
    # === ROBUST PARSING (LLM Compatibility) ===
    try:
        x = _parse_int(x, "x")
        y = _parse_int(y, "y")
    except ValueError as e:
        return f"❌ {e}"
    
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
    start_x: Union[int, str],
    start_y: Union[int, str],
    end_x: Union[int, str],
    end_y: Union[int, str],
    duration: Union[float, str] = 0.5,
) -> str:
    """
    Drag the mouse from one point to another (Async).
    
    ONE ACTION: Click-hold at start position, move to end position, release.
    
    Args:
        start_x: Starting horizontal pixel coordinate. Accepts string for LLM.
        start_y: Starting vertical pixel coordinate. Accepts string for LLM.
        end_x: Ending horizontal pixel coordinate. Accepts string for LLM.
        end_y: Ending vertical pixel coordinate. Accepts string for LLM.
        duration: Time in seconds for the drag motion. Default is 0.5 seconds.
                  Slower drags (higher duration) are more reliable.
    
    Returns:
        Success message with drag details, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await mouse_drag(100, 100, 500, 500)
        "✅ Dragged from (100, 100) to (500, 500)"
        
        >>> await mouse_drag("0", "0", "200", "200", duration=1.0)
        "✅ Dragged from (0, 0) to (200, 200)"
    
    LLM Usage Tip:
        Useful for:
        - Drag-and-drop file operations
        - Slider adjustments
        - Drawing or selection rectangles
        - Window resizing (when Win32 APIs fail)
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
    # === ROBUST PARSING (LLM Compatibility) ===
    try:
        start_x = _parse_int(start_x, "start_x")
        start_y = _parse_int(start_y, "start_y")
        end_x = _parse_int(end_x, "end_x")
        end_y = _parse_int(end_y, "end_y")
        if isinstance(duration, str):
            duration = float(duration.strip())
    except ValueError as e:
        return f"❌ {e}"
    
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

async def mouse_scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Scroll the mouse wheel at a position (Async).
    
    ONE ACTION: Scroll up or down by specified number of "clicks".
    
    Args:
        clicks: Number of scroll units. Positive = scroll UP, negative = scroll DOWN.
                Typical values: 3 for small scroll, 10 for page scroll.
        x: Optional horizontal coordinate to move mouse before scrolling.
           If None, scrolls at current mouse position.
        y: Optional vertical coordinate to move mouse before scrolling.
           If None, scrolls at current mouse position.
    
    Returns:
        Success message with scroll details, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await mouse_scroll(3)
        "📜 Scrolled up by 3 clicks"
        
        >>> await mouse_scroll(-5)
        "📜 Scrolled down by 5 clicks"
        
        >>> await mouse_scroll(10, x=500, y=300)
        "📜 Scrolled up by 10 clicks"
    
    LLM Usage Tip:
        Use to navigate long documents, web pages, or lists.
        Scroll direction: positive clicks = UP (towards top), negative = DOWN.
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
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
    
    ONE ACTION: Simulate typing each character of the provided text.
    
    Args:
        text: The string to type. Can include letters, numbers, and basic symbols.
              Note: Special characters may not work on all keyboard layouts.
        interval: Delay between each keystroke in seconds. Default is 0.05.
                  Slower typing (higher interval) is more reliable on slow apps.
    
    Returns:
        Success message showing what was typed, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await keyboard_type("Hello, World!")
        "⌨️ Typed: 'Hello, World!'"
        
        >>> await keyboard_type("search query", interval=0.1)
        "⌨️ Typed: 'search query'"
    
    LLM Usage Tip:
        - Prefer type_in_element() from uia_ops for form fields
        - Use this when typing into applications that don't support UIA
        - For special keys (Enter, Tab, etc.), use keyboard_press() instead
        - Text longer than 30 chars will be truncated in the success message
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
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

async def keyboard_hotkey(keys: Union[List[str], Tuple[str, ...], str]) -> str:
    """
    Press a keyboard shortcut combination (Async).
    
    ONE ACTION: Press and release multiple keys simultaneously as a hotkey.
    
    Args:
        keys: List, Tuple, or string of key names to press together.
              Accepts stringified lists like "['ctrl', 's']" for LLM compatibility.
              Common modifiers: "ctrl", "alt", "shift", "win"
              Common keys: "a"-"z", "0"-"9", "enter", "tab", "escape", "space"
              Function keys: "f1"-"f12"
              Navigation: "up", "down", "left", "right", "home", "end"
    
    Returns:
        Success message showing the key combination, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await keyboard_hotkey(["ctrl", "c"])
        "⌨️ Pressed: Ctrl+C"
        
        >>> await keyboard_hotkey("['ctrl', 'shift', 'escape']")
        "⌨️ Pressed: Ctrl+Shift+Escape"
        
        >>> await keyboard_hotkey("enter")
        "⌨️ Pressed: Enter"
    
    LLM Usage Tip:
        Common hotkeys:
        - Copy: ["ctrl", "c"]
        - Paste: ["ctrl", "v"]
        - Undo: ["ctrl", "z"]
        - Save: ["ctrl", "s"]
        - Select All: ["ctrl", "a"]
        - Close Window: ["alt", "f4"]
        - Task Manager: ["ctrl", "shift", "escape"]
        - Show Desktop: ["win", "d"]
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
    if not keys:
        return "❌ No keys provided"
    
    # === ROBUST PARSING (LLM Compatibility) ===
    keys = _parse_keys(keys)
    
    # Final validation
    if not keys or not all(isinstance(k, str) for k in keys):
        return "❌ Invalid keys format. Expected list/tuple of strings."
    
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
    Press a single key one or more times (Async).
    
    ONE ACTION: Press and release a single keyboard key.
    
    Args:
        key: Name of the key to press.
             Letters/numbers: "a"-"z", "0"-"9"
             Special keys: "enter", "tab", "escape", "space", "backspace", "delete"
             Navigation: "up", "down", "left", "right", "home", "end", "pageup", "pagedown"
             Function: "f1"-"f12"
        presses: Number of times to press the key. Default is 1.
    
    Returns:
        Success message showing the key pressed, or error message if failed.
    
    Raises:
        No exceptions raised - errors are returned as strings.
    
    Example:
        >>> await keyboard_press("enter")
        "⌨️ Pressed: Enter"
        
        >>> await keyboard_press("tab", presses=3)
        "⌨️ Pressed: Tab x3"
        
        >>> await keyboard_press("backspace", presses=5)
        "⌨️ Pressed: Backspace x5"
    
    LLM Usage Tip:
        Use for:
        - Submitting forms (enter)
        - Navigating between fields (tab)
        - Closing dialogs (escape)
        - Deleting text (backspace, delete)
        - Navigating lists/menus (up, down)
    """
    if not _HAS_PYAUTOGUI:
        return "❌ pyautogui not installed. Run: pip install pyautogui"
    
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "mouse_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_hotkey",
    "keyboard_press",
]
