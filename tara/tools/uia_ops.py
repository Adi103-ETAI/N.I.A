"""
MODULE: UI Automation Operations (Layer 1 - Semantic)
STRICT SCOPE: Smart element interaction using Accessibility Tree.
CONSTRAINTS: Uses Names/Types, NOT coordinates. The "Smart" layer.

TARA 2.0 Atomic Tool Module - ASYNC UPDATE.

Bridges the Vision Gap and Wait Gap from legacy uia_driver.py.

Exports:
    - dump_ui_tree(window_alias: str, depth: int) -> str
    - click_element(window_alias: str, element_name: str, click_type: str) -> str
    - type_in_element(window_alias: str, element_name: str, text: str) -> str
    - read_element_text(window_alias: str, element_name: str) -> str
"""
from __future__ import annotations

import asyncio
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

from core.logger import setup_logger
from core.config import get_settings

from .window_manager import get_registry

logger = setup_logger("TARA.Tools.UIAOps")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import uiautomation as auto
    _HAS_UIA = True
except ImportError:
    _HAS_UIA = False
    auto = None  # type: ignore
    logger.warning("uiautomation not available - UIA operations disabled")

# Load settings
settings = get_settings()

# =============================================================================
# Constants
# =============================================================================

def _load_config() -> dict:
    """Load UIA configuration from JSON."""
    config_path = Path(__file__).parent.parent.parent / "config" / "tara" / "uia.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load uia.json: {e}")
        return {}

_UIA_CONFIG = _load_config()

# Element types we care about (filter out layout noise)
ACTIONABLE_TYPES = set(_UIA_CONFIG.get("ACTIONABLE_TYPES", [
    "ButtonControl", "EditControl", "TextControl", "DocumentControl",
    "MenuItemControl", "ListControl", "ListItemControl", "TabItemControl",
    "HyperlinkControl", "CheckBoxControl", "RadioButtonControl",
    "ComboBoxControl", "TreeItemControl",
]))

# Types to skip (layout/container clutter)
SKIP_TYPES = set(_UIA_CONFIG.get("SKIP_TYPES", [
    "PaneControl", "GroupControl", "WindowControl", "ScrollBarControl",
    "ThumbControl", "SeparatorControl",
]))

# Maximum elements to return (token budget)
MAX_ELEMENTS = _UIA_CONFIG.get("MAX_ELEMENTS", 100)


# =============================================================================
# Helper: Get Window Control from HWND
# =============================================================================

def _get_window_control(window_alias: str) -> Tuple[Optional["auto.Control"], Optional[str]]:
    """
    Get UIA Control from registry alias.
    
    Args:
        window_alias: Window alias from registry (e.g., "notepad_1").
        
    Returns:
        Tuple of (Control, None) if found, or (None, error_message).
    
    NOTE: Caller MUST be inside UIAutomationInitializerInThread context.
    """
    if not _HAS_UIA:
        return None, "uiautomation library not installed"
    
    registry = get_registry()
    
    # Check if alias exists
    if window_alias not in registry:
        available = registry.list_aliases()
        avail_str = ", ".join(available[:5]) if available else "none"
        return None, f"Unknown alias '{window_alias}'. Available: {avail_str}"
    
    # Get HWND
    hwnd = registry.get_handle(window_alias)
    if not hwnd:
        return None, f"Alias '{window_alias}' has no HWND registered"
    
    try:
        # Create control from HWND
        control = auto.ControlFromHandle(hwnd)
        if control is None:
            return None, f"Could not get control for HWND {hwnd}"
        return control, None
    except Exception as e:
        return None, f"Failed to get control: {e}"


# =============================================================================
# Helper: The "Patience" Logic (Wait Gap Fix)
# =============================================================================

def _wait_for_element(
    window: "auto.Control",
    name: str,
    control_type: Optional[str] = None,
    timeout: float = 5.0,
    poll_interval: float = 0.3,
) -> "auto.Control":
    """
    Smart Wait: Retry finding an element until timeout expires.
    """
    start_time = time.time()
    last_error = None
    attempt = 0
    
    name_lower = name.lower()
    
    while (time.time() - start_time) < timeout:
        attempt += 1
        
        try:
            # Strategy 1: Search by Name (exact)
            if control_type:
                # Filter by both name and type
                element = window.GetFirstChildControl(
                    lambda ctrl, depth: (
                        ctrl.Name and name_lower in ctrl.Name.lower() and
                        ctrl.ControlTypeName == control_type
                    )
                )
            else:
                # Search by name only
                element = window.GetFirstChildControl(
                    lambda ctrl, depth: ctrl.Name and name_lower in ctrl.Name.lower()
                )
            
            if element is not None:
                logger.debug(f"✅ Found '{name}' on attempt {attempt}")
                return element
                
        except Exception as e:
            last_error = str(e)
        
        # Strategy 2: Deep search with recursion limit
        try:
            for ctrl, depth in auto.WalkControl(window, maxDepth=5):
                if ctrl.Name and name_lower in ctrl.Name.lower():
                    if control_type is None or ctrl.ControlTypeName == control_type:
                        logger.debug(f"✅ Found '{name}' via walk on attempt {attempt}")
                        return ctrl
        except Exception as e:
            last_error = str(e)
        
        time.sleep(poll_interval)
    
    # Timeout reached
    elapsed = time.time() - start_time
    error_msg = f"Element '{name}' not found after {elapsed:.1f}s ({attempt} attempts)"
    if last_error:
        error_msg += f" (Last error: {last_error})"
    
    raise TimeoutError(error_msg)


# =============================================================================
# Core Function: The "Vision" Logic
# =============================================================================

async def dump_ui_tree(window_alias: str, depth: int = 3, max_elements: int = MAX_ELEMENTS) -> str:
    """
    Scan UI tree and return formatted element map for LLM vision (Async).
    """
    if not _HAS_UIA:
        return "❌ uiautomation library not installed"
    
    def _do_dump():
        try:
            # 🌊 RIPPLE CHECK: Initialize COM for this thread BEFORE any UIA operations
            with auto.UIAutomationInitializerInThread():
                # Get window control (MUST be inside COM context)
                window, error = _get_window_control(window_alias)
                if error:
                    return f"❌ {error}"
                
                elements = []
                element_id = 0
                
                # Walk the tree with depth limit
                for ctrl, current_depth in auto.WalkControl(window, maxDepth=depth):
                    if element_id >= max_elements:
                        elements.append(f"... (truncated at {max_elements} elements)")
                        break
                    
                    try:
                        # Get element info
                        ctrl_type = ctrl.ControlTypeName
                        ctrl_name = ctrl.Name or ""
                        
                        # Skip layout/container noise
                        if ctrl_type in SKIP_TYPES:
                            continue
                        
                        # Skip elements with no name (not actionable)
                        if not ctrl_name.strip():
                            continue
                        
                        # Skip very long names (usually not buttons/inputs)
                        if len(ctrl_name) > 100:
                            continue
                        
                        # Format type name (remove "Control" suffix)
                        short_type = ctrl_type.replace("Control", "")
                        
                        # Escape quotes in name
                        safe_name = ctrl_name.replace('"', '\\"')[:50]
                        
                        # Add to list
                        element_id += 1
                        elements.append(f"[{element_id}] {{{short_type}}} \"{safe_name}\"")
                        
                    except Exception:
                        continue
                
                if not elements:
                    return f"Window '{window_alias}' has no readable controls."
                
                header = f"UI Elements for '{window_alias}' ({len(elements)} items):\n"
                return header + "\n".join(elements)
        except Exception as e:
            logger.error(f"UI tree scan error: {e}")
            return f"❌ Failed to scan UI: {e}"

    # Wrap blocking COM call in thread
    return await asyncio.to_thread(_do_dump)


# =============================================================================
# Action Function: click_element
# =============================================================================

async def click_element(
    window_alias: str,
    element_name: str,
    click_type: str = "left",
    timeout: float = 5.0,
) -> str:
    """
    Click a UI element by its name (Async).
    """
    if not _HAS_UIA:
        return "❌ uiautomation library not installed"
    
    def _do_click():
        try:
            with auto.UIAutomationInitializerInThread():
                window, error = _get_window_control(window_alias)
                if error:
                    return f"❌ {error}"
                
                element = _wait_for_element(window, element_name, timeout=timeout)
                
                if click_type == "left":
                    element.Click()
                elif click_type == "right":
                    element.RightClick()
                elif click_type == "double":
                    element.DoubleClick()
                else:
                    return f"❌ Invalid click_type '{click_type}'. Use 'left', 'right', or 'double'"
                
                logger.debug(f"Clicked '{element_name}' in '{window_alias}'")
                return f"✅ Clicked '{element_name}'"
            
        except TimeoutError as e:
            error_msg = f"Element '{element_name}' not found in '{window_alias}'"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return f"❌ Click failed: {e}"

    return await asyncio.to_thread(_do_click)


# =============================================================================
# Action Function: type_in_element
# =============================================================================

async def type_in_element(
    window_alias: str,
    element_name: str,
    text: str,
    timeout: float = 5.0,
    clear_first: bool = True,
) -> str:
    """
    Type text into a UI element with blind typing fallback (Async).
    """
    if not _HAS_UIA:
        return "❌ uiautomation library not installed"
    
    if not text:
        return "❌ No text provided to type"
    
    def _do_type():
        try:
            with auto.UIAutomationInitializerInThread():
                window, error = _get_window_control(window_alias)
                if error:
                    return f"❌ {error}"
                
                element = None
                use_blind_typing = False
                
                try:
                    element = _wait_for_element(
                        window, element_name, 
                        control_type="EditControl",
                        timeout=timeout
                    )
                except TimeoutError:
                    logger.warning(f"⚠️ Element '{element_name}' not found, falling back to blind typing")
                    use_blind_typing = True
                
                if element and not use_blind_typing:
                    element.SetFocus()
                    time.sleep(0.1)
                    
                    try:
                        if hasattr(element, 'GetValuePattern') and element.GetValuePattern():
                            pattern = element.GetValuePattern()
                            if clear_first:
                                pattern.SetValue("")
                            pattern.SetValue(text)
                            logger.debug(f"Used SetValue for '{element_name}'")
                            return f"✅ Typed '{text[:30]}...' into '{element_name}'"
                    except Exception:
                        pass
                    
                    if clear_first:
                        element.SendKeys("{Ctrl}a", waitTime=0.05)
                    element.SendKeys(text, waitTime=0.01)
                    
                    return f"✅ Typed '{text[:30]}...' into '{element_name}'"
                
                else:
                    logger.info(f"🔤 Blind typing into '{window_alias}'")
                    window.SetFocus()
                    time.sleep(0.2)
                    
                    if clear_first:
                        auto.SendKeys("{Ctrl}a", waitTime=0.05)
                    
                    auto.SendKeys(text, waitTime=0.01)
                    return f"✅ Typed '{text[:30]}...' (blind mode)"
            
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return f"❌ Type failed: {e}"

    return await asyncio.to_thread(_do_type)


# =============================================================================
# Action Function: read_element_text
# =============================================================================

async def read_element_text(
    window_alias: str,
    element_name: str,
    timeout: float = 5.0,
) -> str:
    """
    Read text/value from a UI element (Async).
    """
    if not _HAS_UIA:
        return "❌ uiautomation library not installed"
    
    def _do_read():
        try:
            with auto.UIAutomationInitializerInThread():
                window, error = _get_window_control(window_alias)
                if error:
                    return f"❌ {error}"
                
                element = _wait_for_element(window, element_name, timeout=timeout)
                
                text = None
                try:
                    pattern = element.GetValuePattern()
                    if pattern:
                        text = pattern.Value
                except Exception:
                    pass
                
                if not text:
                    try:
                        pattern = element.GetTextPattern()
                        if pattern:
                            text = pattern.DocumentRange.GetText(-1)
                    except Exception:
                        pass
                
                if not text:
                    text = element.Name
                
                if text:
                    return f"'{element_name}': {text}"
                else:
                    return f"Element '{element_name}' has no readable text"
            
        except TimeoutError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Read failed: {e}"

    return await asyncio.to_thread(_do_read)


# =============================================================================
# Convenience: List Available Element Types
# =============================================================================

async def get_element_by_type(
    window_alias: str,
    control_type: str,
    max_results: int = 10,
) -> str:
    """
    List elements of a specific type in a window (Async).
    """
    if not _HAS_UIA:
        return "❌ uiautomation library not installed"
    
    if not control_type.endswith("Control"):
        control_type = f"{control_type}Control"
    
    def _do_search():
        try:
            with auto.UIAutomationInitializerInThread():
                window, error = _get_window_control(window_alias)
                if error:
                    return f"❌ {error}"
                
                elements = []
                for ctrl, depth in auto.WalkControl(window, maxDepth=5):
                    if len(elements) >= max_results:
                        break
                    
                    if ctrl.ControlTypeName == control_type and ctrl.Name:
                        elements.append(f'  - "{ctrl.Name}"')
                
                if not elements:
                    return f"No {control_type} elements found in '{window_alias}'"
                
                return f"{control_type} in '{window_alias}':\n" + "\n".join(elements)
        except Exception as e:
            return f"❌ Search failed: {e}"

    return await asyncio.to_thread(_do_search)


__all__ = [
    "dump_ui_tree",
    "click_element",
    "type_in_element",
    "read_element_text",
    "get_element_by_type",
]
