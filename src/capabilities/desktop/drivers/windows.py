"""
Windows Desktop Driver - UIAutomation Implementation.

Uses Windows UIAutomation API for semantic UI element interaction.
Only imported on Windows systems to prevent import errors on Linux/macOS.

v3.1 - Operation Universal:
    Extracted from uia_ops.py to enable cross-platform support.
"""
from __future__ import annotations

import asyncio
import time
import json
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

from core.logger import setup_logger
from .base import DesktopDriver

logger = setup_logger("DRIVER.Windows")

# =============================================================================
# Guarded Import: UIAutomation (Windows-only)
# =============================================================================

_HAS_UIA = False
auto = None

try:
    import uiautomation as auto  # type: ignore
    _HAS_UIA = True
except ImportError:
    logger.debug("uiautomation not available (expected on non-Windows)")


# =============================================================================
# Configuration
# =============================================================================

def _load_config() -> dict:
    """Load UIA configuration from centralized ROOT/config/tara/."""
    config_path = Path(__file__).resolve().parents[4] / "config" / "tara" / "uia.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load uia.json: {e}")
        return {}

_UIA_CONFIG = _load_config()

ACTIONABLE_TYPES = set(_UIA_CONFIG.get("ACTIONABLE_TYPES", [
    "ButtonControl", "EditControl", "TextControl", "DocumentControl",
    "MenuItemControl", "ListControl", "ListItemControl", "TabItemControl",
    "HyperlinkControl", "CheckBoxControl", "RadioButtonControl",
    "ComboBoxControl", "TreeItemControl",
]))

SKIP_TYPES = set(_UIA_CONFIG.get("SKIP_TYPES", [
    "PaneControl", "GroupControl", "WindowControl", "ScrollBarControl",
    "ThumbControl", "SeparatorControl",
]))

MAX_ELEMENTS = _UIA_CONFIG.get("MAX_ELEMENTS", 100)


# =============================================================================
# Windows Driver Implementation
# =============================================================================

class WindowsDriver(DesktopDriver):
    """
    Windows Desktop Driver using UIAutomation.
    
    Provides semantic UI interaction via the Windows Accessibility API.
    """
    
    @property
    def name(self) -> str:
        return "Windows UIAutomation"
    
    @property
    def is_available(self) -> bool:
        return _HAS_UIA
    
    def _get_window_control(self, window_alias: str) -> Tuple[Optional[object], Optional[str]]:
        """Get UIA Control from registry alias."""
        if not _HAS_UIA:
            return None, "uiautomation library not installed"
        
        from ..window_manager import get_registry
        registry = get_registry()
        
        if window_alias not in registry:
            available = registry.list_aliases()
            avail_str = ", ".join(available[:5]) if available else "none"
            return None, f"Unknown alias '{window_alias}'. Available: {avail_str}"
        
        hwnd = registry.get_handle(window_alias)
        if not hwnd:
            return None, f"Alias '{window_alias}' has no HWND registered"
        
        try:
            control = auto.ControlFromHandle(hwnd)
            if control is None:
                return None, f"Could not get control for HWND {hwnd}"
            return control, None
        except Exception as e:
            return None, f"Failed to get control: {e}"
    
    def _wait_for_element(
        self,
        window: object,
        name: str,
        control_type: Optional[str] = None,
        timeout: float = 5.0,
        poll_interval: float = 0.3,
    ) -> object:
        """Smart Wait: Retry finding an element until timeout expires."""
        start_time = time.time()
        last_error = None
        attempt = 0
        name_lower = name.lower()
        
        while (time.time() - start_time) < timeout:
            attempt += 1
            
            try:
                if control_type:
                    element = window.GetFirstChildControl(
                        lambda ctrl, depth: (
                            ctrl.Name and name_lower in ctrl.Name.lower() and
                            ctrl.ControlTypeName == control_type
                        )
                    )
                else:
                    element = window.GetFirstChildControl(
                        lambda ctrl, depth: ctrl.Name and name_lower in ctrl.Name.lower()
                    )
                
                if element is not None:
                    logger.debug(f"✅ Found '{name}' on attempt {attempt}")
                    return element
                    
            except Exception as e:
                last_error = str(e)
            
            try:
                for ctrl, depth in auto.WalkControl(window, maxDepth=5):
                    if ctrl.Name and name_lower in ctrl.Name.lower():
                        if control_type is None or ctrl.ControlTypeName == control_type:
                            logger.debug(f"✅ Found '{name}' via walk on attempt {attempt}")
                            return ctrl
            except Exception as e:
                last_error = str(e)
            
            time.sleep(poll_interval)
        
        elapsed = time.time() - start_time
        error_msg = f"Element '{name}' not found after {elapsed:.1f}s ({attempt} attempts)"
        if last_error:
            error_msg += f" (Last error: {last_error})"
        raise TimeoutError(error_msg)
    
    # =========================================================================
    # Interface Implementation
    # =========================================================================
    
    async def dump_ui_tree(
        self,
        window_alias: str,
        depth: int = 3,
        max_elements: int = MAX_ELEMENTS,
    ) -> str:
        if not _HAS_UIA:
            return "❌ uiautomation library not installed"
        
        def _do_dump():
            try:
                with auto.UIAutomationInitializerInThread():
                    window, error = self._get_window_control(window_alias)
                    if error:
                        return f"❌ {error}"
                    
                    elements = []
                    element_id = 0
                    
                    for ctrl, current_depth in auto.WalkControl(window, maxDepth=depth):
                        if element_id >= max_elements:
                            elements.append(f"... (truncated at {max_elements} elements)")
                            break
                        
                        try:
                            ctrl_type = ctrl.ControlTypeName
                            ctrl_name = ctrl.Name or ""
                            
                            if ctrl_type in SKIP_TYPES:
                                continue
                            if not ctrl_name.strip():
                                continue
                            if len(ctrl_name) > 100:
                                continue
                            
                            short_type = ctrl_type.replace("Control", "")
                            safe_name = ctrl_name.replace('"', '\\"')[:50]
                            
                            element_id += 1
                            elements.append(f'[{element_id}] {{{short_type}}} "{safe_name}"')
                            
                        except Exception:
                            continue
                    
                    if not elements:
                        return f"Window '{window_alias}' has no readable controls."
                    
                    header = f"UI Elements for '{window_alias}' ({len(elements)} items):\n"
                    return header + "\n".join(elements)
            except Exception as e:
                logger.error(f"UI tree scan error: {e}")
                return f"❌ Failed to scan UI: {e}"
        
        return await asyncio.to_thread(_do_dump)
    
    async def click_element(
        self,
        window_alias: str,
        element_name: str,
        click_type: str = "left",
        timeout: float = 5.0,
    ) -> str:
        if not _HAS_UIA:
            return "❌ uiautomation library not installed"
        
        def _do_click():
            try:
                with auto.UIAutomationInitializerInThread():
                    window, error = self._get_window_control(window_alias)
                    if error:
                        return f"❌ {error}"
                    
                    element = self._wait_for_element(window, element_name, timeout=timeout)
                    
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
    
    async def type_text(
        self,
        window_alias: str,
        element_name: str,
        text: str,
        timeout: float = 5.0,
        clear_first: bool = True,
    ) -> str:
        if not _HAS_UIA:
            return "❌ uiautomation library not installed"
        
        if not text:
            return "❌ No text provided to type"
        
        def _do_type():
            try:
                with auto.UIAutomationInitializerInThread():
                    window, error = self._get_window_control(window_alias)
                    if error:
                        return f"❌ {error}"
                    
                    element = None
                    use_blind_typing = False
                    
                    try:
                        element = self._wait_for_element(
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
    
    async def read_element_text(
        self,
        window_alias: str,
        element_name: str,
        timeout: float = 5.0,
    ) -> str:
        if not _HAS_UIA:
            return "❌ uiautomation library not installed"
        
        def _do_read():
            try:
                with auto.UIAutomationInitializerInThread():
                    window, error = self._get_window_control(window_alias)
                    if error:
                        return f"❌ {error}"
                    
                    element = self._wait_for_element(window, element_name, timeout=timeout)
                    
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
    
    async def get_elements_by_type(
        self,
        window_alias: str,
        control_type: str,
        max_results: int = 10,
    ) -> str:
        if not _HAS_UIA:
            return "❌ uiautomation library not installed"
        
        if not control_type.endswith("Control"):
            control_type = f"{control_type}Control"
        
        def _do_search():
            try:
                with auto.UIAutomationInitializerInThread():
                    window, error = self._get_window_control(window_alias)
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


__all__ = ["WindowsDriver"]
