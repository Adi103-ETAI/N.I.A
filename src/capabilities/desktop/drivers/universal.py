"""
Universal Desktop Driver - PyAutoGUI Fallback.

Cross-platform fallback driver using PyAutoGUI for Linux/macOS.
Provides basic automation capabilities when platform-specific drivers
are unavailable.

v3.1 - Operation Universal:
    Created as fallback for non-Windows platforms.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from core.logger import setup_logger
from .base import DesktopDriver

logger = setup_logger("DRIVER.Universal")

# =============================================================================
# Guarded Import: PyAutoGUI (Cross-platform)
# =============================================================================

_HAS_PYAUTOGUI = False
pyautogui = None

try:
    import pyautogui  # type: ignore
    _HAS_PYAUTOGUI = True
    # Disable fail-safe for automated use (move mouse to corner to abort)
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
except ImportError:
    logger.debug("pyautogui not available")


# =============================================================================
# Universal Driver Implementation
# =============================================================================

class UniversalDriver(DesktopDriver):
    """
    Universal Desktop Driver using PyAutoGUI.
    
    Provides basic cross-platform automation as a fallback when
    platform-specific drivers (Windows UIAutomation) are unavailable.
    
    Limitations:
        - No semantic UI tree access (use Vision instead)
        - Click requires coordinates or image matching
        - Type requires window to be focused
    """
    
    @property
    def name(self) -> str:
        return "Universal PyAutoGUI"
    
    @property
    def is_available(self) -> bool:
        return _HAS_PYAUTOGUI
    
    # =========================================================================
    # Interface Implementation
    # =========================================================================
    
    async def dump_ui_tree(
        self,
        window_alias: str,
        depth: int = 3,
        max_elements: int = 100,
    ) -> str:
        """
        UI Tree not available in Universal mode.
        
        On Linux/macOS, use the Vision module (IRIS) to analyze the screen.
        """
        return (
            "⚠️ UI Tree not available on Linux/macOS.\n"
            "The Universal driver cannot access the Accessibility Tree.\n\n"
            "Alternatives:\n"
            "  1. Use IRIS Vision: 'analyze my screen'\n"
            "  2. Use coordinates: click at specific x,y position\n"
            "  3. Use image matching (not yet implemented)"
        )
    
    async def click_element(
        self,
        window_alias: str,
        element_name: str,
        click_type: str = "left",
        timeout: float = 5.0,
    ) -> str:
        """
        Click by element name not supported in Universal mode.
        
        Future: Could use image matching or OCR to find elements.
        """
        if not _HAS_PYAUTOGUI:
            return "❌ pyautogui library not installed. Run: pip install pyautogui"
        
        # For now, log the limitation
        logger.warning(f"Universal driver cannot click by name: '{element_name}'")
        
        return (
            f"⚠️ Cannot click '{element_name}' by name in Universal mode.\n"
            "The Universal driver requires coordinates or image matching.\n\n"
            "Alternatives:\n"
            "  1. Use Vision to find element coordinates\n"
            "  2. Use click_at(x, y) with specific coordinates"
        )
    
    async def click_at(self, x: int, y: int, click_type: str = "left") -> str:
        """
        Click at specific screen coordinates.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            click_type: 'left', 'right', or 'double'.
            
        Returns:
            Success or error message.
        """
        if not _HAS_PYAUTOGUI:
            return "❌ pyautogui library not installed"
        
        def _do_click():
            try:
                if click_type == "left":
                    pyautogui.click(x, y)
                elif click_type == "right":
                    pyautogui.rightClick(x, y)
                elif click_type == "double":
                    pyautogui.doubleClick(x, y)
                else:
                    return f"❌ Invalid click_type '{click_type}'"
                
                logger.debug(f"Clicked at ({x}, {y})")
                return f"✅ Clicked at ({x}, {y})"
            except Exception as e:
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
        """
        Type text using PyAutoGUI (blind typing).
        
        Note: This types into whatever window/element is currently focused.
        """
        if not _HAS_PYAUTOGUI:
            return "❌ pyautogui library not installed"
        
        if not text:
            return "❌ No text provided to type"
        
        def _do_type():
            try:
                if clear_first:
                    # Select all and replace
                    pyautogui.hotkey('ctrl', 'a')
                    pyautogui.sleep(0.1)
                
                pyautogui.typewrite(text, interval=0.02)
                
                logger.debug(f"Typed '{text[:30]}...' (blind mode)")
                return f"✅ Typed '{text[:30]}...' (Universal blind mode)"
            except Exception as e:
                return f"❌ Type failed: {e}"
        
        return await asyncio.to_thread(_do_type)
    
    async def read_element_text(
        self,
        window_alias: str,
        element_name: str,
        timeout: float = 5.0,
    ) -> str:
        """
        Read element text not supported in Universal mode.
        
        Use Vision module for screen content analysis.
        """
        return (
            "⚠️ Reading element text not available in Universal mode.\n"
            "Use IRIS Vision to analyze screen content."
        )
    
    async def get_elements_by_type(
        self,
        window_alias: str,
        control_type: str,
        max_results: int = 10,
    ) -> str:
        """
        Element enumeration not available in Universal mode.
        """
        return (
            "⚠️ Element enumeration not available in Universal mode.\n"
            "Use IRIS Vision to analyze screen content."
        )
    
    # =========================================================================
    # Universal-Specific Methods
    # =========================================================================
    
    async def screenshot(self, region: Optional[tuple] = None) -> str:
        """
        Take a screenshot using PyAutoGUI.
        
        Args:
            region: Optional (x, y, width, height) tuple.
            
        Returns:
            Path to saved screenshot or error.
        """
        if not _HAS_PYAUTOGUI:
            return "❌ pyautogui library not installed"
        
        from pathlib import Path
        from datetime import datetime
        
        def _do_screenshot():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = Path.home() / "Downloads" / filename
                
                if region:
                    img = pyautogui.screenshot(region=region)
                else:
                    img = pyautogui.screenshot()
                
                img.save(str(filepath))
                return f"✅ Screenshot saved: {filepath}"
            except Exception as e:
                return f"❌ Screenshot failed: {e}"
        
        return await asyncio.to_thread(_do_screenshot)
    
    async def hotkey(self, *keys: str) -> str:
        """
        Press a keyboard shortcut.
        
        Args:
            keys: Key names (e.g., 'ctrl', 'c').
            
        Returns:
            Success or error message.
        """
        if not _HAS_PYAUTOGUI:
            return "❌ pyautogui library not installed"
        
        def _do_hotkey():
            try:
                pyautogui.hotkey(*keys)
                return f"✅ Pressed: {'+'.join(keys)}"
            except Exception as e:
                return f"❌ Hotkey failed: {e}"
        
        return await asyncio.to_thread(_do_hotkey)


__all__ = ["UniversalDriver"]
