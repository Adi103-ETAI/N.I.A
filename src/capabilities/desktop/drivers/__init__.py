"""
Desktop Drivers Package - Cross-Platform Abstraction.

Provides OS-specific desktop automation drivers via Factory pattern.

Usage:
    from src.capabilities.desktop.drivers import get_desktop_driver
    
    driver = get_desktop_driver()
    await driver.click_element("notepad_1", "Submit")
"""
from .base import DesktopDriver
from .factory import get_desktop_driver, get_driver_info

__all__ = [
    "DesktopDriver",
    "get_desktop_driver",
    "get_driver_info",
]
