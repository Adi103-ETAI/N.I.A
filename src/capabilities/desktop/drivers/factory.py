"""
Desktop Driver Factory - OS-Aware Driver Selection.

Provides the get_desktop_driver() function that returns the appropriate
desktop automation driver based on the current operating system.

v3.1 - Operation Universal:
    Uses OSContext for cross-platform driver selection.
"""
from __future__ import annotations

from typing import Optional

from src.core.logger import setup_logger
from src.core.os.platform import get_os_context
from .base import DesktopDriver

logger = setup_logger("DRIVER.Factory")

# Cached driver instance
_cached_driver: Optional[DesktopDriver] = None


def get_desktop_driver(force_new: bool = False) -> DesktopDriver:
    """
    Get the appropriate desktop driver for the current OS.
    
    Uses Factory pattern with caching for performance.
    
    Args:
        force_new: Force creation of a new driver instance.
        
    Returns:
        DesktopDriver instance (WindowsDriver or UniversalDriver).
    """
    global _cached_driver
    
    if _cached_driver is not None and not force_new:
        return _cached_driver
    
    ctx = get_os_context()
    
    if ctx.is_windows:
        # Try Windows UIAutomation driver
        try:
            from .windows import WindowsDriver
            driver = WindowsDriver()
            
            if driver.is_available:
                logger.debug(f"Using driver: {driver.name}")
                _cached_driver = driver
                return driver
            else:
                logger.warning("WindowsDriver not available, falling back to Universal")
        except ImportError as e:
            logger.warning(f"Failed to import WindowsDriver: {e}")
    
    # Fallback: Universal driver (Linux, macOS, or Windows without UIAutomation)
    from .universal import UniversalDriver
    driver = UniversalDriver()
    
    if driver.is_available:
        logger.debug(f"Using driver: {driver.name}")
    else:
        logger.warning("UniversalDriver dependencies not available - limited functionality")
    
    _cached_driver = driver
    return driver


def get_driver_info() -> dict:
    """
    Get information about the current driver.
    
    Returns:
        Dict with driver name, OS, and availability status.
    """
    ctx = get_os_context()
    driver = get_desktop_driver()
    
    return {
        "os": ctx.os_name,
        "driver": driver.name,
        "available": driver.is_available,
    }


__all__ = ["get_desktop_driver", "get_driver_info"]
