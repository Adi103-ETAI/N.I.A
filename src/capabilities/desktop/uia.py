"""
MODULE: UI Automation Operations (Layer 1 - Semantic)
STRICT SCOPE: Smart element interaction using Accessibility Tree.
CONSTRAINTS: Uses Names/Types, NOT coordinates. The "Smart" layer.

TARA 2.0 Atomic Tool Module - ASYNC UPDATE.

v3.1 - Operation Universal:
    Now uses Driver Factory pattern for cross-platform support.
    WindowsDriver (UIAutomation) on Windows, UniversalDriver (PyAutoGUI) on Linux/macOS.

Exports:
    - dump_ui_tree(window_alias: str, depth: int) -> str
    - click_element(window_alias: str, element_name: str, click_type: str) -> str
    - type_in_element(window_alias: str, element_name: str, text: str) -> str
    - read_element_text(window_alias: str, element_name: str) -> str
    - get_element_by_type(window_alias: str, control_type: str) -> str
"""
from __future__ import annotations

from src.core.logger import setup_logger
from .drivers import get_desktop_driver

logger = setup_logger("TARA.Tools.UIAOps")


# =============================================================================
# Wrapper Functions (Use Driver via Factory)
# =============================================================================

async def dump_ui_tree(window_alias: str, depth: int = 3, max_elements: int = 100) -> str:
    """
    Scan UI tree and return formatted element map for LLM vision (Async).
    
    Args:
        window_alias: Window alias from registry (e.g., "notepad_1").
        depth: Maximum tree traversal depth.
        max_elements: Maximum elements to return.
        
    Returns:
        Formatted string listing UI elements.
    """
    driver = get_desktop_driver()
    return await driver.dump_ui_tree(window_alias, depth, max_elements)


async def click_element(
    window_alias: str,
    element_name: str,
    click_type: str = "left",
    timeout: float = 5.0,
) -> str:
    """
    Click a UI element by its name (Async).
    
    Args:
        window_alias: Window alias from registry.
        element_name: Name/text of the element to click.
        click_type: Type of click ('left', 'right', 'double').
        timeout: Max wait time for element.
        
    Returns:
        Success or error message.
    """
    driver = get_desktop_driver()
    return await driver.click_element(window_alias, element_name, click_type, timeout)


async def type_in_element(
    window_alias: str,
    element_name: str,
    text: str,
    timeout: float = 5.0,
    clear_first: bool = True,
) -> str:
    """
    Type text into a UI element with blind typing fallback (Async).
    
    Args:
        window_alias: Window alias from registry.
        element_name: Name of the input element.
        text: Text to type.
        timeout: Max wait time for element.
        clear_first: Clear existing text before typing.
        
    Returns:
        Success or error message.
    """
    driver = get_desktop_driver()
    return await driver.type_text(window_alias, element_name, text, timeout, clear_first)


async def read_element_text(
    window_alias: str,
    element_name: str,
    timeout: float = 5.0,
) -> str:
    """
    Read text/value from a UI element (Async).
    
    Args:
        window_alias: Window alias from registry.
        element_name: Name of the element to read.
        timeout: Max wait time for element.
        
    Returns:
        Element text or error message.
    """
    driver = get_desktop_driver()
    return await driver.read_element_text(window_alias, element_name, timeout)


async def get_element_by_type(
    window_alias: str,
    control_type: str,
    max_results: int = 10,
) -> str:
    """
    List elements of a specific type in a window (Async).
    
    Args:
        window_alias: Window alias from registry.
        control_type: Type of control (e.g., 'Button', 'Edit').
        max_results: Maximum elements to return.
        
    Returns:
        Formatted list of matching elements.
    """
    driver = get_desktop_driver()
    return await driver.get_elements_by_type(window_alias, control_type, max_results)


__all__ = [
    "dump_ui_tree",
    "click_element",
    "type_in_element",
    "read_element_text",
    "get_element_by_type",
]
