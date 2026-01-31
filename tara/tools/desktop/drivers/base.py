"""
Base Desktop Driver - Abstract Interface.

Defines the contract that all OS-specific drivers must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List


class DesktopDriver(ABC):
    """
    Abstract base class for desktop automation drivers.
    
    All platform-specific drivers (Windows, Linux, macOS) must implement
    these methods to provide consistent desktop automation capabilities.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the driver name (e.g., 'Windows UIAutomation', 'Universal PyAutoGUI')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this driver's dependencies are available."""
        pass
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    @abstractmethod
    async def dump_ui_tree(
        self,
        window_alias: str,
        depth: int = 3,
        max_elements: int = 100,
    ) -> str:
        """
        Scan UI tree and return formatted element map for LLM vision.
        
        Args:
            window_alias: Window identifier from registry.
            depth: Maximum tree traversal depth.
            max_elements: Maximum elements to return.
            
        Returns:
            Formatted string listing UI elements.
        """
        pass
    
    @abstractmethod
    async def click_element(
        self,
        window_alias: str,
        element_name: str,
        click_type: str = "left",
        timeout: float = 5.0,
    ) -> str:
        """
        Click a UI element by name.
        
        Args:
            window_alias: Window identifier from registry.
            element_name: Name/text of the element to click.
            click_type: Type of click ('left', 'right', 'double').
            timeout: Max wait time for element.
            
        Returns:
            Success or error message.
        """
        pass
    
    @abstractmethod
    async def type_text(
        self,
        window_alias: str,
        element_name: str,
        text: str,
        timeout: float = 5.0,
        clear_first: bool = True,
    ) -> str:
        """
        Type text into a UI element.
        
        Args:
            window_alias: Window identifier from registry.
            element_name: Name of the input element.
            text: Text to type.
            timeout: Max wait time for element.
            clear_first: Clear existing text before typing.
            
        Returns:
            Success or error message.
        """
        pass
    
    @abstractmethod
    async def read_element_text(
        self,
        window_alias: str,
        element_name: str,
        timeout: float = 5.0,
    ) -> str:
        """
        Read text/value from a UI element.
        
        Args:
            window_alias: Window identifier from registry.
            element_name: Name of the element to read.
            timeout: Max wait time for element.
            
        Returns:
            Element text or error message.
        """
        pass
    
    @abstractmethod
    async def get_elements_by_type(
        self,
        window_alias: str,
        control_type: str,
        max_results: int = 10,
    ) -> str:
        """
        List elements of a specific type in a window.
        
        Args:
            window_alias: Window identifier from registry.
            control_type: Type of control (e.g., 'Button', 'Edit').
            max_results: Maximum elements to return.
            
        Returns:
            Formatted list of matching elements.
        """
        pass


__all__ = ["DesktopDriver"]
