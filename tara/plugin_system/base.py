"""TARA Plugin System - Base Definitions.

VERSION: 3.0.0

This module defines the foundational types for the plugin architecture:
- PluginInterface: Protocol defining what a plugin must provide
- PluginMetadata: Dataclass for plugin metadata
- PluginType: Enum for plugin source types
- PluginError: Exception hierarchy for plugin failures

Design Principles:
    1. Convention over Configuration: Sensible defaults, minimal boilerplate
    2. Future-Proof: Extensible without breaking changes
    3. Type-Safe: Full typing support for IDE and static analysis

Usage:
    from tara.plugin_system.base import PluginInterface, PluginType
    
    class MyPlugin(PluginInterface):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="my_plugin",
                description="Does something cool",
                version="1.0.0",
            )
        
        def get_tools(self) -> List[Callable]:
            return [my_tool_function]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# TYPE_CHECKING guard for ServiceContainer (avoid circular import)
if TYPE_CHECKING:
    from core.container import ServiceContainer


# =============================================================================
# Plugin Type Enumeration
# =============================================================================

class PluginType(Enum):
    """Classification of plugin source types.
    
    Determines how the plugin loader discovers and loads the plugin.
    
    Attributes:
        PYTHON_SCRIPT: Single .py file dropped into plugins/ directory.
        PYTHON_PACKAGE: Directory with __init__.py (structured plugin).
        BINARY: External executable (Go/Rust/Node.js) - Future support.
    """
    PYTHON_SCRIPT = auto()   # Single .py file
    PYTHON_PACKAGE = auto()  # Folder with __init__.py
    BINARY = auto()          # External executable (future-proofing)
    
    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


# =============================================================================
# Plugin Metadata
# =============================================================================

@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata describing a plugin.
    
    Attributes:
        name: Unique identifier (lowercase, underscores allowed).
        description: Human-readable description for documentation.
        version: Semantic version string (e.g., "1.0.0").
        author: Optional author name or email.
        category: Tool category for grouping (e.g., "system", "web", "vision").
        requires_container: Whether plugin needs ServiceContainer injection.
        tags: Optional list of searchable tags.
        
    Example:
        metadata = PluginMetadata(
            name="screenshot_tool",
            description="Capture screen regions",
            version="1.0.0",
            category="vision",
            requires_container=True,
        )
    """
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    category: str = "general"
    requires_container: bool = False
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metadata fields."""
        # Validate name format (lowercase, underscores, no spaces)
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Plugin name '{self.name}' must be alphanumeric with underscores only"
            )
        if self.name != self.name.lower():
            raise ValueError(
                f"Plugin name '{self.name}' must be lowercase"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "category": self.category,
            "requires_container": self.requires_container,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            category=data.get("category", "general"),
            requires_container=data.get("requires_container", False),
            tags=data.get("tags", []),
        )


# =============================================================================
# Plugin Interface (Protocol/ABC)
# =============================================================================

class PluginInterface(ABC):
    """Abstract base class defining the plugin contract.
    
    All plugins must implement this interface to be loadable by the system.
    The interface is intentionally minimal to reduce boilerplate.
    
    Required:
        metadata: Property returning PluginMetadata
        get_tools(): Method returning list of callable tools
    
    Optional:
        initialize(): Called after loading, before tool registration
        shutdown(): Called when plugin is unloaded
        on_container_inject(): Called when ServiceContainer is provided
    
    Example:
        class MyPlugin(PluginInterface):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_plugin",
                    description="Example plugin",
                    version="1.0.0",
                )
            
            def get_tools(self) -> List[Callable]:
                return [self.my_tool]
            
            def my_tool(self, param: str) -> str:
                '''Do something with param.'''
                return f"Result: {param}"
    """
    
    # Optional: Container reference (injected if requires_container=True)
    _container: Optional["ServiceContainer"] = None
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.
        
        Returns:
            PluginMetadata instance with name, description, version, etc.
        """
        ...
    
    @abstractmethod
    def get_tools(self) -> List[Callable[..., Any]]:
        """Return list of tool functions to register.
        
        Each function should:
        - Have a docstring (used as tool description)
        - Have type hints for parameters
        - Return a string result
        
        Returns:
            List of callable functions to expose as TARA tools.
        """
        ...
    
    def initialize(self) -> None:
        """Called after plugin is loaded, before tools are registered.
        
        Override to perform setup tasks like loading config files,
        establishing connections, or validating dependencies.
        
        Default implementation does nothing.
        """
        pass
    
    def shutdown(self) -> None:
        """Called when plugin is being unloaded.
        
        Override to perform cleanup tasks like closing connections,
        saving state, or releasing resources.
        
        Default implementation does nothing.
        """
        pass
    
    def on_container_inject(self, container: "ServiceContainer") -> None:
        """Called when ServiceContainer is injected.
        
        Only called if metadata.requires_container is True.
        Override to store container reference for tool access.
        
        Args:
            container: The global ServiceContainer instance.
            
        Default implementation stores container in _container attribute.
        """
        self._container = container
    
    @property
    def container(self) -> Optional["ServiceContainer"]:
        """Get injected ServiceContainer (if available).
        
        Returns:
            ServiceContainer or None if not injected.
        """
        return self._container
    
    def __repr__(self) -> str:
        return f"<Plugin: {self.metadata.name} v{self.metadata.version}>"


# =============================================================================
# Exception Hierarchy
# =============================================================================

class PluginError(Exception):
    """Base exception for all plugin-related errors.
    
    Attributes:
        plugin_name: Name of the plugin that caused the error (if known).
        message: Human-readable error description.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.plugin_name = plugin_name
        self.cause = cause
        
        # Build full message
        full_message = message
        if plugin_name:
            full_message = f"[{plugin_name}] {message}"
        if cause:
            full_message = f"{full_message} (caused by: {cause})"
        
        super().__init__(full_message)


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load.
    
    Common causes:
    - Missing required metadata
    - Import errors in plugin code
    - Invalid plugin structure
    - Dependency not available
    """
    pass


class PluginExecutionError(PluginError):
    """Raised when a plugin tool fails during execution.
    
    Common causes:
    - Runtime exception in tool function
    - Invalid arguments passed to tool
    - External service unavailable
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        self.tool_name = tool_name
        super().__init__(message, plugin_name, cause)


class PluginValidationError(PluginError):
    """Raised when plugin metadata validation fails.
    
    Common causes:
    - Invalid plugin name format
    - Missing required metadata fields
    - Version string malformed
    """
    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core Types
    "PluginInterface",
    "PluginMetadata",
    "PluginType",
    # Exceptions
    "PluginError",
    "PluginLoadError",
    "PluginExecutionError",
    "PluginValidationError",
]
