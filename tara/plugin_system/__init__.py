"""TARA Plugin System - Hot-Swappable Tool Ecosystem.

VERSION: 3.0.0

This package provides the infrastructure for dynamically loading, managing,
and executing plugins that extend TARA's capabilities without system restart.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Plugin System v3.0                        │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │    base     │  │   loader    │  │     registry        │  │
    │  │ (Protocol)  │  │ (Discovery) │  │ (Runtime Tracking)  │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Modules:
    base: Core definitions (PluginInterface, PluginType, PluginError)
    loader: Plugin discovery and loading (Phase 1)
    registry: Runtime plugin management (Phase 2)

Usage:
    from tara.plugin_system import PluginInterface, PluginType, PluginError
    from tara.plugin_system import get_plugin_loader
    
    # Auto-discover plugins
    loader = get_plugin_loader()
    tools = loader.discover_tools()
"""
from .base import (
    PluginInterface,
    PluginMetadata,
    PluginType,
    PluginError,
    PluginLoadError,
    PluginExecutionError,
    PluginValidationError,
)

from .loader import (
    PluginLoader,
    LoadedPlugin,
    get_plugin_loader,
    reset_plugin_loader,
)

__all__ = [
    # Base types
    "PluginInterface",
    "PluginMetadata",
    "PluginType",
    # Exceptions
    "PluginError",
    "PluginLoadError",
    "PluginExecutionError",
    "PluginValidationError",
    # Loader
    "PluginLoader",
    "LoadedPlugin",
    "get_plugin_loader",
    "reset_plugin_loader",
]

__version__ = "3.0.0"
