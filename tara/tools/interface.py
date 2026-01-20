"""
TARA 2.0 Dynamic Tool Interface.

This module provides automatic discovery and loading of TARA tools as LangChain
StructuredTools. Any Python file in tara/tools/ is auto-scanned for public
functions and wrapped as tools.

Architecture:
    1. Scan tara/tools/*.py using pkgutil
    2. Import each module dynamically
    3. Extract public functions (not starting with _)
    4. Wrap as StructuredTool using docstrings
    
Usage:
    from tara.tools.interface import get_tara_tools
    
    tools = get_tara_tools()  # Auto-discovers all tool functions
    llm_with_tools = llm.bind_tools(tools)

Plug & Play:
    To add a new tool, simply create a new file in tara/tools/ with public
    functions. They will be automatically discovered on next get_tara_tools() call.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List, Optional, Set

from langchain_core.tools import StructuredTool

from core.logger import setup_logger

logger = setup_logger("TARA.Interface")


# =============================================================================
# Configuration
# =============================================================================

# Modules to skip during auto-discovery
SKIP_MODULES: Set[str] = {
    "interface",      # This file
    "__init__",       # Package init
    "registry",       # Shared state (not tools)
}

# Functions to skip (common helpers that shouldn't be tools)
SKIP_FUNCTIONS: Set[str] = {
    "get_registry",
    "get_browser_manager",
    "get_tara_tools",
    "get_tool_by_name",
    "list_tool_names",
}


# =============================================================================
# Dynamic Tool Discovery
# =============================================================================

def _is_valid_tool_function(func: Callable, module_name: str) -> bool:
    """
    Check if a function should be exposed as a tool.
    
    Criteria:
    - Must be a function (not class, not method)
    - Must be defined in the module (not imported)
    - Must not start with underscore (private)
    - Must not be in skip list
    - Must have a docstring
    """
    # Must be a function
    if not inspect.isfunction(func):
        return False
    
    # Must be defined in this module (not imported)
    func_module = getattr(func, "__module__", "")
    if not func_module.endswith(module_name):
        return False
    
    # Must not be private
    if func.__name__.startswith("_"):
        return False
    
    # Must not be in skip list
    if func.__name__ in SKIP_FUNCTIONS:
        return False
    
    # Must have a docstring (for tool description)
    if not func.__doc__:
        logger.warning(f"Skipping {func.__name__}: no docstring")
        return False
    
    return True


def _create_tool_from_function(func: Callable) -> Optional[StructuredTool]:
    """
    Wrap a function as a LangChain StructuredTool.
    
    
    For async functions, we use coroutine= parameter so LangChain
    knows to await the function. For sync functions, we use func=.
    """
    try:
        # Extract first line of docstring as description
        docstring = func.__doc__ or ""
        first_line = docstring.strip().split("\n")[0].strip()
        description = first_line if first_line else f"Execute {func.__name__}"
        
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            # ASYNC FUNCTIONS: Use coroutine= parameter
            # This tells LangChain to await the function when called
            logger.debug(f"Registering async tool: {func.__name__}")
            tool = StructuredTool.from_function(
                coroutine=func,  # KEY FIX: coroutine= for async
                name=func.__name__,
                description=description,
            )
        else:
            # SYNC FUNCTIONS: Use func= parameter
            tool = StructuredTool.from_function(
                func=func,
                name=func.__name__,
                description=description,
            )
        
        return tool
        
    except Exception as e:
        logger.error(f"Failed to create tool from {func.__name__}: {e}")
        return None


def _discover_module_tools(module_name: str) -> List[StructuredTool]:
    """
    Dynamically import a module and extract its tool functions.
    
    Args:
        module_name: Name of module to import (e.g., "app_launcher")
        
    Returns:
        List of StructuredTools from the module.
    """
    tools: List[StructuredTool] = []
    full_module_name = f"tara.tools.{module_name}"
    
    try:
        # Dynamic import
        module = importlib.import_module(full_module_name)
        
        # Get all public attributes
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            
            attr = getattr(module, attr_name)
            
            if _is_valid_tool_function(attr, module_name):
                tool = _create_tool_from_function(attr)
                if tool:
                    tools.append(tool)
                    logger.debug(f"Loaded tool: {attr_name} from {module_name}")
        
        if tools:
            logger.info(f"Loaded {len(tools)} tools from {module_name}")
            
    except ImportError as e:
        logger.error(f"Failed to import {full_module_name}: {e}")
    except Exception as e:
        logger.error(f"Error scanning {full_module_name}: {e}")
    
    return tools


# =============================================================================
# Public API
# =============================================================================

_cached_tools: Optional[List[StructuredTool]] = None


def get_tara_tools(refresh: bool = False) -> List[StructuredTool]:
    """
    Get all TARA tools as LangChain StructuredTools.
    
    Auto-discovers and loads all public functions from tara/tools/*.py modules.
    Results are cached for performance; use refresh=True to reload.
    
    Args:
        refresh: Force re-scan of tool modules (default: False).
        
    Returns:
        List of StructuredTool objects ready for agent binding.
        
    Example:
        >>> tools = get_tara_tools()
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    global _cached_tools
    
    if _cached_tools is not None and not refresh:
        return _cached_tools
    
    logger.info("Discovering TARA tools...")
    
    all_tools: List[StructuredTool] = []
    
    # Get the tara.tools package path
    try:
        import tara.tools as tools_package
        package_path = tools_package.__path__
    except (ImportError, AttributeError) as e:
        logger.error(f"Cannot access tara.tools package: {e}")
        return []
    
    # Iterate all modules in the package
    for importer, module_name, is_pkg in pkgutil.iter_modules(package_path):
        # Skip non-tool modules
        if module_name in SKIP_MODULES:
            continue
        
        # Skip sub-packages (only want .py files)
        if is_pkg:
            continue
        
        # Discover tools from this module
        module_tools = _discover_module_tools(module_name)
        all_tools.extend(module_tools)
    
    # Cache results
    _cached_tools = all_tools
    
    logger.info(f"TARA toolset ready: {len(all_tools)} tools discovered")
    return all_tools


def get_tool_by_name(name: str) -> Optional[StructuredTool]:
    """
    Get a specific tool by its name.
    
    Args:
        name: Tool name to find.
        
    Returns:
        StructuredTool if found, None otherwise.
    """
    for tool in get_tara_tools():
        if tool.name == name:
            return tool
    return None


def list_tool_names() -> List[str]:
    """
    Get list of all available tool names.
    
    Returns:
        List of tool name strings.
    """
    return [tool.name for tool in get_tara_tools()]


def get_tools_by_category() -> Dict[str, List[str]]:
    """
    Get tools organized by their source module.
    
    Returns:
        Dict mapping module names to tool name lists.
    """
    categories: Dict[str, List[str]] = {}
    
    try:
        import tara.tools as tools_package
        package_path = tools_package.__path__
    except (ImportError, AttributeError):
        return {}
    
    for importer, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if module_name in SKIP_MODULES or is_pkg:
            continue
        
        full_name = f"tara.tools.{module_name}"
        try:
            module = importlib.import_module(full_name)
            tool_names = []
            
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if _is_valid_tool_function(attr, module_name):
                    tool_names.append(attr_name)
            
            if tool_names:
                categories[module_name] = sorted(tool_names)
                
        except Exception:
            pass
    
    return categories


def clear_cache() -> None:
    """Clear the cached tools list to force re-discovery."""
    global _cached_tools
    _cached_tools = None
    logger.info("Tool cache cleared")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_tara_tools",
    "get_tool_by_name",
    "list_tool_names",
    "get_tools_by_category",
    "clear_cache",
]
