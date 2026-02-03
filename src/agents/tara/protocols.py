"""TARA Protocols - Tool Contract Definitions.

Defines the standardized structure of a "Tool" and provides
the @tara_tool decorator for automatic metadata generation.

Usage:
    from tara.protocols import tara_tool, TaraTool
    
    @tara_tool(name="get_cpu", category="system", description="Get CPU usage")
    def get_cpu_usage() -> str:
        return f"CPU: {psutil.cpu_percent()}%"
"""
from __future__ import annotations

import inspect
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, get_type_hints


# =============================================================================
# TaraTool Data Structure
# =============================================================================

@dataclass
class TaraTool:
    """Standardized tool definition for TARA agent.
    
    Attributes:
        name: Unique tool identifier (e.g., "system_stats")
        func: The callable function to execute
        category: Tool category for grouping (e.g., "system", "web", "desktop")
        description: Human-readable description for LLM prompt
        parameters: JSON schema of function arguments
    """
    name: str
    func: Callable[..., Any]
    category: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct calling of the tool."""
        return self.func(*args, **kwargs)
    
    def to_prompt_entry(self) -> str:
        """Format tool for inclusion in LLM system prompt."""
        params_str = ""
        if self.parameters.get("properties"):
            params = []
            for pname, pinfo in self.parameters["properties"].items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                params.append(f"{pname} ({ptype}): {pdesc}")
            params_str = ", ".join(params)
        
        return f"- **{self.name}**: {self.description}" + (f"\n  Args: {params_str}" if params_str else "")


# =============================================================================
# Parameter Schema Generator
# =============================================================================

def _generate_parameters_schema(func: Callable) -> Dict[str, Any]:
    """Auto-generate JSON schema from function signature and type hints.
    
    Args:
        func: The function to analyze.
        
    Returns:
        JSON schema dict with properties and required fields.
    """
    sig = inspect.signature(func)
    
    # Try to get type hints (may fail on some functions)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    
    properties: Dict[str, Any] = {}
    required: List[str] = []
    
    for param_name, param in sig.parameters.items():
        # Skip self, cls, *args, **kwargs
        if param_name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        
        # Get type hint
        param_type = hints.get(param_name, Any)
        
        # Convert Python type to JSON schema type
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        json_type = "string"  # Default
        if param_type in type_mapping:
            json_type = type_mapping[param_type]
        elif hasattr(param_type, "__origin__"):
            # Handle Optional, List, etc.
            origin = getattr(param_type, "__origin__", None)
            if origin in type_mapping:
                json_type = type_mapping[origin]
        
        # Build property schema
        prop_schema: Dict[str, Any] = {"type": json_type}
        
        # Check for default value
        if param.default is not param.empty:
            prop_schema["default"] = param.default
        else:
            required.append(param_name)
        
        properties[param_name] = prop_schema
    
    schema = {
        "type": "object",
        "properties": properties,
    }
    
    if required:
        schema["required"] = required
    
    return schema


# =============================================================================
# @tara_tool Decorator
# =============================================================================

def tara_tool(
    name: str,
    category: str,
    description: str,
) -> Callable[[Callable], Callable]:
    """Decorator to register a function as a TARA tool.
    
    Automatically generates parameter schema from type hints and
    attaches metadata for the ToolRegistry to discover.
    
    Args:
        name: Unique tool identifier.
        category: Tool category (e.g., "system", "web", "desktop").
        description: Human-readable description for LLM prompt.
        
    Returns:
        Decorated function with _tool_metadata attribute.
        
    Example:
        @tara_tool(
            name="open_app",
            category="desktop",
            description="Open an application on the desktop"
        )
        def open_app(app_name: str) -> str:
            # Implementation
            return f"Opened {app_name}"
    """
    def decorator(func: Callable) -> Callable:
        # Generate parameter schema from signature
        parameters = _generate_parameters_schema(func)
        
        # Create TaraTool metadata object
        tool_meta = TaraTool(
            name=name,
            func=func,
            category=category,
            description=description,
            parameters=parameters,
        )
        
        # Attach metadata to function for registry discovery
        func._tool_metadata = tool_meta  # type: ignore
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Also attach to wrapper
        wrapper._tool_metadata = tool_meta  # type: ignore
        
        return wrapper
    
    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TaraTool",
    "tara_tool",
]
