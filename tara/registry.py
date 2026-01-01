"""TARA Tool Registry - Dynamic Tool Discovery Engine.

Scans the tara/units/ directory to automatically discover and load
tools decorated with @tara_tool.

Usage:
    from tara.registry import ToolRegistry
    
    registry = ToolRegistry()
    registry.discover_tools()
    
    # Get a specific tool
    tool = registry.get_tool("system_stats")
    result = tool()
    
    # Build prompt for LLM
    prompt = registry.build_system_prompt()
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .protocols import TaraTool

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# Tool Registry Class
# =============================================================================

class ToolRegistry:
    """Dynamic tool discovery and management for TARA.
    
    Scans the units/ directory for modules containing @tara_tool
    decorated functions and registers them for execution.
    
    Attributes:
        tools: Dictionary mapping tool names to TaraTool objects.
        categories: Dictionary grouping tools by category.
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self.tools: Dict[str, TaraTool] = {}
        self.categories: Dict[str, List[TaraTool]] = {}
        self._discovered = False
    
    def discover_tools(self, units_dir: Optional[str] = None) -> int:
        """Scan units directory and register all decorated tools.
        
        Args:
            units_dir: Path to units directory (default: tara/units/).
            
        Returns:
            Number of tools discovered.
        """
        if units_dir is None:
            # Default to tara/units/ relative to this file
            base_dir = Path(__file__).parent
            units_path = base_dir / "units"
        else:
            units_path = Path(units_dir)
        
        # Ensure units directory exists
        if not units_path.exists():
            units_path.mkdir(parents=True, exist_ok=True)
            # Create __init__.py
            init_file = units_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""TARA Units - Auto-discovered tool modules."""\n')
            logger.info("Created units directory: %s", units_path)
            return 0
        
        discovered_count = 0
        
        # Walk through units package
        package_name = "tara.units"
        
        try:
            # Import the units package first
            units_package = importlib.import_module(package_name)
            package_path = getattr(units_package, "__path__", [str(units_path)])
        except ImportError as e:
            logger.warning("Could not import units package: %s", e)
            package_path = [str(units_path)]
        
        # Iterate through all modules in units/
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if modname.startswith("_"):
                continue  # Skip private modules
            
            full_module_name = f"{package_name}.{modname}"
            
            try:
                # Import the module
                module = importlib.import_module(full_module_name)
                
                # Scan all attributes for decorated functions
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    
                    attr = getattr(module, attr_name)
                    
                    # Check if it has our metadata
                    if hasattr(attr, "_tool_metadata"):
                        tool_meta: TaraTool = attr._tool_metadata
                        self._register_tool(tool_meta)
                        discovered_count += 1
                        logger.debug("Discovered tool: %s (category: %s)", 
                                   tool_meta.name, tool_meta.category)
                
            except Exception as e:
                # Don't let one broken module break everything
                logger.warning("Failed to import module %s: %s", full_module_name, e)
                continue
        
        self._discovered = True
        logger.info("🛠️ Discovered %d tools from %s", discovered_count, units_path)
        return discovered_count
    
    def _register_tool(self, tool: TaraTool) -> None:
        """Register a tool in the registry.
        
        Args:
            tool: TaraTool object to register.
        """
        # Store by name
        self.tools[tool.name] = tool
        
        # Group by category
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        self.categories[tool.category].append(tool)
    
    def register_manual(
        self,
        name: str,
        func: Callable,
        category: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually register a tool (for legacy compatibility).
        
        Args:
            name: Tool name.
            func: Callable function.
            category: Tool category.
            description: Tool description.
            parameters: Optional parameter schema.
        """
        tool = TaraTool(
            name=name,
            func=func,
            category=category,
            description=description,
            parameters=parameters or {},
        )
        self._register_tool(tool)
    
    def get_tool(self, name: str) -> Optional[TaraTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name.
            
        Returns:
            TaraTool object or None if not found.
        """
        return self.tools.get(name)
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments.
        
        Args:
            name: Tool name.
            **kwargs: Arguments to pass to the tool.
            
        Returns:
            Tool execution result.
            
        Raises:
            KeyError: If tool not found.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool.func(**kwargs)
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self.tools.keys())
    
    def list_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[TaraTool]:
        """Get all tools in a category.
        
        Args:
            category: Category name.
            
        Returns:
            List of TaraTool objects in that category.
        """
        return self.categories.get(category, [])
    
    def build_system_prompt(self) -> str:
        """Generate formatted prompt section listing all tools.
        
        Returns:
            Markdown-formatted string for LLM system prompt.
        """
        if not self.tools:
            return "No tools available."
        
        lines = ["## Available Tools\n"]
        
        # Group by category
        for category in sorted(self.categories.keys()):
            tools = self.categories[category]
            
            # Category header
            category_title = category.replace("_", " ").title()
            lines.append(f"### {category_title} Tools\n")
            
            # List tools
            for tool in tools:
                lines.append(tool.to_prompt_entry())
            
            lines.append("")  # Blank line between categories
        
        # Add usage instructions
        lines.append("## How To Use Tools\n")
        lines.append("When you need to use a tool, respond with EXACTLY this format:")
        lines.append("```")
        lines.append("TOOL: <tool_name>")
        lines.append("ARGS: {\"param\": \"value\"}")
        lines.append("```\n")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self.tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self.tools


# =============================================================================
# Global Registry Instance
# =============================================================================

# Singleton instance for convenience
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ToolRegistry",
    "get_registry",
]
