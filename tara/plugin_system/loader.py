"""TARA Plugin System - Plugin Loader.

VERSION: 3.0.0

This module provides hot-swappable plugin discovery and loading. Plugins are
automatically discovered from the `plugins/` directory and can be reloaded
at runtime without system restart.

Discovery Modes:
    1. PYTHON_SCRIPT: Single .py file with public functions
    2. PYTHON_PACKAGE: Directory with __init__.py exposing PluginInterface

Directory Structure:
    N.I.A/
    └── plugins/                    # User plugins directory
        ├── my_tool.py              # Simple script plugin
        └── advanced_plugin/        # Package plugin
            ├── __init__.py         # Must export 'plugin' or 'Plugin' class
            └── tools.py

Usage:
    from tara.plugin_system.loader import get_plugin_loader
    
    loader = get_plugin_loader()
    tools = loader.discover_tools()  # Returns List[StructuredTool]
    
    # Hot-reload a specific plugin
    loader.reload_plugin("my_tool")
    
    # Force full re-discovery
    loader.refresh()
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from core.logger import setup_logger
from core.config import get_settings

from .base import (
    PluginInterface,
    PluginMetadata,
    PluginType,
    PluginError,
    PluginLoadError,
    PluginExecutionError,
    PluginValidationError,
)

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool
    from core.container import ServiceContainer

logger = setup_logger("TARA.Plugins")


# =============================================================================
# Configuration
# =============================================================================

import json

def _load_config() -> dict:
    """Load Plugin configuration from JSON."""
    config_path = Path(__file__).parent.parent.parent / "config" / "tara" / "plugins.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load plugins.json: {e}")
        return {}

_PLUGIN_CONFIG = _load_config()

# Default plugins directory (relative to project root)
DEFAULT_PLUGINS_DIR = _PLUGIN_CONFIG.get("DEFAULT_PLUGINS_DIR", "plugins")

# Files/folders to skip during discovery
SKIP_NAMES: Set[str] = {
    "__pycache__",
    "__init__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
}

# Maximum plugin loading time before warning (seconds)
LOAD_TIMEOUT_WARNING = _PLUGIN_CONFIG.get("LOAD_TIMEOUT_WARNING", 5.0)


# =============================================================================
# Loaded Plugin Container
# =============================================================================

@dataclass
class LoadedPlugin:
    """Container for a loaded plugin instance.
    
    Attributes:
        instance: The PluginInterface implementation instance.
        source_path: Path to the plugin file/directory.
        plugin_type: How the plugin was loaded (script/package).
        module_name: Python module name for reloading.
        tools: Cached list of StructuredTool objects.
    """
    instance: PluginInterface
    source_path: Path
    plugin_type: PluginType
    module_name: str
    tools: List["StructuredTool"] = field(default_factory=list)
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.instance.metadata
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.metadata.name


# =============================================================================
# Plugin Loader
# =============================================================================

class PluginLoader:
    """Dynamic plugin discovery and loading system.
    
    Scans a directory for plugins, loads them, and converts their tools
    to LangChain StructuredTool objects for use with TARA.
    
    Features:
        - Auto-discovery of .py files and packages
        - Hot-reload without restart
        - Container injection for stateful plugins
        - Error isolation (bad plugin doesn't crash system)
    
    Example:
        loader = PluginLoader()
        tools = loader.discover_tools()
        
        # Later, reload a specific plugin
        loader.reload_plugin("my_plugin")
    """
    
    def __init__(
        self,
        plugins_dir: Optional[Path] = None,
        container: Optional["ServiceContainer"] = None,
    ):
        """Initialize the plugin loader.
        
        Args:
            plugins_dir: Path to plugins directory. Defaults to PROJECT/plugins/
            container: Optional ServiceContainer for injection.
        """
        self._plugins_dir = plugins_dir or self._get_default_plugins_dir()
        self._container = container
        self._loaded_plugins: Dict[str, LoadedPlugin] = {}
        self._tool_cache: Optional[List["StructuredTool"]] = None
        
        # Thread lock for safe concurrent access (hot-reload)
        self._lock = threading.Lock()
        
        logger.info(f"PluginLoader initialized (dir: {self._plugins_dir})")
    
    def _get_default_plugins_dir(self) -> Path:
        """Get default plugins directory path."""
        settings = get_settings()
        project_root = Path(settings.PROJECT_ROOT) if hasattr(settings, "PROJECT_ROOT") else Path.cwd()
        return project_root / DEFAULT_PLUGINS_DIR
    
    @property
    def plugins_dir(self) -> Path:
        """Get the plugins directory path."""
        return self._plugins_dir
    
    @property
    def loaded_plugins(self) -> Dict[str, LoadedPlugin]:
        """Get currently loaded plugins."""
        return self._loaded_plugins.copy()
    
    # =========================================================================
    # Discovery
    # =========================================================================
    
    def discover(self) -> List[LoadedPlugin]:
        """Discover and load all plugins from the plugins directory.
        
        Returns:
            List of successfully loaded plugins.
            
        Note:
            Plugins that fail to load are logged but don't raise exceptions.
        """
        if not self._plugins_dir.exists():
            logger.info(f"Plugins directory not found: {self._plugins_dir}")
            logger.info("Creating plugins directory...")
            self._plugins_dir.mkdir(parents=True, exist_ok=True)
            self._create_example_plugin()
            return []
        
        loaded: List[LoadedPlugin] = []
        
        for item in self._plugins_dir.iterdir():
            # Skip hidden and special files
            if item.name.startswith(".") or item.name in SKIP_NAMES:
                continue
            
            try:
                plugin = self._load_item(item)
                if plugin:
                    self._loaded_plugins[plugin.name] = plugin
                    loaded.append(plugin)
                    logger.info(f"Loaded plugin: {plugin.name} v{plugin.metadata.version}")
            except PluginError as e:
                logger.error(f"Failed to load plugin from {item}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading {item}: {e}", exc_info=True)
        
        if loaded:
            logger.info(f"Discovered {len(loaded)} plugin(s)")
        
        return loaded
    
    def _load_item(self, path: Path) -> Optional[LoadedPlugin]:
        """Load a plugin from a file or directory.
        
        Args:
            path: Path to .py file or package directory.
            
        Returns:
            LoadedPlugin if successful, None if not a valid plugin.
        """
        if path.is_file() and path.suffix == ".py":
            return self._load_script_plugin(path)
        elif path.is_dir() and (path / "__init__.py").exists():
            return self._load_package_plugin(path)
        else:
            logger.debug(f"Skipping non-plugin: {path}")
            return None
    
    def _load_script_plugin(self, path: Path) -> LoadedPlugin:
        """Load a single-file Python plugin.
        
        Script plugins can either:
        1. Define a 'Plugin' class implementing PluginInterface
        2. Define public functions directly (auto-wrapped)
        
        Args:
            path: Path to .py file.
            
        Returns:
            LoadedPlugin instance.
        """
        module_name = f"plugins.{path.stem}"
        
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot load spec from {path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise PluginLoadError(f"Error executing {path}", cause=e)
        
        # Look for Plugin class or create wrapper
        plugin_instance = self._extract_plugin_instance(module, path)
        
        # Inject container if needed
        if plugin_instance.metadata.requires_container and self._container:
            plugin_instance.on_container_inject(self._container)
        
        # Initialize plugin
        plugin_instance.initialize()
        
        # Convert tools to StructuredTool
        tools = self._convert_tools(plugin_instance)
        
        return LoadedPlugin(
            instance=plugin_instance,
            source_path=path,
            plugin_type=PluginType.PYTHON_SCRIPT,
            module_name=module_name,
            tools=tools,
        )
    
    def _load_package_plugin(self, path: Path) -> LoadedPlugin:
        """Load a package plugin (directory with __init__.py).
        
        Package plugins must export a 'Plugin' class or 'plugin' instance
        implementing PluginInterface.
        
        Args:
            path: Path to plugin directory.
            
        Returns:
            LoadedPlugin instance.
        """
        module_name = f"plugins.{path.name}"
        
        # Add parent to path if needed
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        
        try:
            # Import the package
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
        except Exception as e:
            raise PluginLoadError(f"Error importing {path}", cause=e)
        
        # Extract plugin instance
        plugin_instance = self._extract_plugin_instance(module, path)
        
        # Inject container if needed
        if plugin_instance.metadata.requires_container and self._container:
            plugin_instance.on_container_inject(self._container)
        
        # Initialize plugin
        plugin_instance.initialize()
        
        # Convert tools to StructuredTool
        tools = self._convert_tools(plugin_instance)
        
        return LoadedPlugin(
            instance=plugin_instance,
            source_path=path,
            plugin_type=PluginType.PYTHON_PACKAGE,
            module_name=module_name,
            tools=tools,
        )
    
    def _extract_plugin_instance(
        self,
        module: Any,
        path: Path,
    ) -> PluginInterface:
        """Extract or create a PluginInterface instance from a module.
        
        Priority:
        1. 'plugin' instance (pre-instantiated)
        2. 'Plugin' class (instantiate it)
        3. Auto-wrap public functions as SimplePlugin
        """
        # Check for 'plugin' instance
        if hasattr(module, "plugin"):
            instance = module.plugin
            if isinstance(instance, PluginInterface):
                return instance
        
        # Check for 'Plugin' class
        if hasattr(module, "Plugin"):
            cls = module.Plugin
            if inspect.isclass(cls) and issubclass(cls, PluginInterface):
                return cls()
        
        # Auto-wrap: Find all public functions
        functions = self._extract_public_functions(module)
        if not functions:
            raise PluginLoadError(
                f"No Plugin class or public functions found",
                plugin_name=path.stem,
            )
        
        # Create a simple wrapper plugin
        return self._create_simple_plugin(path.stem, functions)
    
    def _extract_public_functions(self, module: Any) -> List[Callable]:
        """Extract public functions from a module."""
        functions = []
        
        for name in dir(module):
            if name.startswith("_"):
                continue
            
            attr = getattr(module, name)
            if not inspect.isfunction(attr):
                continue
            
            # Must be defined in this module
            if getattr(attr, "__module__", "") != module.__name__:
                continue
            
            # Must have a docstring
            if not attr.__doc__:
                logger.warning(f"Skipping {name}: no docstring")
                continue
            
            functions.append(attr)
        
        return functions
    
    def _create_simple_plugin(
        self,
        name: str,
        functions: List[Callable],
    ) -> PluginInterface:
        """Create a simple plugin wrapper for bare functions."""
        
        class SimplePlugin(PluginInterface):
            """Auto-generated plugin wrapper for script functions."""
            
            def __init__(self, plugin_name: str, funcs: List[Callable]):
                self._name = plugin_name
                self._functions = funcs
            
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=self._name,
                    description=f"Auto-loaded plugin from {self._name}.py",
                    version="1.0.0",
                    category="plugins",
                )
            
            def get_tools(self) -> List[Callable]:
                return self._functions
        
        return SimplePlugin(name, functions)
    
    # =========================================================================
    # Tool Conversion
    # =========================================================================
    
    def _convert_tools(self, plugin: PluginInterface) -> List["StructuredTool"]:
        """Convert plugin functions to LangChain StructuredTool objects."""
        from langchain_core.tools import StructuredTool
        
        tools = []
        
        for func in plugin.get_tools():
            try:
                tool = self._create_structured_tool(func, plugin.metadata.name)
                tools.append(tool)
            except Exception as e:
                logger.error(
                    f"Failed to create tool from {func.__name__}: {e}"
                )
        
        return tools
    
    def _create_structured_tool(
        self,
        func: Callable,
        plugin_name: str,
    ) -> "StructuredTool":
        """Create a StructuredTool from a function."""
        from langchain_core.tools import StructuredTool
        
        # Extract description from docstring
        docstring = func.__doc__ or ""
        first_line = docstring.strip().split("\n")[0].strip()
        description = first_line if first_line else f"Execute {func.__name__}"
        
        # Prefix tool name with plugin name for namespacing
        # e.g., "my_plugin__my_function"
        tool_name = f"{plugin_name}__{func.__name__}"
        
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            return StructuredTool.from_function(
                coroutine=func,
                name=tool_name,
                description=description,
            )
        else:
            return StructuredTool.from_function(
                func=func,
                name=tool_name,
                description=description,
            )
    
    # =========================================================================
    # Hot-Reload
    # =========================================================================
    
    def reload_plugin(self, name: str) -> bool:
        """Hot-reload a specific plugin by name.
        
        Args:
            name: Plugin name to reload.
            
        Returns:
            True if reload successful, False otherwise.
        """
        if name not in self._loaded_plugins:
            logger.warning(f"Plugin not found: {name}")
            return False
        
        plugin = self._loaded_plugins[name]
        path = plugin.source_path
        
        # Shutdown old instance
        try:
            plugin.instance.shutdown()
        except Exception as e:
            logger.error(f"Error during plugin shutdown: {e}")
        
        # Remove from sys.modules to force reload
        if plugin.module_name in sys.modules:
            del sys.modules[plugin.module_name]
        
        # Reload
        try:
            new_plugin = self._load_item(path)
            if new_plugin:
                self._loaded_plugins[name] = new_plugin
                self._tool_cache = None  # Invalidate cache
                logger.info(f"Reloaded plugin: {name}")
                return True
        except Exception as e:
            logger.error(f"Failed to reload plugin {name}: {e}")
        
        return False
    
    def reload_plugin_by_path(self, file_path: Path) -> bool:
        """Hot-reload a plugin from file path (used by watchdog).
        
        This method is called by the PluginEventHandler when a file
        is created or modified. It handles both new plugins and reloads.
        
        Args:
            file_path: Path to the .py file that was created/modified.
            
        Returns:
            True if reload successful, False otherwise.
            
        Thread Safety:
            Uses internal lock to prevent race conditions during reload.
            Old plugin is kept if reload fails (SyntaxError, etc).
        """
        file_path = Path(file_path)
        
        # Skip non-Python files
        if file_path.suffix != ".py":
            return False
        
        # Skip __pycache__ and __init__
        if file_path.stem.startswith("__") or file_path.stem in SKIP_NAMES:
            return False
        
        plugin_name = file_path.stem
        
        with self._lock:
            try:
                # Check if plugin exists (reload vs new)
                if plugin_name in self._loaded_plugins:
                    # Existing plugin - shutdown old instance
                    old_plugin = self._loaded_plugins[plugin_name]
                    try:
                        old_plugin.instance.shutdown()
                    except Exception as e:
                        logger.warning(f"Error during {plugin_name} shutdown: {e}")
                    
                    # Remove from sys.modules to force reload
                    if old_plugin.module_name in sys.modules:
                        del sys.modules[old_plugin.module_name]
                
                # Load the plugin (new or reload)
                new_plugin = self._load_item(file_path)
                
                if new_plugin:
                    self._loaded_plugins[plugin_name] = new_plugin
                    self._tool_cache = None  # Invalidate cache
                    
                    # Notify interface to refresh
                    try:
                        from tara.tools.interface import clear_cache
                        clear_cache()
                    except ImportError:
                        pass
                    
                    logger.info(f"✅ Hot-Reloaded Plugin: {plugin_name}")
                    return True
                else:
                    logger.warning(f"⚠️ No valid plugin found in: {file_path}")
                    return False
                    
            except SyntaxError as e:
                # CRITICAL SAFETY: Syntax errors should NOT crash the system
                logger.error(f"❌ Failed to reload: {plugin_name} (SyntaxError: {e.msg} at line {e.lineno})")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to reload: {plugin_name} ({type(e).__name__}: {e})")
                return False
    
    def unload_plugin(self, file_path: Path) -> List[str]:
        """Unload a plugin when its file is deleted.
        
        Args:
            file_path: Path to the .py file that was deleted.
            
        Returns:
            List of tool names that were removed.
            
        Thread Safety:
            Uses internal lock to prevent race conditions.
        """
        file_path = Path(file_path)
        plugin_name = file_path.stem
        removed_tools: List[str] = []
        
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                logger.debug(f"Plugin not loaded, nothing to unload: {plugin_name}")
                return removed_tools
            
            plugin = self._loaded_plugins[plugin_name]
            
            # Collect tool names for removal from interface
            removed_tools = [tool.name for tool in plugin.tools]
            
            # Shutdown plugin
            try:
                plugin.instance.shutdown()
            except Exception as e:
                logger.warning(f"Error during {plugin_name} shutdown: {e}")
            
            # Remove from sys.modules
            if plugin.module_name in sys.modules:
                del sys.modules[plugin.module_name]
            
            # Remove from loaded plugins
            del self._loaded_plugins[plugin_name]
            
            # Invalidate cache
            self._tool_cache = None
            
            # Notify interface to remove specific tools
            try:
                from tara.tools.interface import remove_plugin_tools, clear_cache
                remove_plugin_tools(removed_tools)
                clear_cache()
            except ImportError:
                pass
            
            logger.info(f"🗑️ Unloaded Plugin: {plugin_name} ({len(removed_tools)} tools removed)")
        
        return removed_tools
    
    def refresh(self) -> None:
        """Force re-discovery of all plugins.
        
        Unloads all current plugins and re-scans the plugins directory.
        """
        # Shutdown all plugins
        for plugin in self._loaded_plugins.values():
            try:
                plugin.instance.shutdown()
            except Exception as e:
                logger.error(f"Error during {plugin.name} shutdown: {e}")
        
        # Clear state
        self._loaded_plugins.clear()
        self._tool_cache = None
        
        # Re-discover
        self.discover()
    
    # =========================================================================
    # Tool Access
    # =========================================================================
    
    def discover_tools(self) -> List["StructuredTool"]:
        """Get all tools from all loaded plugins.
        
        This is the main method called by get_tara_tools() to integrate
        plugin tools with the core tool system.
        
        Returns:
            List of StructuredTool objects from all plugins.
        """
        if self._tool_cache is not None:
            return self._tool_cache
        
        # Ensure plugins are discovered
        if not self._loaded_plugins:
            self.discover()
        
        # Collect all tools
        all_tools = []
        for plugin in self._loaded_plugins.values():
            all_tools.extend(plugin.tools)
        
        self._tool_cache = all_tools
        return all_tools
    
    def clear_cache(self) -> None:
        """Clear the tool cache to force reconstruction."""
        self._tool_cache = None
        logger.debug("Plugin tool cache cleared")
    
    # =========================================================================
    # Example Plugin Generator
    # =========================================================================
    
    def _create_example_plugin(self) -> None:
        """Create an example plugin in the plugins directory."""
        example_path = self._plugins_dir / "example_plugin.py"
        
        example_code = '''\
"""Example Plugin - Demonstrates the plugin system.

Drop this file in the plugins/ directory to see it automatically loaded.
Delete or rename to disable.

To create your own plugin:
1. Create a .py file in plugins/
2. Define public functions with docstrings
3. Restart N.I.A. or call plugin reload

Example tool below will be auto-registered as "example_plugin__greet"
"""


def greet(name: str) -> str:
    """Greet someone by name.
    
    Args:
        name: The person's name to greet.
        
    Returns:
        A friendly greeting message.
    """
    return f"Hello, {name}! Welcome to N.I.A. plugins."


def add_numbers(a: int, b: int) -> str:
    """Add two numbers together.
    
    Args:
        a: First number.
        b: Second number.
        
    Returns:
        The sum as a string.
    """
    return f"The sum of {a} and {b} is {a + b}"


# Private functions (prefixed with _) are not exposed as tools
def _helper():
    """This won't be registered as a tool."""
    pass
'''
        
        example_path.write_text(example_code, encoding="utf-8")
        logger.info(f"Created example plugin: {example_path}")


# =============================================================================
# Module-level Singleton
# =============================================================================

_loader: Optional[PluginLoader] = None


def get_plugin_loader(
    plugins_dir: Optional[Path] = None,
    container: Optional["ServiceContainer"] = None,
) -> PluginLoader:
    """Get or create the global PluginLoader singleton.
    
    Args:
        plugins_dir: Override default plugins directory.
        container: ServiceContainer for injection.
        
    Returns:
        The global PluginLoader instance.
    """
    global _loader
    
    if _loader is None:
        _loader = PluginLoader(plugins_dir=plugins_dir, container=container)
    elif container and _loader._container is None:
        _loader._container = container
    
    return _loader


def reset_plugin_loader() -> None:
    """Reset the global PluginLoader (for testing)."""
    global _loader
    if _loader:
        _loader.refresh()
    _loader = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PluginLoader",
    "LoadedPlugin",
    "get_plugin_loader",
    "reset_plugin_loader",
]
