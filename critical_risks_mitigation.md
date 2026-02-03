# N.I.A. v4.0.0 - Critical Implementation Risks & Solutions

## ⚠️ HIGH-RISK AREAS

This document addresses the three critical failure points in large-scale refactoring:
1. **Circular Dependency Hell** (Phase 3)
2. **Plugin Ecosystem Breaking** (Phase 5)
3. **Platform Abstraction Weakness** (Architecture)

---

## RISK #1: Circular Dependency Trap

### The Problem

Moving to nested `src/` structure creates MORE import depth, increasing circular dependency risk:

```python
# CIRCULAR DEPENDENCY EXAMPLE

# File: src/core/registry.py
from nia.src.agents.nia.agent import SupervisorAgent  # Import agent

# File: src/agents/nia/agent.py
from nia.src.capabilities.desktop.windows import WindowManagement  # Import capability

# File: src/capabilities/desktop/windows.py
from nia.src.core.events import EventBus  # Import core
from nia.src.agents.tara.security import WardenService  # Import agent - CIRCULAR!

# File: src/agents/tara/security.py
from nia.src.core.registry import get_service  # Back to registry - BOOM! 💥
```

### Solution: Dependency Injection First Strategy

#### Step 1: Establish Clear Layer Rules

```
DEPENDENCY FLOW (One direction only):
capabilities → agents → core
     ↑           ↑        ↑
     └───────────┴────────┘
     ALL get dependencies via ServiceRegistry
```

**Golden Rule**: Never import "upward" in the dependency chain. Always use ServiceRegistry.

#### Step 2: Enhanced ServiceRegistry

```python
# File: src/core/registry.py
"""Service registry with lazy loading and circular dependency prevention."""

from typing import Any, Callable, TypeVar, Generic, Dict, Type
from collections import defaultdict
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceRegistry:
    """
    Centralized dependency injection container.
    
    Prevents circular dependencies by:
    1. Lazy initialization
    2. Dependency resolution at runtime
    3. Clear registration order
    """
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._initializing: set = set()  # Track initialization to detect cycles
        self._dependencies: Dict[str, list] = defaultdict(list)
        
    def register(
        self,
        name: str,
        factory: Callable[[], T],
        dependencies: list[str] = None,
        singleton: bool = True
    ) -> None:
        """
        Register a service factory.
        
        Args:
            name: Service name
            factory: Factory function that creates the service
            dependencies: List of service names this depends on
            singleton: If True, only create once and reuse
        """
        if name in self._factories:
            logger.warning(f"Service '{name}' already registered, overwriting")
        
        self._factories[name] = factory
        self._dependencies[name] = dependencies or []
        
        if not singleton:
            # Non-singleton services create new instance each time
            self._instances[name] = None
    
    def get(self, name: str) -> Any:
        """
        Get service instance, creating if needed.
        
        Detects circular dependencies during initialization.
        """
        # Return existing singleton
        if name in self._instances and self._instances[name] is not None:
            return self._instances[name]
        
        # Check factory exists
        if name not in self._factories:
            available = ', '.join(self._factories.keys())
            raise KeyError(
                f"Service '{name}' not registered. "
                f"Available services: {available}"
            )
        
        # Detect circular dependency
        if name in self._initializing:
            chain = ' → '.join(self._initializing) + f' → {name}'
            raise RuntimeError(
                f"Circular dependency detected: {chain}\n"
                f"Fix: Use dependency injection instead of direct imports"
            )
        
        # Initialize dependencies first
        self._initializing.add(name)
        try:
            # Resolve dependencies
            deps = {}
            for dep_name in self._dependencies[name]:
                deps[dep_name] = self.get(dep_name)
            
            # Create instance (pass dependencies to factory)
            factory = self._factories[name]
            
            # Smart dependency injection
            sig = inspect.signature(factory)
            if sig.parameters:
                # Factory expects dependencies
                instance = factory(**{k: v for k, v in deps.items() if k in sig.parameters})
            else:
                # Factory takes no args
                instance = factory()
            
            self._instances[name] = instance
            logger.info(f"✓ Initialized service: {name}")
            return instance
            
        finally:
            self._initializing.discard(name)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all registered services."""
        return {name: self.get(name) for name in self._factories}
    
    def clear(self) -> None:
        """Clear all instances (for testing)."""
        self._instances.clear()
        self._initializing.clear()

# Global singleton
_registry = ServiceRegistry()

def get_service(name: str) -> Any:
    """Get service from global registry."""
    return _registry.get(name)

def register_service(name: str, factory: Callable, dependencies: list[str] = None):
    """Register service in global registry."""
    _registry.register(name, factory, dependencies)

def get_registry() -> ServiceRegistry:
    """Get global registry instance."""
    return _registry
```

#### Step 3: Proper Service Registration

```python
# File: main.py
"""Application bootstrap with correct dependency order."""

from nia.src.core.registry import register_service, get_service
from nia.src.core.logger import get_logger
from config.loader import get_settings

logger = get_logger(__name__)

def bootstrap_services():
    """
    Register all services in correct dependency order.
    
    Order matters! Dependencies must be registered before dependents.
    """
    settings = get_settings()
    
    # LAYER 1: Core Infrastructure (no dependencies)
    register_service(
        "logger",
        lambda: get_logger("nia"),
        dependencies=[]
    )
    
    register_service(
        "event_bus",
        lambda: EventBus(),
        dependencies=["logger"]
    )
    
    register_service(
        "memory_manager",
        lambda logger: MemoryManager(logger),
        dependencies=["logger"]
    )
    
    register_service(
        "config",
        lambda: settings,
        dependencies=[]
    )
    
    # LAYER 2: Platform Services
    register_service(
        "platform_driver",
        lambda: create_platform_driver(),  # Factory from src/core/platform/
        dependencies=["logger"]
    )
    
    # LAYER 3: Capabilities (depend on core + platform)
    register_service(
        "window_management",
        lambda driver: WindowManagement(driver),
        dependencies=["platform_driver"]
    )
    
    register_service(
        "file_operations",
        lambda logger: FileOperations(logger),
        dependencies=["logger"]
    )
    
    # ... register all capabilities
    
    # LAYER 4: Security Services
    register_service(
        "warden",
        lambda logger, config: WardenService(logger, config),
        dependencies=["logger", "config"]
    )
    
    # LAYER 5: Agents (depend on everything)
    register_service(
        "supervisor_agent",
        lambda event_bus, memory, warden: SupervisorAgent(
            event_bus=event_bus,
            memory=memory,
            warden=warden,
            config=settings.nia
        ),
        dependencies=["event_bus", "memory_manager", "warden"]
    )
    
    register_service(
        "tara_agent",
        lambda event_bus, warden: TARAAgent(event_bus, warden),
        dependencies=["event_bus", "warden"]
    )
    
    # ... register other agents
    
    logger.info("✓ All services registered")

async def main():
    """Main entry point."""
    
    # Bootstrap dependency injection
    bootstrap_services()
    
    # Get orchestrator (will auto-initialize all dependencies)
    orchestrator = get_service("supervisor_agent")
    
    # Run
    await orchestrator.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

#### Step 4: Using ServiceRegistry in Code

```python
# File: src/capabilities/desktop/windows.py
"""Window management capability - NO DIRECT IMPORTS OF AGENTS."""

from nia.src.capabilities.base import BaseCapability, capability
from nia.src.core.registry import get_service  # Only import registry
from nia.src.core.events import EventBus  # OK - core is lower layer

class WindowManagement(BaseCapability):
    """Window control operations."""
    
    def __init__(self, platform_driver):
        """
        Initialize with injected dependencies.
        
        Args:
            platform_driver: Injected by ServiceRegistry
        """
        super().__init__(domain="desktop", name="window_management")
        self.driver = platform_driver
        
        # Get other services from registry (lazy)
        self._event_bus = None
        self._warden = None
    
    @property
    def event_bus(self) -> EventBus:
        """Lazy-load event bus."""
        if self._event_bus is None:
            self._event_bus = get_service("event_bus")
        return self._event_bus
    
    @property
    def warden(self):
        """Lazy-load warden for permission checks."""
        if self._warden is None:
            self._warden = get_service("warden")
        return self._warden
    
    @capability(name="focus_window")
    async def focus(self, window_title: str) -> str:
        """Focus a window by title."""
        
        # Check permission (via registry, not direct import)
        await self.warden.check_permission("desktop.window_management")
        
        # Emit event (via registry)
        await self.event_bus.emit("capability.window.focus", {
            "window": window_title
        })
        
        # Execute
        return await self.driver.focus_window(window_title)
```

#### Step 5: Migration Script for Circular Dependency Detection

```bash
cat > scripts/migration/detect_circular_deps.py << 'EOF'
"""Detect potential circular dependencies before they happen."""

import ast
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Set, Dict, List

class ImportAnalyzer(ast.NodeVisitor):
    """Extract imports from Python file."""
    
    def __init__(self):
        self.imports: Set[str] = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])

def analyze_file(file_path: Path) -> Set[str]:
    """Get all top-level imports from a file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return analyzer.imports
    except Exception as e:
        print(f"⚠ Error analyzing {file_path}: {e}")
        return set()

def build_dependency_graph(src_dir: Path) -> Dict[str, Set[str]]:
    """Build module dependency graph."""
    graph = defaultdict(set)
    
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        # Module name from file path
        rel_path = py_file.relative_to(src_dir)
        module = str(rel_path.with_suffix('')).replace('/', '.')
        
        # Get imports
        imports = analyze_file(py_file)
        
        # Filter to only internal imports
        for imp in imports:
            if imp in ['core', 'agents', 'capabilities', 'models', 'persona']:
                graph[module].add(imp)
    
    return graph

def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Detect circular dependencies using DFS."""
    
    cycles = []
    visited = set()
    rec_stack = []
    
    def dfs(node: str):
        visited.add(node)
        rec_stack.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found cycle
                idx = rec_stack.index(neighbor)
                cycle = rec_stack[idx:] + [neighbor]
                cycles.append(cycle)
        
        rec_stack.pop()
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles

def main():
    """Detect circular dependencies in src/."""
    src_dir = Path("src")
    
    if not src_dir.exists():
        print("⚠ src/ directory not found")
        return
    
    print("Analyzing dependency graph...")
    graph = build_dependency_graph(src_dir)
    
    print(f"Found {len(graph)} modules with dependencies")
    
    cycles = detect_cycles(graph)
    
    if cycles:
        print(f"\n❌ Found {len(cycles)} circular dependencies:\n")
        for i, cycle in enumerate(cycles, 1):
            print(f"{i}. {' → '.join(cycle)}")
        print("\n⚠ Fix these before proceeding with migration!")
        sys.exit(1)
    else:
        print("\n✓ No circular dependencies detected!")

if __name__ == "__main__":
    main()
EOF
```

---

## RISK #2: Plugin Breaking Changes

### The Problem

All v3.1.0 user plugins will break:

```python
# User's v3.1.0 plugin
from tara.tools.decorators import tool
from tara.tools.desktop.window_ops import focus_window

@tool(name="my_custom_tool")
def my_tool():
    focus_window("Chrome")
    return "Done"
```

After migration → `ImportError: No module named 'tara.tools'`

### Solution: Compatibility Layer

#### Step 1: Create Import Shim

```python
# File: src/extensions/compat/__init__.py
"""
Compatibility layer for v3.1.0 plugins.

Provides import aliases for old paths.
"""

import sys
import importlib
from types import ModuleType

# Mapping of old imports to new locations
IMPORT_ALIASES = {
    # Tool decorators
    'tara.tools.decorators': 'nia.src.capabilities.decorators',
    
    # Desktop tools
    'tara.tools.desktop.app_launcher': 'nia.src.capabilities.desktop.apps',
    'tara.tools.desktop.window_ops': 'nia.src.capabilities.desktop.windows',
    'tara.tools.desktop.screen_ops': 'nia.src.capabilities.desktop.screen',
    'tara.tools.desktop.uia_ops': 'nia.src.capabilities.desktop.uia',
    
    # System tools
    'tara.tools.system.file_ops': 'nia.src.capabilities.system.files',
    'tara.tools.system.input_ops': 'nia.src.capabilities.desktop.input',
    'tara.tools.system.system_ops': 'nia.src.capabilities.system.stats',
    
    # Web tools
    'tara.tools.web.browser_ops': 'nia.src.capabilities.web.browser',
    
    # Memory tools
    'tara.tools.memory.preferences': 'nia.src.capabilities.memory.preferences',
    
    # AI tools
    'tara.tools.ai.llm_ops': 'nia.src.capabilities.ai.llm',
    
    # Core imports
    'core.logger': 'nia.src.core.logger',
    'core.memory': 'nia.src.core.memory',
    
    # Plugins -> Extensions
    'plugins': 'nia.src.extensions',
}

class CompatibilityImporter:
    """
    Custom import hook that redirects old imports to new locations.
    
    This allows v3.1.0 plugins to work without modification.
    """
    
    def __init__(self):
        self.warned = set()
    
    def find_module(self, fullname, path=None):
        """Check if this is a legacy import we should redirect."""
        if fullname in IMPORT_ALIASES:
            return self
        
        # Check parent modules
        parts = fullname.split('.')
        for i in range(len(parts)):
            parent = '.'.join(parts[:i+1])
            if parent in IMPORT_ALIASES:
                return self
        
        return None
    
    def load_module(self, fullname):
        """Redirect import to new location."""
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        # Find redirect target
        redirect_to = IMPORT_ALIASES.get(fullname)
        
        if not redirect_to:
            # Check if parent is aliased
            parts = fullname.split('.')
            for i in range(len(parts), 0, -1):
                parent = '.'.join(parts[:i])
                if parent in IMPORT_ALIASES:
                    # Construct new path
                    suffix = '.'.join(parts[i:])
                    redirect_to = f"{IMPORT_ALIASES[parent]}.{suffix}" if suffix else IMPORT_ALIASES[parent]
                    break
        
        if not redirect_to:
            raise ImportError(f"Cannot redirect {fullname}")
        
        # Warn user (once per import)
        if fullname not in self.warned:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Legacy import detected: {fullname}\n"
                f"  → Redirected to: {redirect_to}\n"
                f"  → Please update your plugin to use new imports"
            )
            self.warned.add(fullname)
        
        # Import from new location
        module = importlib.import_module(redirect_to)
        sys.modules[fullname] = module
        
        return module

def enable_compatibility_mode():
    """Enable import compatibility for v3.1.0 plugins."""
    importer = CompatibilityImporter()
    if importer not in sys.meta_path:
        sys.meta_path.insert(0, importer)
        print("✓ Plugin compatibility mode enabled")

def disable_compatibility_mode():
    """Disable compatibility mode."""
    sys.meta_path = [
        imp for imp in sys.meta_path 
        if not isinstance(imp, CompatibilityImporter)
    ]
```

#### Step 2: Enable in Extension Loader

```python
# File: src/extensions/loader.py
"""Extension loader with v3.1.0 compatibility."""

import sys
from pathlib import Path
from typing import List, Type
import importlib.util
import logging

from nia.src.extensions.base import BaseExtension
from nia.src.extensions.compat import enable_compatibility_mode

logger = logging.getLogger(__name__)

class ExtensionLoader:
    """Load user extensions with backward compatibility."""
    
    def __init__(self, extensions_dir: Path, enable_compat: bool = True):
        self.extensions_dir = extensions_dir
        self.loaded_extensions: List[BaseExtension] = []
        
        # Enable v3.1.0 plugin compatibility
        if enable_compat:
            enable_compatibility_mode()
            logger.info("Plugin compatibility mode enabled for v3.1.0 plugins")
    
    def discover_extensions(self) -> List[Path]:
        """Find all extension files."""
        if not self.extensions_dir.exists():
            logger.warning(f"Extensions directory not found: {self.extensions_dir}")
            return []
        
        extensions = []
        
        # Find in custom/ subdirectory
        custom_dir = self.extensions_dir / "custom"
        if custom_dir.exists():
            extensions.extend(custom_dir.glob("*.py"))
        
        # Also check root for backward compat
        extensions.extend(self.extensions_dir.glob("*.py"))
        
        # Filter out __init__.py
        return [e for e in extensions if e.name != "__init__.py"]
    
    def load_extension(self, extension_path: Path) -> BaseExtension:
        """Load a single extension with error handling."""
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(
                extension_path.stem,
                extension_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[extension_path.stem] = module
            spec.loader.exec_module(module)
            
            # Find extension class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseExtension) and 
                    attr is not BaseExtension):
                    
                    # Instantiate and initialize
                    instance = attr()
                    instance.initialize()
                    
                    logger.info(f"✓ Loaded extension: {extension_path.name}")
                    return instance
            
            logger.warning(f"No extension class found in {extension_path.name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load extension {extension_path.name}: {e}", exc_info=True)
            return None
    
    def load_all(self) -> List[BaseExtension]:
        """Load all extensions."""
        extension_files = self.discover_extensions()
        logger.info(f"Found {len(extension_files)} extension(s)")
        
        for ext_file in extension_files:
            ext = self.load_extension(ext_file)
            if ext:
                self.loaded_extensions.append(ext)
        
        logger.info(f"✓ Loaded {len(self.loaded_extensions)} extension(s)")
        return self.loaded_extensions
```

#### Step 3: User Migration Guide

```markdown
# Plugin Migration Guide

## Option 1: Keep Using Old Imports (Automatic)

Your v3.1.0 plugins will work automatically with a compatibility shim:

```python
# Your old plugin - STILL WORKS
from tara.tools.decorators import tool

@tool(name="my_tool")
def my_tool():
    return "Works!"
```

You'll see a warning:
```
⚠ Legacy import detected: tara.tools.decorators
  → Redirected to: nia.src.capabilities.decorators
  → Please update your plugin to use new imports
```

## Option 2: Update to v4.0.0 Imports (Recommended)

Update your plugin for better performance:

```python
# Updated plugin
from nia.src.capabilities.decorators import capability
from nia.src.extensions.base import BaseExtension

class MyExtension(BaseExtension):
    def initialize(self):
        @capability(name="my_tool")
        def my_tool():
            return "Modern and fast!"
```

## Import Mapping Reference

| Old Import (v3.1.0) | New Import (v4.0.0) |
|---------------------|---------------------|
| `from tara.tools.decorators` | `from nia.src.capabilities.decorators` |
| `from tara.tools.desktop.window_ops` | `from nia.src.capabilities.desktop.windows` |
| `from core.logger` | `from nia.src.core.logger` |
| `from plugins` | `from nia.src.extensions` |
```

---

## RISK #3: Platform Abstraction Weakness

### The Problem

Current design buries platform drivers:
```
src/capabilities/desktop/drivers/
├── windows.py
├── universal.py
└── factory.py
```

Issues:
1. Platform logic mixed with capabilities
2. Hard to add macOS/Linux support
3. Capabilities not truly platform-agnostic

### Solution: Elevated Platform Layer

#### Step 1: New Platform Structure

```
src/
├── core/
│   ├── platform/                 # ← NEW: Platform abstraction
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract driver interfaces
│   │   ├── factory.py            # Auto-detect OS and create driver
│   │   ├── capabilities.py       # Platform capability definitions
│   │   └── drivers/
│   │       ├── __init__.py
│   │       ├── windows.py        # Windows implementation
│   │       ├── macos.py          # macOS implementation
│   │       ├── linux.py          # Linux implementation
│   │       └── universal.py      # Cross-platform fallback
```

#### Step 2: Platform Base Interfaces

```python
# File: src/core/platform/base.py
"""Abstract platform driver interfaces."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class WindowInfo:
    """Cross-platform window information."""
    handle: int
    title: str
    process_name: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    is_visible: bool
    is_minimized: bool

@dataclass
class ScreenRegion:
    """Screen region coordinates."""
    x: int
    y: int
    width: int
    height: int

class IPlatformDriver(ABC):
    """
    Abstract platform driver interface.
    
    All platform-specific operations MUST go through this interface.
    Capabilities should NEVER directly import platform-specific code.
    """
    
    # Window Management
    @abstractmethod
    async def list_windows(self) -> List[WindowInfo]:
        """Get list of all windows."""
        pass
    
    @abstractmethod
    async def focus_window(self, window_id: int) -> bool:
        """Bring window to foreground."""
        pass
    
    @abstractmethod
    async def minimize_window(self, window_id: int) -> bool:
        """Minimize window."""
        pass
    
    @abstractmethod
    async def maximize_window(self, window_id: int) -> bool:
        """Maximize window."""
        pass
    
    @abstractmethod
    async def close_window(self, window_id: int) -> bool:
        """Close window."""
        pass
    
    @abstractmethod
    async def move_window(self, window_id: int, x: int, y: int) -> bool:
        """Move window to position."""
        pass
    
    @abstractmethod
    async def resize_window(self, window_id: int, width: int, height: int) -> bool:
        """Resize window."""
        pass
    
    # Screen Capture
    @abstractmethod
    async def capture_screen(self, region: Optional[ScreenRegion] = None) -> bytes:
        """Capture screenshot as PNG bytes."""
        pass
    
    @abstractmethod
    async def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution."""
        pass
    
    # Input Control
    @abstractmethod
    async def mouse_move(self, x: int, y: int) -> None:
        """Move mouse to absolute position."""
        pass
    
    @abstractmethod
    async def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """Click mouse at position."""
        pass
    
    @abstractmethod
    async def keyboard_type(self, text: str) -> None:
        """Type text."""
        pass
    
    @abstractmethod
    async def keyboard_hotkey(self, *keys: str) -> None:
        """Press hotkey combination."""
        pass
    
    # Process Management
    @abstractmethod
    async def launch_process(self, executable: str, args: List[str] = None) -> int:
        """Launch process, return PID."""
        pass
    
    @abstractmethod
    async def kill_process(self, pid: int) -> bool:
        """Kill process by PID."""
        pass
    
    @abstractmethod
    async def list_processes(self) -> List[dict]:
        """List running processes."""
        pass
    
    # UI Automation
    @abstractmethod
    async def find_ui_element(self, selector: dict) -> Optional[dict]:
        """Find UI element by selector."""
        pass
    
    @abstractmethod
    async def click_ui_element(self, element: dict) -> bool:
        """Click UI element."""
        pass
    
    @abstractmethod
    async def get_ui_tree(self, root_element: Optional[dict] = None) -> dict:
        """Get UI automation tree."""
        pass
```

#### Step 3: Platform Factory

```python
# File: src/core/platform/factory.py
"""Platform driver factory with auto-detection."""

import platform
import logging
from typing import Type

from nia.src.core.platform.base import IPlatformDriver
from nia.src.core.platform.drivers.windows import WindowsDriver
from nia.src.core.platform.drivers.macos import MacOSDriver
from nia.src.core.platform.drivers.linux import LinuxDriver
from nia.src.core.platform.drivers.universal import UniversalDriver

logger = logging.getLogger(__name__)

def create_platform_driver() -> IPlatformDriver:
    """
    Auto-detect OS and create appropriate driver.
    
    Returns:
        Platform-specific driver instance
    """
    system = platform.system().lower()
    
    drivers = {
        'windows': WindowsDriver,
        'darwin': MacOSDriver,  # macOS
        'linux': LinuxDriver,
    }
    
    driver_class = drivers.get(system, UniversalDriver)
    
    logger.info(f"Creating platform driver for {system} → {driver_class.__name__}")
    
    return driver_class()

def get_platform_capabilities() -> dict:
    """
    Get capabilities of current platform.
    
    Returns:
        Dictionary of capability: bool
    """
    system = platform.system().lower()
    
    return {
        'window_management': system in ['windows', 'darwin', 'linux'],
        'ui_automation': system == 'windows',  # UIA is Windows-specific
        'accessibility_api': system == 'darwin',  # macOS Accessibility
        'screen_capture': True,  # All platforms
        'input_control': True,  # All platforms
        'process_management': True,  # All platforms
    }
```

#### Step 4: Windows Implementation Example

```python
# File: src/core/platform/drivers/windows.py
"""Windows platform driver implementation."""

import asyncio
from typing import List, Optional, Tuple
import pygetwindow as gw
import pyautogui
import psutil

from nia.src.core.platform.base import IPlatformDriver, WindowInfo, ScreenRegion

class WindowsDriver(IPlatformDriver):
    """Windows-specific platform driver."""
    
    def __init__(self):
        # Windows-specific setup
        import ctypes
        self.user32 = ctypes.windll.user32
        pyautogui.FAILSAFE = False  # Disable PyAutoGUI failsafe
    
    async def list_windows(self) -> List[WindowInfo]:
        """Get all windows."""
        windows = []
        
        for win in gw.getAllWindows():
            if win.title and win.visible:
                windows.append(WindowInfo(
                    handle=win._hWnd,
                    title=win.title,
                    process_name="",  # Would need psutil to get this
                    position=(win.left, win.top),
                    size=(win.width, win.height),
                    is_visible=win.visible,
                    is_minimized=win.isMinimized
                ))
        
        return windows
    
    async def focus_window(self, window_id: int) -> bool:
        """Focus window by handle."""
        try:
            # Windows API call
            self.user32.SetForegroundWindow(window_id)
            return True
        except Exception as e:
            logger.error(f"Failed to focus window {window_id}: {e}")
            return False
    
    async def capture_screen(self, region: Optional[ScreenRegion] = None) -> bytes:
        """Capture screenshot."""
        import io
        
        if region:
            screenshot = pyautogui.screenshot(
                region=(region.x, region.y, region.width, region.height)
            )
        else:
            screenshot = pyautogui.screenshot()
        
        # Convert to PNG bytes
        buf = io.BytesIO()
        screenshot.save(buf, format='PNG')
        return buf.getvalue()
    
    async def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """Click mouse."""
        pyautogui.click(x, y, button=button)
    
    async def keyboard_type(self, text: str) -> None:
        """Type text."""
        pyautogui.write(text, interval=0.05)
    
    async def keyboard_hotkey(self, *keys: str) -> None:
        """Press hotkey."""
        pyautogui.hotkey(*keys)
    
    async def find_ui_element(self, selector: dict) -> Optional[dict]:
        """Find UI element using Windows UIA."""
        # Would use uiautomation library
        import uiautomation as auto
        
        # Example: find by name
        if 'name' in selector:
            element = auto.ControlFromName(selector['name'])
            if element:
                return {
                    'handle': element.NativeWindowHandle,
                    'name': element.Name,
                    'type': element.ControlTypeName,
                    'bounds': element.BoundingRectangle
                }
        
        return None
    
    # ... implement remaining methods
```

#### Step 5: Update Capabilities to Use Platform Layer

```python
# File: src/capabilities/desktop/windows.py
"""Window management capability - NOW PLATFORM-AGNOSTIC."""

from nia.src.capabilities.base import BaseCapability, capability
from nia.src.core.platform.base import IPlatformDriver
from nia.src.core.registry import get_service

class WindowManagement(BaseCapability):
    """
    Window control operations.
    
    Platform-agnostic! Works on Windows, macOS, Linux.
    """
    
    def __init__(self):
        super().__init__(domain="desktop", name="window_management")
        
        # Get platform driver from registry (injected)
        self.driver: IPlatformDriver = get_service("platform_driver")
    
    @capability(
        name="list_windows",
        description="List all open windows"
    )
    async def list_windows(self) -> str:
        """List all windows - works on any platform!"""
        windows = await self.driver.list_windows()
        
        result = "Open Windows:\n"
        for i, win in enumerate(windows, 1):
            result += f"{i}. {win.title} ({win.size[0]}x{win.size[1]})\n"
        
        return result
    
    @capability(
        name="focus_window",
        description="Focus a window by title"
    )
    async def focus(self, window_title: str) -> str:
        """Focus window - platform abstracted!"""
        
        # Find window
        windows = await self.driver.list_windows()
        target = next((w for w in windows if window_title.lower() in w.title.lower()), None)
        
        if not target:
            return f"Window '{window_title}' not found"
        
        # Focus using platform driver
        success = await self.driver.focus_window(target.handle)
        
        if success:
            return f"Focused window: {target.title}"
        else:
            return f"Failed to focus window: {target.title}"
    
    # No platform-specific code here!
    # All platform logic is in src/core/platform/drivers/
```

### Benefits of This Approach

1. **True Platform Independence**
   - Capabilities have ZERO platform-specific code
   - Easy to add macOS/Linux support
   - Just implement `IPlatformDriver` for new platforms

2. **Testability**
   - Mock `IPlatformDriver` for testing
   - Test capabilities without real platform calls

3. **Maintainability**
   - All platform quirks in one place
   - Clear contract (IPlatformDriver)
   - Easier to fix platform-specific bugs

4. **Extensibility**
   - Add new platform operations to interface
   - All drivers must implement them
   - Capabilities automatically get new features

---

## Implementation Checklist

### Phase 0: Risk Mitigation Setup

- [ ] Implement enhanced ServiceRegistry with cycle detection
- [ ] Create circular dependency detection script
- [ ] Set up plugin compatibility layer
- [ ] Create platform abstraction layer

### Phase 1: Core Foundation

- [ ] Move core to `src/core/`
- [ ] Create `src/core/platform/` structure
- [ ] Implement platform drivers (Windows first)
- [ ] Register core services in main.py

### Phase 2: Gradual Migration

- [ ] Migrate one capability at a time
- [ ] Run circular dependency detector after each
- [ ] Update tests incrementally
- [ ] Keep v3.1.0 working until all migrated

### Phase 3: Plugin Migration

- [ ] Enable compatibility mode
- [ ] Test with real user plugins
- [ ] Provide migration guide
- [ ] Support dual versions for 1-2 releases

---

## Monitoring Success

### Metrics to Track

1. **Import Health**
   ```bash
   python scripts/migration/detect_circular_deps.py
   # Should always show 0 cycles
   ```

2. **Plugin Compatibility**
   ```bash
   # Count compatibility warnings in logs
   grep "Legacy import detected" data/logs/nia.log | wc -l
   # Should decrease over time as users migrate
   ```

3. **Platform Coverage**
   ```bash
   # Test on each platform
   python -c "from src.core.platform.factory import get_platform_capabilities; print(get_platform_capabilities())"
   ```

---

## Summary

The three critical risks and their solutions:

1. **Circular Dependencies** → ServiceRegistry + Dependency Injection
2. **Plugin Breakage** → Compatibility Layer + Import Shims
3. **Platform Coupling** → Elevated Platform Abstraction

**Key Principle**: The extra work upfront (DI, compat layer, platform abstraction) saves MONTHS of debugging later.

