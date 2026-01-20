"""N.I.A. Service Container - Centralized Dependency Injection.

Phase 3A: Heart Transplant - Replacing implicit singletons with explicit DI.

The ServiceContainer provides:
1. **Explicit Dependencies** - All services are created in known order
2. **Testability** - Easy to inject mocks for unit testing
3. **Type Safety** - Dataclass with typed fields for IDE support
4. **Backward Compatibility** - Legacy get_*() functions still work

Usage:
    # Modern (recommended)
    from core.container import get_container
    container = get_container()
    memory = container.memory
    registry = container.window_registry
    
    # Legacy (backward compatible)
    from core.memory import get_memory_manager
    memory = get_memory_manager()  # Delegates to container
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from core.logger import setup_logger
from core.config import get_settings, Settings

logger = setup_logger("CONTAINER")

# =============================================================================
# TYPE_CHECKING Imports (IDE only - prevents circular imports)
# =============================================================================

if TYPE_CHECKING:
    from core.memory import MemoryManager
    from tara.tools.registry import WindowRegistry
    from tara.tools.browser_ops import AsyncBrowserManager
    from nola.manager import NOLAManager
    from nia.graph.builder import NIAGraph


# =============================================================================
# Service Container
# =============================================================================

@dataclass
class ServiceContainer:
    """Centralized Dependency Injection container for N.I.A.
    
    Replaces singleton patterns with explicit dependency injection.
    All core services are instantiated once and shared via this container,
    ensuring consistent state and enabling unit testing with mock services.
    
    Design Pattern:
        - **Composition Root**: All services assembled here at startup
        - **Lazy Loading**: Heavy services (NOLA, Graph) loaded on demand
        - **Backward Compatibility**: Legacy get_*() functions still work
    
    Attributes:
        settings: Application configuration (pydantic-settings).
        memory: 4-Layer Hybrid Memory System (ChromaDB + SQLite).
        window_registry: Window tracking for TARA desktop automation.
        browser_manager: Playwright browser session manager.
        
    Lazy Services (initialized on first access):
        nola: Voice I/O manager (microphone + TTS).
        graph: LangGraph reasoning engine.
    """
    settings: Settings
    memory: 'MemoryManager'
    window_registry: 'WindowRegistry'
    browser_manager: 'AsyncBrowserManager'
    
    # Lazy-loaded services (heavy/optional)
    _nola: Optional['NOLAManager'] = field(default=None, repr=False)
    _graph: Optional['NIAGraph'] = field(default=None, repr=False)
    
    @property
    def nola(self) -> Optional['NOLAManager']:
        """Get NOLA manager (lazy-loaded on first access)."""
        if self._nola is None:
            try:
                from nola.manager import NOLAManager, NOLAConfig
                config = NOLAConfig(
                    wake_words=self.settings.WAKE_WORDS,
                    wake_word_enabled=True,
                )
                self._nola = NOLAManager(config=config)
                logger.debug("NOLA manager lazy-loaded")
            except ImportError as e:
                logger.warning(f"NOLA not available: {e}")
        return self._nola
    
    @property
    def graph(self) -> Optional['NIAGraph']:
        """Get NIA graph (lazy-loaded on first access)."""
        if self._graph is None:
            try:
                from nia.graph.builder import NIAGraph
                self._graph = NIAGraph()
                logger.debug("NIAGraph lazy-loaded")
            except ImportError as e:
                logger.warning(f"NIAGraph not available: {e}")
        return self._graph


# =============================================================================
# Container Factory
# =============================================================================

def create_container(
    settings: Optional[Settings] = None,
    memory: Optional['MemoryManager'] = None,
    window_registry: Optional['WindowRegistry'] = None,
    browser_manager: Optional['AsyncBrowserManager'] = None,
) -> ServiceContainer:
    """Create a new ServiceContainer with specified or default services.
    
    This factory creates services in the correct dependency order:
    1. Settings (config)
    2. MemoryManager (requires settings paths)
    3. WindowRegistry (stateless)
    4. AsyncBrowserManager (stateless)
    
    Args:
        settings: Override default settings (for testing)
        memory: Override default memory manager (for testing)
        window_registry: Override default registry (for testing)
        browser_manager: Override default browser manager (for testing)
        
    Returns:
        Configured ServiceContainer instance.
    """
    # 1. Settings
    _settings = settings or get_settings()
    
    # 2. Memory Manager
    if memory is None:
        from core.memory import MemoryManager
        memory = MemoryManager()
        logger.debug("MemoryManager created")
    
    # 3. Window Registry
    if window_registry is None:
        from tara.tools.registry import WindowRegistry
        window_registry = WindowRegistry()
        logger.debug("WindowRegistry created")
    
    # 4. Browser Manager
    if browser_manager is None:
        from tara.tools.browser_ops import AsyncBrowserManager
        browser_manager = AsyncBrowserManager()
        logger.debug("AsyncBrowserManager created")
    
    container = ServiceContainer(
        settings=_settings,
        memory=memory,
        window_registry=window_registry,
        browser_manager=browser_manager,
    )
    
    logger.info("🏛️ ServiceContainer initialized")
    return container


# =============================================================================
# Global Container (Singleton Bridge)
# =============================================================================

_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get or create the global ServiceContainer.
    
    This is the recommended way to access services in N.I.A.
    First call initializes all core services.
    
    Returns:
        The global ServiceContainer instance.
    """
    global _container
    if _container is None:
        _container = create_container()
        _inject_backward_compat(_container)
    return _container


def set_container(container: ServiceContainer) -> None:
    """Set the global container (for testing/custom initialization).
    
    Args:
        container: Container to use as global instance.
    """
    global _container
    _container = container
    _inject_backward_compat(container)
    logger.debug("Global container overridden")


def _inject_backward_compat(container: ServiceContainer) -> None:
    """Inject container services into legacy global accessors.
    
    This ensures that old code using get_memory_manager(), get_registry(), etc.
    will get the same instances as the container.
    """
    # Inject into memory module
    try:
        import core.memory as memory_module
        memory_module._instance = container.memory
        logger.debug("Injected MemoryManager into legacy accessor")
    except Exception as e:
        logger.warning(f"Failed to inject MemoryManager: {e}")
    
    # Inject into registry module
    try:
        import tara.tools.registry as registry_module
        registry_module._registry = container.window_registry
        logger.debug("Injected WindowRegistry into legacy accessor")
    except Exception as e:
        logger.warning(f"Failed to inject WindowRegistry: {e}")
    
    # Inject into browser_ops module
    try:
        import tara.tools.browser_ops as browser_module
        browser_module._browser_manager = container.browser_manager
        logger.debug("Injected AsyncBrowserManager into legacy accessor")
    except Exception as e:
        logger.warning(f"Failed to inject AsyncBrowserManager: {e}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ServiceContainer",
    "create_container",
    "get_container",
    "set_container",
]
