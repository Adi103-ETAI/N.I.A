"""Service Registry - Universal Socket for N.I.A. Components.

v3.1 "Decouple" - Enhanced with Lifecycle Management

Acts as a central dictionary for loose coupling between the Core Engine
and its peripherals (Voice, Vision, Tools, Security, Plugins).

Pattern:
    - Main.py acts as the "Assembler", instantiating services and registering them.
    - Core Engine acts as the "Consumer", asking the registry for services.
    - Services implement start()/stop() for lifecycle management.

Usage:
    from core.registry import ServiceRegistry
    
    # Registration
    ServiceRegistry.register("voice", nola_manager)
    
    # Consumption (safe)
    voice = ServiceRegistry.get("voice")
    if voice:
        voice.speak("Hello")
    
    # Consumption (strict)
    voice = ServiceRegistry.require("voice")  # Raises if missing
    
    # Lifecycle
    ServiceRegistry.start_all()
    ServiceRegistry.stop_all()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from core.logger import setup_logger

logger = setup_logger("REGISTRY")


# =============================================================================
# Exceptions
# =============================================================================

class ServiceNotFoundError(Exception):
    """Raised when a required service is not registered."""
    pass


class ServiceStartupError(Exception):
    """Raised when a service fails to start."""
    pass


# =============================================================================
# Service Metadata
# =============================================================================

@dataclass
class ServiceInfo:
    """Metadata about a registered service."""
    name: str
    instance: Any
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    started: bool = False
    priority: int = 100  # Lower = starts first
    
    @property
    def type_name(self) -> str:
        return type(self.instance).__name__


# =============================================================================
# Service Registry (Enhanced Singleton)
# =============================================================================

class ServiceRegistry:
    """Central service registry for dependency decoupling.
    
    Provides:
        - Dynamic service registration/lookup
        - Lifecycle management (start_all/stop_all)
        - Dependency checking
        - Service metadata
    """
    
    _services: Dict[str, ServiceInfo] = {}
    _start_order: List[str] = []  # Track registration order
    
    # =========================================================================
    # Core Registration
    # =========================================================================
    
    @classmethod
    def register(
        cls,
        name: str,
        service: Any,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        priority: int = 100,
    ) -> None:
        """Register a service instance.
        
        Args:
            name: Unique service identifier (e.g., "voice", "vision", "security").
            service: The service instance to register.
            description: Human-readable description of the service.
            dependencies: List of service names this service depends on.
            priority: Start order priority (lower = first).
        """
        info = ServiceInfo(
            name=name,
            instance=service,
            description=description or f"{type(service).__name__} service",
            dependencies=dependencies or [],
            priority=priority,
        )
        cls._services[name] = info
        cls._start_order.append(name)
        logger.debug(f"Service registered: '{name}' -> {info.type_name}")
    
    @classmethod
    def deregister(cls, name: str) -> bool:
        """Remove a service from the registry.
        
        Args:
            name: Service name to remove.
            
        Returns:
            True if service was removed, False if not found.
        """
        if name in cls._services:
            del cls._services[name]
            if name in cls._start_order:
                cls._start_order.remove(name)
            logger.debug(f"Service deregistered: '{name}'")
            return True
        return False
    
    # =========================================================================
    # Lookups
    # =========================================================================
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a service instance by name.
        
        Args:
            name: Service identifier.
            
        Returns:
            Service instance or None if not found.
        """
        info = cls._services.get(name)
        return info.instance if info else None
    
    @classmethod
    def require(cls, name: str) -> Any:
        """Get a service instance, raising if not found.
        
        Args:
            name: Service identifier.
            
        Returns:
            Service instance.
            
        Raises:
            ServiceNotFoundError: If service is not registered.
        """
        info = cls._services.get(name)
        if info is None:
            raise ServiceNotFoundError(
                f"Required service '{name}' not found. "
                f"Available: {list(cls._services.keys())}"
            )
        return info.instance
    
    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a service is registered.
        
        Args:
            name: Service identifier.
            
        Returns:
            True if service exists.
        """
        return name in cls._services
    
    @classmethod
    def list_services(cls) -> Dict[str, str]:
        """List all registered services.
        
        Returns:
            Dict mapping service names to their type names.
        """
        return {name: info.type_name for name, info in cls._services.items()}
    
    @classmethod
    def get_info(cls, name: str) -> Optional[ServiceInfo]:
        """Get full service metadata.
        
        Args:
            name: Service identifier.
            
        Returns:
            ServiceInfo or None.
        """
        return cls._services.get(name)
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    @classmethod
    def start_all(cls) -> Dict[str, bool]:
        """Start all registered services.
        
        Services are started in priority order (lower priority first).
        Each service must have a start() method.
        
        Returns:
            Dict mapping service names to start success (True/False).
        """
        results: Dict[str, bool] = {}
        
        # Sort by priority (lower = first)
        sorted_services = sorted(
            cls._services.values(),
            key=lambda s: s.priority
        )
        
        for info in sorted_services:
            if info.started:
                results[info.name] = True
                continue
            
            # Check dependencies
            for dep in info.dependencies:
                if not cls.has(dep):
                    logger.warning(
                        f"Service '{info.name}' missing dependency: '{dep}'"
                    )
            
            try:
                if hasattr(info.instance, 'start'):
                    info.instance.start()
                info.started = True
                results[info.name] = True
                logger.debug(f"Service started: '{info.name}'")
            except Exception as e:
                results[info.name] = False
                logger.error(f"Service '{info.name}' failed to start: {e}")
        
        return results
    
    @classmethod
    def stop_all(cls) -> Dict[str, bool]:
        """Stop all registered services.
        
        Services are stopped in reverse registration order.
        Each service should have a stop() method.
        
        Returns:
            Dict mapping service names to stop success (True/False).
        """
        results: Dict[str, bool] = {}
        
        # Reverse order (LIFO)
        for name in reversed(cls._start_order):
            info = cls._services.get(name)
            if not info:
                continue
            
            try:
                if hasattr(info.instance, 'stop'):
                    info.instance.stop()
                info.started = False
                results[name] = True
                logger.debug(f"Service stopped: '{name}'")
            except Exception as e:
                results[name] = False
                logger.error(f"Service '{name}' failed to stop: {e}")
        
        return results
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    @classmethod
    def clear(cls) -> None:
        """Clear all services (for testing/shutdown)."""
        cls._services.clear()
        cls._start_order.clear()
        logger.debug("Registry cleared")
    
    @classmethod
    def print_status(cls) -> str:
        """Get a formatted status string for all services.
        
        Returns:
            Multi-line status string.
        """
        lines = ["╭─── Service Registry ───╮"]
        
        if not cls._services:
            lines.append("│ (no services)          │")
        else:
            for name, info in cls._services.items():
                status = "✅" if info.started else "⏹️"
                lines.append(f"│ {status} {name}: {info.type_name}")
        
        lines.append("╰─────────────────────────╯")
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_service(name: str) -> Optional[Any]:
    """Shortcut to ServiceRegistry.get()."""
    return ServiceRegistry.get(name)


def require_service(name: str) -> Any:
    """Shortcut to ServiceRegistry.require()."""
    return ServiceRegistry.require(name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ServiceRegistry",
    "ServiceInfo",
    "ServiceNotFoundError",
    "ServiceStartupError",
    "get_service",
    "require_service",
]
