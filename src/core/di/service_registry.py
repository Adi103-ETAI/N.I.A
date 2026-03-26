"""Service Registry — Dependency Injection Container.

``ServiceRegistry`` is the single source of truth for all runtime service
instances in N.I.A. It decouples components from each other: instead of
importing a concrete class, a consumer calls ``ServiceRegistry.get("name")``
and receives whatever instance is currently registered under that key.

This module is the **canonical implementation** for ``src.core.di``.
A backward-compat shim at ``src.core.registry`` re-exports everything so
existing imports continue to work unchanged.

Registered services (by convention):

    ============  =======================================
    Key           Service
    ============  =======================================
    ``memory``    ``MemoryManager`` (4-layer store)
    ``voice``     ``NolaManager`` (TTS+STT)
    ``vision``    IRIS agent
    ``security``  ``Warden`` (security gate)
    ``warden``    alias for ``security``
    ============  =======================================

Usage::

    from src.core.di import ServiceRegistry

    # Register
    ServiceRegistry.register("memory", my_mem_manager, description="Memory layer")

    # Retrieve (returns None if not registered)
    mem = ServiceRegistry.get("memory")

    # Retrieve or raise ServiceNotFoundError
    mem = ServiceRegistry.require("memory")

Lifecycle::

    await ServiceRegistry.start_all()   # calls service.start() in priority order
    await ServiceRegistry.stop_all()    # calls service.stop() in reverse order
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.logger import setup_logger

logger = setup_logger("REGISTRY")


# =============================================================================
# Exceptions
# =============================================================================

class ServiceNotFoundError(Exception):
    """Raised by ``ServiceRegistry.require()`` when the service is missing."""


class ServiceStartupError(Exception):
    """Raised when a service fails during ``start_all()``."""


# =============================================================================
# Service Metadata
# =============================================================================

@dataclass
class ServiceInfo:
    """Metadata envelope for a registered service.

    Attributes:
        name:         Unique service key.
        instance:     The actual service object.
        description:  Human-readable description (logged on registration).
        dependencies: Keys of services that must be started before this one.
        started:      True once ``service.start()`` has been called.
        priority:     Lower value = started earlier in ``start_all()``.
    """

    name: str
    instance: Any
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    started: bool = False
    priority: int = 100

    @property
    def type_name(self) -> str:
        """Return the class name of the registered service instance."""
        return type(self.instance).__name__


# =============================================================================
# ServiceRegistry
# =============================================================================

class ServiceRegistry:
    """Central service registry for dependency decoupling.

    All methods are class-methods so no instance is needed — the registry
    is effectively a process-wide singleton backed by a class-level dict.

    Provides:
        - Dynamic service registration and lookup.
        - Lazy lifecycle management (``start_all`` / ``stop_all``).
        - Dependency ordering (lower ``priority`` starts first).
    """

    _services: Dict[str, ServiceInfo] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        service: Any,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        priority: int = 100,
    ) -> None:
        """Register a service instance under a unique key.

        Args:
            name:         Unique identifier, e.g. ``"memory"``.
            service:      The service instance.
            description:  Human-readable label (for logging and status).
            dependencies: Keys that must be started before this service.
            priority:     Start order — lower = earlier.  Default ``100``.
        """
        info = ServiceInfo(
            name=name,
            instance=service,
            description=description,
            dependencies=dependencies or [],
            priority=priority,
        )
        cls._services[name] = info
        logger.debug(
            f"Registered service '{name}' ({info.type_name})"
            + (f": {description}" if description else "")
        )

    @classmethod
    def deregister(cls, name: str) -> bool:
        """Remove a service from the registry.

        Args:
            name: Service key to remove.

        Returns:
            ``True`` if the service was present and removed, ``False``
            if it was not registered.
        """
        if name not in cls._services:
            logger.warning(f"deregister: '{name}' not found")
            return False
        del cls._services[name]
        logger.debug(f"Deregistered service '{name}'")
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Return a service instance, or ``None`` if not registered.

        Args:
            name: Service identifier.

        Returns:
            The registered service instance, or ``None``.
        """
        info = cls._services.get(name)
        return info.instance if info else None

    @classmethod
    def require(cls, name: str) -> Any:
        """Return a service instance, raising if not found.

        Args:
            name: Service identifier.

        Returns:
            The registered service instance.

        Raises:
            ServiceNotFoundError: If ``name`` is not in the registry.
        """
        info = cls._services.get(name)
        if not info:
            available = list(cls._services.keys())
            raise ServiceNotFoundError(
                f"Service '{name}' not registered. "
                f"Available: {available}"
            )
        return info.instance

    @classmethod
    def has(cls, name: str) -> bool:
        """Check whether a service is registered.

        Args:
            name: Service identifier.

        Returns:
            ``True`` if the service exists.
        """
        return name in cls._services

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_services(cls) -> Dict[str, str]:
        """Return a mapping of service names to their class names.

        Returns:
            ``{"memory": "MemoryManager", "voice": "NolaManager", ...}``
        """
        return {n: info.type_name for n, info in cls._services.items()}

    @classmethod
    def get_info(cls, name: str) -> Optional[ServiceInfo]:
        """Return full ``ServiceInfo`` metadata for a service.

        Args:
            name: Service identifier.

        Returns:
            ``ServiceInfo`` or ``None`` if not registered.
        """
        return cls._services.get(name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start_all(cls) -> Dict[str, bool]:
        """Start all registered services in priority order.

        Services with a lower ``priority`` value are started first.
        Each service is expected to have a ``start()`` method.

        Returns:
            Dict mapping service name → ``True`` (success) / ``False``
            (failed or no ``start()`` method).
        """
        results: Dict[str, bool] = {}
        ordered = sorted(cls._services.values(), key=lambda s: s.priority)

        for info in ordered:
            if info.started:
                results[info.name] = True
                continue

            service = info.instance
            if not hasattr(service, "start"):
                logger.debug(f"'{info.name}' has no start() — marking started")
                info.started = True
                results[info.name] = True
                continue

            try:
                service.start()
                info.started = True
                logger.info(f"✅ Started: {info.name} ({info.type_name})")
                results[info.name] = True
            except Exception as e:
                logger.error(f"❌ Failed to start '{info.name}': {e}")
                results[info.name] = False

        return results

    @classmethod
    def stop_all(cls) -> Dict[str, bool]:
        """Stop all registered services in reverse registration order.

        Each service is expected to have a ``stop()`` method.

        Returns:
            Dict mapping service name → ``True`` (success) / ``False``
            (failed or no ``stop()`` method).
        """
        results: Dict[str, bool] = {}
        ordered = list(reversed(list(cls._services.values())))

        for info in ordered:
            service = info.instance
            if not hasattr(service, "stop"):
                info.started = False
                results[info.name] = True
                continue

            try:
                service.stop()
                info.started = False
                logger.info(f"⏹ Stopped: {info.name}")
                results[info.name] = True
            except Exception as e:
                logger.error(f"❌ Failed to stop '{info.name}': {e}")
                results[info.name] = False

        return results

    @classmethod
    def clear(cls) -> None:
        """Remove all registered services.  Used in tests and clean shutdown."""
        cls._services.clear()
        logger.debug("ServiceRegistry cleared")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @classmethod
    def print_status(cls) -> str:
        """Return a formatted multi-line status string for all services.

        Returns:
            Human-readable registry state, suitable for logging or display.
        """
        if not cls._services:
            return "ServiceRegistry: (empty)"

        lines = ["ServiceRegistry status:"]
        for name, info in cls._services.items():
            state = "🟢" if info.started else "⚪"
            lines.append(f"  {state} {name:20s} ({info.type_name})"
                         + (f" — {info.description}" if info.description else ""))
        return "\n".join(lines)


# =============================================================================
# Convenience Shortcuts
# =============================================================================

def get_service(name: str) -> Optional[Any]:
    """Shortcut for ``ServiceRegistry.get(name)``."""
    return ServiceRegistry.get(name)


def require_service(name: str) -> Any:
    """Shortcut for ``ServiceRegistry.require(name)``."""
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
