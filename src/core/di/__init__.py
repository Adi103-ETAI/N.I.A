"""src.core.di — Dependency Injection Package.

Central service registry for decoupled component access.

Re-exports:
    ServiceRegistry     — Class-level service container
    ServiceInfo         — Metadata dataclass for registered services
    ServiceNotFoundError / ServiceStartupError — Exception types
    get_service         — Shortcut for ServiceRegistry.get()
    require_service     — Shortcut for ServiceRegistry.require()

Backward-compat:
    Legacy imports should use ``from src.core.di.registry import ServiceRegistry``.
"""
from src.core.di.service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceNotFoundError,
    ServiceStartupError,
    get_service,
    require_service,
)

__all__ = [
    "ServiceRegistry",
    "ServiceInfo",
    "ServiceNotFoundError",
    "ServiceStartupError",
    "get_service",
    "require_service",
]
