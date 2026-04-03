"""Backward-compatible re-export for legacy DI registry imports."""

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
