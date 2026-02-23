"""src.core.bus — Event Bus Package.

Publish/subscribe message bus for decoupled inter-component communication.

Re-exports:
    AsyncEventBus — The event bus class
    get_event_bus — Accessor for the global singleton bus

Backward-compat:
    ``from src.core.events import get_event_bus``  also works.
"""
from src.core.bus.events import AsyncEventBus, get_event_bus

__all__ = [
    "AsyncEventBus",
    "get_event_bus",
]
