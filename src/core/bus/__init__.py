"""src.core.bus — Event Bus Package.

Publish/subscribe message bus for decoupled inter-component communication.

Re-exports:
    AsyncEventBus        — The event bus class
    get_event_bus        — Accessor for the global singleton bus
    ContextWormhole      — Coordinator-side listener for cross-agent observations
    emit_observation     — Subagent-side helper to emit ContextObservation events
    get_subagent_context — Builds focused context blocks for subagent prompts

Backward-compat:
    ``from src.core.events import get_event_bus``  also works.
"""
from src.core.bus.events import AsyncEventBus, get_event_bus
from src.core.bus.context_wormhole import (
    ContextWormhole,
    emit_observation,
    get_subagent_context,
)

__all__ = [
    "AsyncEventBus",
    "get_event_bus",
    "ContextWormhole",
    "emit_observation",
    "get_subagent_context",
]
