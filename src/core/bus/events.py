"""Async Event Bus — Inter-Component Messaging Layer.

Provides a singleton ``AsyncEventBus`` for fully decoupled communication
between N.I.A. components (NOLA voice thread, Engine, TARA, IRIS, etc.)
without direct import dependencies between them.

This module is the **canonical implementation** for ``src.core.bus``.
A backward-compat shim at ``src.core.events`` re-exports everything so
existing imports continue to work unchanged.

Architecture::

    Component A           AsyncEventBus           Component B
    ─────────────         ──────────────           ─────────────
    bus.subscribe() ───►  _subscribers dict
    bus.emit()      ───►  fan-out to all      ───► callback(data)
    NOLA thread:
    bus.emit_threadsafe() ─► asyncio.run_coroutine_threadsafe()

Features:
    - AsyncIO-native (``await bus.emit(...)``).
    - Thread-safe emission via ``emit_threadsafe()`` (NOLA runs in a thread).
    - Error isolation — one bad subscriber never crashes the bus.
    - GC-safe task tracking (background tasks held in ``_background_tasks``).

Usage::

    from src.core.bus import get_event_bus

    bus = get_event_bus()
    bus.subscribe("voice_command", my_async_callback)
    await bus.emit("voice_command", payload)
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

from src.core.logger import setup_logger

logger = setup_logger("EVENTS")


# =============================================================================
# AsyncEventBus
# =============================================================================

class AsyncEventBus:
    """Asynchronous publish/subscribe event bus.

    Thread-safe for subscription and emission; designed for a single asyncio
    event loop shared across all N.I.A. coroutines.

    Attributes:
        _subscribers: Map of event name → list of callbacks.
        _loop: Reference to the main event loop (set via ``set_loop()``).
        _background_tasks: Set of pending tasks — held to prevent GC abort.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # RIPPLE GUARD: prevent GC from aborting in-flight tasks
        self._background_tasks: set = set()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main event loop for thread-safe scheduling.

        Must be called from the async entry point before any
        ``emit_threadsafe()`` calls from background threads.
        """
        self._loop = loop

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(self, event_name: str, callback: Callable) -> None:
        """Subscribe a callback to a named event.

        Args:
            event_name: Unique string identifier for the event
                (e.g. ``"voice_command"``, ``"tara_result"``).
            callback: Async *or* sync callable.  Async callbacks are
                awaited directly; sync callbacks run in a thread pool.
        """
        self._subscribers.setdefault(event_name, []).append(callback)
        logger.debug(f"Subscribed '{callback.__name__}' to '{event_name}'")

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    async def emit(self, event_name: str, data: Any = None) -> None:
        """Publish an event to all registered subscribers (async).

        Args:
            event_name: Event to publish.
            data: Optional payload passed to every callback.
        """
        callbacks = self._subscribers.get(event_name)
        if not callbacks:
            return

        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    coro = self._safe_execute(callback, data)
                    task = asyncio.create_task(coro)
                else:
                    task = asyncio.create_task(
                        asyncio.to_thread(self._safe_execute_sync, callback, data)
                    )

                # Hold reference; discarded on completion
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            except Exception as e:
                logger.error(
                    f"Error scheduling callback for '{event_name}': {e}",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Error-isolated execution helpers
    # ------------------------------------------------------------------

    async def _safe_execute(self, callback: Callable, data: Any) -> None:
        """Await an async callback, swallowing exceptions."""
        try:
            await callback(data) if data is not None else await callback()
        except Exception as e:
            logger.error(f"Error in async listener '{callback.__name__}': {e}", exc_info=True)

    def _safe_execute_sync(self, callback: Callable, data: Any) -> None:
        """Call a sync callback, swallowing exceptions."""
        try:
            callback(data) if data is not None else callback()
        except Exception as e:
            logger.error(f"Error in sync listener '{callback.__name__}': {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Thread-safe emission (for NOLA background thread)
    # ------------------------------------------------------------------

    def emit_threadsafe(self, event_name: str, data: Any = None) -> None:
        """Emit an event safely from a non-async background thread.

        Uses ``asyncio.run_coroutine_threadsafe`` to schedule on the main
        event loop.  Falls back gracefully if no loop is available.

        Args:
            event_name: Event to publish.
            data: Optional payload.
        """
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit(event_name, data), loop)
            return

        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(self.emit(event_name, data), loop)
        except RuntimeError:
            logger.warning("EventBus: No running loop for threadsafe emit")
        except Exception as e:
            logger.error(f"EventBus emit_threadsafe failed: {e}")


# =============================================================================
# Module-Level Singleton
# =============================================================================

_event_bus = AsyncEventBus()


def get_event_bus() -> AsyncEventBus:
    """Return the global ``AsyncEventBus`` singleton."""
    return _event_bus


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AsyncEventBus",
    "get_event_bus",
]
