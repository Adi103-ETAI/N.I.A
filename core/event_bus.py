"""
Core Event Bus - The Central Nervous System.

Provides a singleton AsyncEventBus for decoupled communication between
components (NOLA, Engine, TARA, etc.).

Features:
- AsyncIO native.
- Thread-safe emission (can be called from NOLA thread).
- Error handling isolation (one subscriber doesn't crash the bus).

Usage:
    from core.event_bus import ServiceRegistry  # Via registry
    # or
    from core.event_bus import get_event_bus
    
    bus = get_event_bus()
    bus.subscribe("voice_command", my_callback)
    await bus.emit("voice_command", "Hello")
"""
import asyncio
import inspect
from typing import Any, Callable, Dict, List, Coroutine, Optional
from core.logger import setup_logger
from core.services import ServiceRegistry

logger = setup_logger("EVENTS")

class AsyncEventBus:
    """Asynchronous Event Bus."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._loop = None
        # 🌊 RIPPLE SAFE: Track background tasks to prevent GC execution aborts
        self._background_tasks = set()
        
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main event loop for thread-safe scheduling."""
        self._loop = loop

    def subscribe(self, event_name: str, callback: Callable):
        """Subscribe a callback to an event.
        
        Args:
            event_name: Name of the event.
            callback: Async or sync function to call.
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)
        logger.debug(f"Subscribed to '{event_name}': {callback.__name__}")

    async def emit(self, event_name: str, data: Any = None):
        """Emit an event to all subscribers (Async).
        
        Args:
            event_name: Name of the event.
            data: Payload to pass to subscribers.
        """
        if event_name not in self._subscribers:
            # logger.debug(f"No subscribers for '{event_name}'")
            return

        for callback in self._subscribers[event_name]:
            try:
                # 1. Determine execution strategy
                if inspect.iscoroutinefunction(callback):
                    # Async callback: Schedule directly on loop
                    coro = self._safe_execute(callback, data)
                    task = asyncio.create_task(coro)
                else:
                    # Sync callback: Offload to thread to prevent loop blocking
                    task = asyncio.create_task(
                        asyncio.to_thread(self._safe_execute_sync, callback, data)
                    )
                
                # 2. RIPPLE GUARD: Track task to prevent GC
                self._background_tasks.add(task)
                
                # 3. Cleanup callback
                task.add_done_callback(self._background_tasks.discard)
                
            except Exception as e:
                logger.error(f"Error scheduling callback for '{event_name}': {e}", exc_info=True)

    async def _safe_execute(self, callback: Callable, data: Any):
        """Execute async callback with error handling."""
        try:
            if data is None:
                await callback()
            else:
                await callback(data)
        except Exception as e:
            logger.error(f"Error in async listener {callback.__name__}: {e}", exc_info=True)

    def _safe_execute_sync(self, callback: Callable, data: Any):
        """Execute sync callback with error handling."""
        try:
            if data is None:
                callback()
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Error in sync listener {callback.__name__}: {e}", exc_info=True)

    def emit_threadsafe(self, event_name: str, data: Any = None):
        """Emit, but safe to call from background threads (like NOLA)."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit(event_name, data), self._loop)
        else:
            # Fallback if loop not captured or running?
            # Could check if there is a running loop in current thread (unlikely)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.emit(event_name, data), loop)
                else:
                    logger.warning("EventBus: No running loop for threadsafe emit")
            except Exception as e:
                logger.error(f"EventBus emit_threadsafe failed: {e}")

# Singleton
_event_bus = AsyncEventBus()

def get_event_bus() -> AsyncEventBus:
    return _event_bus
