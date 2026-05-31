"""Sleep and timeout utilities.

Provides abort-responsive sleep and promise-with-timeout for async operations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


async def sleep(
    seconds: float,
    cancel_event: Optional[asyncio.Event] = None,
    throw_on_cancel: bool = False,
    cancel_error_factory: Optional[Callable[[], Exception]] = None,
) -> None:
    """Abort-responsive sleep.

    Resolves after ``seconds`` milliseconds, or immediately when
    ``cancel_event`` is set (so backoff loops don't block shutdown).

    Args:
        seconds: Duration to sleep in seconds.
        cancel_event: An asyncio.Event to check for cancellation.
        throw_on_cancel: If True, raise an error when cancelled.
        cancel_error_factory: Factory to create the cancellation error.
    """
    if cancel_event is not None and cancel_event.is_set():
        if throw_on_cancel or cancel_error_factory:
            raise (cancel_error_factory() if cancel_error_factory else asyncio.CancelledError())
        return

    try:
        await asyncio.wait_for(
            asyncio.sleep(seconds),
            timeout=seconds + 0.1,  # Small buffer
        )
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        if throw_on_cancel or cancel_error_factory:
            raise (cancel_error_factory() if cancel_error_factory else asyncio.CancelledError())
        return

    # Check cancellation after sleep
    if cancel_event is not None and cancel_event.is_set():
        if throw_on_cancel or cancel_error_factory:
            raise (cancel_error_factory() if cancel_error_factory else asyncio.CancelledError())


async def with_timeout(
    coro: Any,
    seconds: float,
    message: str = "Operation timed out",
) -> Any:
    """Race a coroutine against a timeout.

    Raises TimeoutError with the given message if the coroutine doesn't
    complete within the specified seconds.

    Note: This doesn't cancel the underlying work - if the coroutine is
    backed by a runaway async operation, that keeps running.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(message)
