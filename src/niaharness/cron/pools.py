"""P1 Cron thread pools — parallel + sequential dispatch.

Ported from Hermes Agent's ``cron/scheduler.py`` pool management
functions (lines 364-407).

Two persistent thread pools:
  - **Parallel pool** — ``max_workers`` threads for workdir-less jobs
    that can run concurrently. Default max_workers=4.
  - **Sequential pool** — single-thread pool for workdir jobs that
    mutate ``os.environ["TERMINAL_CWD"]``. A single worker guarantees
    env-mutating jobs never overlap, even across ticks: a job queued
    by a newer tick waits for the previous tick's sequential jobs to
    finish rather than corrupting their os.environ state.

Both pools are persistent (created on first use, reused across ticks)
and shut down at process exit via ``atexit``.

Usage::

    from niaharness.cron.pools import submit_parallel, submit_sequential

    if job.get("workdir"):
        future = submit_sequential(run_job, job)
    else:
        future = submit_parallel(run_job, job)
    result = future.result(timeout=300)
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level pool handles.
_parallel_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_parallel_pool_max_workers: Optional[int] = None
_sequential_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

# Default parallelism for workdir-less jobs.
DEFAULT_PARALLEL_WORKERS = 4


def get_parallel_pool(
    max_workers: Optional[int] = None,
) -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent parallel pool.

    Args:
        max_workers: Max worker threads. If None, uses DEFAULT_PARALLEL_WORKERS.
            If the pool was previously created with a different max_workers,
            the old pool is shut down + a new one created.
    """
    global _parallel_pool, _parallel_pool_max_workers
    effective_max = max_workers or DEFAULT_PARALLEL_WORKERS
    if _parallel_pool is None or _parallel_pool_max_workers != effective_max:
        if _parallel_pool is not None:
            _parallel_pool.shutdown(wait=False, cancel_futures=False)
        _parallel_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_max,
            thread_name_prefix="cron-parallel",
        )
        _parallel_pool_max_workers = effective_max
    return _parallel_pool


def get_sequential_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent single-thread sequential pool.

    A single worker guarantees env-mutating jobs never overlap, even
    across ticks: a job queued by a newer tick waits for the previous
    tick's sequential jobs to finish rather than corrupting their
    os.environ state.
    """
    global _sequential_pool
    if _sequential_pool is None:
        _sequential_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cron-seq",
        )
    return _sequential_pool


def submit_parallel(
    fn: Callable[..., T],
    *args: Any,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> concurrent.futures.Future:
    """Submit a callable to the parallel pool.

    Use for workdir-less jobs that can run concurrently.
    """
    pool = get_parallel_pool(max_workers)
    return pool.submit(fn, *args, **kwargs)


def submit_sequential(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future:
    """Submit a callable to the sequential pool.

    Use for workdir jobs that mutate os.environ["TERMINAL_CWD"].
    """
    pool = get_sequential_pool()
    return pool.submit(fn, *args, **kwargs)


def shutdown_pools() -> None:
    """Shut down both persistent pools on process exit."""
    global _parallel_pool, _parallel_pool_max_workers, _sequential_pool
    if _parallel_pool is not None:
        try:
            _parallel_pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass
        _parallel_pool = None
        _parallel_pool_max_workers = None
    if _sequential_pool is not None:
        try:
            _sequential_pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass
        _sequential_pool = None


def interpreter_shutting_down(exc: Optional[BaseException] = None) -> bool:
    """Detect if the interpreter is shutting down.

    During shutdown, thread pools may be partially torn down, causing
    submit() to raise RuntimeError. This helper detects that condition
    so callers can fall back to inline execution.
    """
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "cannot schedule new futures" in msg or "shutdown" in msg:
            return True
    return False


atexit.register(shutdown_pools)


__all__ = [
    "DEFAULT_PARALLEL_WORKERS",
    "get_parallel_pool",
    "get_sequential_pool",
    "interpreter_shutting_down",
    "shutdown_pools",
    "submit_parallel",
    "submit_sequential",
]
