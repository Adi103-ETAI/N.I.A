"""P1 Cron ReadWriteLock — writer-preferring readers-writer lock.

Ported from Hermes Agent's ``cron/scheduler.py`` _ReadWriteLock class.

Guards the process-global ``os.environ["TERMINAL_CWD"]`` override that
a workdir cron job applies for the whole of its agent run. Workdir jobs
are writers: they mutate the shared env and need exclusive access.
Workdir-less jobs are readers: they only observe TERMINAL_CWD
(indirectly, via the terminal / file / code-exec tools), so any number
of them may run concurrently with each other, but none may run
alongside a writer — that is exactly what stops a workdir-less job
from picking up another job's workdir override and running its commands
in the wrong directory.

Writer preference bounds the wait for a workdir job (dispatched on the
single-thread sequential pool) so a stream of workdir-less readers
cannot starve it.

Usage::

    from niaharness.cron.readwrite_lock import terminal_cwd_lock

    if job.get("workdir"):
        terminal_cwd_lock.acquire_write()
        try:
            os.environ["TERMINAL_CWD"] = job["workdir"]
            await run_job(job)
        finally:
            del os.environ["TERMINAL_CWD"]
            terminal_cwd_lock.release_write()
    else:
        terminal_cwd_lock.acquire_read()
        try:
            await run_job(job)
        finally:
            terminal_cwd_lock.release_read()
"""

from __future__ import annotations

import threading


class ReadWriteLock:
    """Writer-preferring readers-writer lock.

    - Multiple readers can hold the lock simultaneously.
    - Only one writer can hold the lock at a time.
    - Writers are preferred: if a writer is waiting, new readers block
      until the writer acquires + releases the lock.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers: int = 0
        self._writer_active: bool = False
        self._writers_waiting: int = 0

    def acquire_read(self) -> None:
        """Acquire a read lock. Blocks if a writer is active or waiting."""
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release a read lock."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        """Acquire a write lock. Blocks until all readers + writers release."""
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        """Release the write lock."""
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    @property
    def reader_count(self) -> int:
        """Current number of active readers (for diagnostics)."""
        with self._cond:
            return self._readers

    @property
    def writer_active(self) -> bool:
        """True if a writer currently holds the lock."""
        with self._cond:
            return self._writer_active

    @property
    def writers_waiting(self) -> int:
        """Number of writers waiting to acquire the lock."""
        with self._cond:
            return self._writers_waiting


# Process-global lock for TERMINAL_CWD coordination.
# Serializes the per-job TERMINAL_CWD override against every other
# concurrently running cron job.
terminal_cwd_lock = ReadWriteLock()


__all__ = [
    "ReadWriteLock",
    "terminal_cwd_lock",
]
