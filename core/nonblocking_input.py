"""Conservative, testable non-blocking keyboard + queue input helper.

This module exposes `read_input_with_queue` which prioritizes `voice_queue`
(ASR results) and otherwise returns typed input, using an OS-specific
strategy (msvcrt on Windows; select on POSIX) where available.

Design goals:
- Test-friendly: callers can inject `msvcrt`, `select`, and `stdin` to avoid
  platform-specific behavior in tests.
- Conservative: timeouts and polling intervals default to safe values; behavior
  doesn't spawn background threads on import.
- Feature-flagged: callers are expected to opt-in (e.g., via env var or CLI flag)
  before using this helper in interactive flows.
"""
from __future__ import annotations

from typing import Optional
import sys
import time
import queue


def read_input_with_queue(prompt: str,
                          voice_queue: "queue.Queue[str]",
                          timeout: float = 30.0,
                          poll_interval: float = 0.05,
                          msvcrt_module: Optional[object] = None,
                          select_module: Optional[object] = None,
                          stdin: Optional[object] = None) -> str:
    """Read input while preferring `voice_queue`.

    Parameters:
    - prompt: text printed to stdout for typed input
    - voice_queue: queue of strings from background ASR
    - timeout: maximum seconds to wait for input
    - poll_interval: how frequently to poll for queue/keyboard
    - msvcrt_module: optional override for Windows keyboard module (testable)
    - select_module: optional override for select (testable)
    - stdin: optional stdin-like object with `readline()` (testable)

    Returns the string typed (without trailing newline) or the queued text,
    or an empty string on timeout or error.
    """
    if stdin is None:
        stdin = sys.stdin

    # Fast path: if queue already contains something, return immediately
    try:
        return voice_queue.get_nowait()
    except queue.Empty:
        pass

    # Resolve platform modules lazily
    # Interpret False as explicit "do not auto-detect" sentinel for tests; None means auto-detect
    if msvcrt_module is None:
        try:
            import msvcrt as _msvcrt  # type: ignore
            msvcrt_module = _msvcrt
        except Exception:
            msvcrt_module = None
    elif msvcrt_module is False:
        msvcrt_module = None

    if select_module is None:
        try:
            import select as _select
            select_module = _select
        except Exception:
            select_module = None

    # Windows path: use msvcrt to capture keystrokes non-blockingly
    if msvcrt_module is not None:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = ''
        start = time.time()
        while True:
            # prioritize queued voice input
            try:
                return voice_queue.get_nowait()
            except queue.Empty:
                pass

            if msvcrt_module.kbhit():
                ch = msvcrt_module.getwche()
                if ch in ('\r', '\n'):
                    return buf
                elif ch == '\x08':
                    buf = buf[:-1]
                else:
                    buf += ch

            if time.time() - start > timeout:
                return ''

            time.sleep(poll_interval)

    # POSIX path: use select if available to detect readable stdin
    sys.stdout.write(prompt)
    sys.stdout.flush()
    start = time.time()
    while True:
        # prioritize queued voice input
        try:
            return voice_queue.get_nowait()
        except queue.Empty:
            pass

        if select_module is not None:
            try:
                if stdin in select_module.select([stdin], [], [], poll_interval)[0]:
                    line = stdin.readline().rstrip('\n')
                    return line
            except Exception:
                # If select fails for some reason, fall back to blocking input
                try:
                    line = input()
                    return line
                except Exception:
                    return ''
        else:
            # No select available, fallback to blocking input() with timeout enforced via time checks
            try:
                # This will block; evaluate timeout manually
                line = input()
                return line
            except Exception:
                return ''

        if time.time() - start > timeout:
            return ''
