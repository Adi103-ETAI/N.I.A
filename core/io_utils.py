"""I/O helpers for non-blocking keyboard input with voice queue fallback.

Provides a single testable function `read_input_with_queue(prompt, voice_queue, timeout, msvcrt_module)`
that supports Windows (msvcrt-based) and POSIX (select/input) flows and is easy to monkeypatch
in tests.
"""
from __future__ import annotations

from typing import Optional
import sys
import time
import queue


def read_input_with_queue(prompt: str, voice_queue: "queue.Queue[str]", timeout: float = 30.0, msvcrt_module: Optional[object] = None) -> str:
    """Read input while giving priority to `voice_queue` messages.

    Behavior:
    - If `voice_queue` has an item, return it immediately.
    - On Windows (msvcrt available), read keystrokes non-blockingly while polling the queue; return once Enter is pressed.
    - On POSIX, attempt non-blocking detection with `select.select` and fallback to `input()` if not available.

    `msvcrt_module` can be provided (for tests) to inject a fake msvcrt replacement.
    """
    # Quick check if there's already voice data
    try:
        val = voice_queue.get_nowait()
        return val
    except queue.Empty:
        pass

    # Prefer injected module for tests over real msvcrt
    if msvcrt_module is None:
        try:
            import msvcrt as _msvcrt  # type: ignore
            msvcrt_module = _msvcrt
        except Exception:
            msvcrt_module = None

    # Windows-style non-blocking keyboard
    if msvcrt_module is not None:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = ''
        while True:
            # check for queued voice input first
            try:
                val = voice_queue.get_nowait()
                return val
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
            time.sleep(0.05)

    # POSIX-style fallback: try select to detect stdin readability
    sys.stdout.write(prompt)
    sys.stdout.flush()
    start = time.time()
    line = ''
    while True:
        try:
            val = voice_queue.get_nowait()
            return val
        except queue.Empty:
            pass

        # try to use select for non-blocking stdin detection. Use any
        # injected/test monkeypatched `select` in the module globals if present,
        # otherwise import the real select module.
        try:
            sel = globals().get('select', None)
            if sel is None:
                import select as sel
            if sys.stdin in sel.select([sys.stdin], [], [], 0.2)[0]:
                line = sys.stdin.readline().rstrip('\n')
                return line
        except Exception:
            # No select available or failed; fall back to blocking input()
            try:
                line = input()
                return line
            except Exception:
                return ''

        if time.time() - start > timeout:
            return ''
