"""
MODULE: Process Management Tools
VERSION: 1.0.0
SCOPE: Tool wrappers for Host-side process management.
RUNS ON: Host OS (NOT Docker).

These are async tool functions that wrap HostProcessManager for use
by TARA's tool executor. They are registered as HOST tools (not sandboxed).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from src.core.logger import setup_logger
from ..decorators import security_level

logger = setup_logger("TARA.Tools.ProcessTools")


@security_level("high_risk")
async def terminate_process(name: str, force: bool = False) -> str:
    """
    Kill a running process by name with smart lookup and safety checks.

    Uses the HostProcessManager to find and terminate processes on the Host OS.
    Supports human-friendly names (e.g., 'Chrome', 'File Explorer').
    Protected system processes are blocked automatically.

    Args:
        name: Process or application name (e.g., "notepad", "chrome", "File Explorer").
        force: If True, force-kill (SIGKILL). Default is graceful termination.

    Returns:
        Status message with results.
    """
    if not name or not name.strip():
        return "❌ Error: process name is required"

    from src.infrastructure.host_os.process_manager import get_process_manager

    manager = get_process_manager()

    # Run the blocking psutil calls in a thread to keep async loop free
    result = await asyncio.to_thread(manager.kill_by_name, name.strip(), force)
    return result


@security_level("read_only")
async def find_process(name: str) -> str:
    """
    Search for running processes by name with smart matching.

    Uses alias mapping (e.g., 'chrome' -> 'chrome.exe') and fuzzy
    substring matching. Does NOT kill anything - read-only inspection.

    Args:
        name: Process or application name to search for.

    Returns:
        List of matching processes with PID and details.
    """
    if not name or not name.strip():
        return "❌ Error: process name is required"

    from src.infrastructure.host_os.process_manager import get_process_manager

    manager = get_process_manager()
    matches = await asyncio.to_thread(manager.find_process_by_name, name.strip())

    if not matches:
        return f"⚠️ No running process found matching '{name}'"

    lines = [f"🔍 Found {len(matches)} process(es) matching '{name}':"]
    for m in matches[:20]:  # Limit output
        blocked = manager.is_blocked(m.name)
        flag = " 🛡️ PROTECTED" if blocked else ""
        lines.append(f"  {m.name:<30} PID: {m.pid}{flag}")

    return "\n".join(lines)


__all__ = [
    "terminate_process",
    "find_process",
]
