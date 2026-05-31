"""Process management utilities.

Provides functions for running subprocesses with proper error handling.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ProcessResult:
    """Result of a subprocess execution."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if the process succeeded (returncode 0)."""
        return self.returncode == 0


async def run_command(
    args: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    capture_output: bool = True,
) -> ProcessResult:
    """Run a command asynchronously and return the result.

    Args:
        args: Command and arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        capture_output: Whether to capture stdout and stderr.

    Returns:
        ProcessResult with return code and output.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
        cwd=cwd,
        env=full_env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout} seconds: {args[0]}")

    return ProcessResult(
        returncode=proc.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
        stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
    )


def run_command_sync(
    args: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    capture_output: bool = True,
) -> ProcessResult:
    """Run a command synchronously and return the result.

    Args:
        args: Command and arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        capture_output: Whether to capture stdout and stderr.

    Returns:
        ProcessResult with return code and output.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=full_env,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return ProcessResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out after {timeout} seconds: {args[0]}")


async def find_executable(name: str) -> Optional[str]:
    """Find an executable in PATH.

    Returns the full path to the executable, or None if not found.
    """
    import shutil

    # Try shutil.which first
    path = shutil.which(name)
    if path:
        return path

    # Try with common extensions on Windows
    if sys.platform == "win32":
        for ext in [".exe", ".cmd", ".bat", ".com"]:
            path = shutil.which(name + ext)
            if path:
                return path

    return None


def kill_process_tree(pid: int, signal_num: int = signal.SIGTERM) -> bool:
    """Kill a process and all its children.

    Returns True if the process was killed successfully.
    """
    try:
        if sys.platform == "win32":
            # On Windows, use taskkill
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            # On Unix, use os.killpg
            try:
                os.killpg(os.getpgid(pid), signal_num)
            except (ProcessLookupError, PermissionError):
                pass
        return True
    except Exception:
        return False


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
