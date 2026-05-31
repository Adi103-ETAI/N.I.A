"""Ripgrep integration utilities.

Provides functions for running ripgrep searches with proper error handling
and timeout support.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional


class RipgrepTimeoutError(Exception):
    """Raised when a ripgrep search times out."""

    def __init__(self, message: str, partial_results: Optional[List[str]] = None):
        super().__init__(message)
        self.partial_results = partial_results or []


class RipgrepUnavailableError(Exception):
    """Raised when ripgrep cannot be found or executed."""

    def __init__(self, message: str, mode: str = "system", command: str = "rg"):
        super().__init__(message)
        self.mode = mode
        self.command = command


@dataclass
class RipgrepConfig:
    """Ripgrep configuration."""

    mode: str  # 'system', 'builtin', 'embedded'
    command: str
    args: List[str]


def resolve_ripgrep_config() -> RipgrepConfig:
    """Resolve the ripgrep configuration.

    Determines the best available ripgrep binary.
    """
    # Check for system ripgrep
    rg_path = shutil.which("rg")
    if rg_path:
        return RipgrepConfig(mode="system", command=rg_path, args=[])

    # Try to find bundled ripgrep (for packaged applications)
    # This is a placeholder - actual implementation depends on packaging
    return RipgrepConfig(mode="system", command="rg", args=[])


def _get_ripgrep_config() -> RipgrepConfig:
    """Get or cache the ripgrep configuration."""
    if not hasattr(_get_ripgrep_config, "_cache"):
        _get_ripgrep_config._cache = resolve_ripgrep_config()
    return _get_ripgrep_config._cache


def get_ripgrep_install_hint() -> str:
    """Get a platform-specific hint for installing ripgrep."""
    if sys.platform == "win32":
        return (
            "Install ripgrep and confirm `rg --version` works in the same terminal. "
            "Windows: `winget install BurntSushi.ripgrep.MSVC` or `choco install ripgrep`."
        )
    if sys.platform == "darwin":
        return (
            "Install ripgrep and confirm `rg --version` works in the same terminal. "
            "macOS: `brew install ripgrep`."
        )
    return (
        "Install ripgrep and confirm `rg --version` works in the same terminal. "
        "Linux: use your distro package manager, for example `apt install ripgrep`."
    )


import sys  # noqa: E402 (needed for get_ripgrep_install_hint)


async def rip_grep(
    args: List[str],
    target: str,
    timeout_seconds: Optional[float] = None,
) -> List[str]:
    """Run a ripgrep search and return matching lines.

    Args:
        args: Additional ripgrep arguments (e.g., ['-i', 'pattern']).
        target: The file or directory to search.
        timeout_seconds: Optional timeout in seconds.

    Returns:
        A list of matching lines.

    Raises:
        RipgrepTimeoutError: If the search times out.
        RipgrepUnavailableError: If ripgrep cannot be found.
    """
    config = _get_ripgrep_config()

    # Check if ripgrep is available
    if config.mode == "system" and not shutil.which(config.command):
        raise RipgrepUnavailableError(
            f"ripgrep (rg) is required for file search but could not be found. "
            f"{get_ripgrep_install_hint()}",
            mode=config.mode,
            command=config.command,
        )

    full_args = [config.command] + config.args + args + [target]

    # Set timeout based on platform
    if timeout_seconds is None:
        timeout_seconds = 60.0 if sys.platform == "linux" else 20.0

    try:
        proc = await asyncio.create_subprocess_exec(
            *full_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RipgrepTimeoutError(
                f"Ripgrep search timed out after {timeout_seconds} seconds. "
                "Try searching a more specific path or pattern.",
                partial_results=[],
            )

        if proc.returncode in (0, 1):
            # 0 = matches found, 1 = no matches (both are success)
            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return []
            return [
                line.rstrip("\r")
                for line in output.split("\n")
                if line.strip()
            ]
        else:
            error_msg = stderr.decode("utf-8", errors="replace")
            if "os error 11" in error_msg.lower() or "resource temporarily unavailable" in error_msg.lower():
                # EAGAIN error - retry with single thread
                return await rip_grep(
                    ["-j", "1"] + args,
                    target,
                    timeout_seconds,
                )
            raise RipgrepUnavailableError(
                f"ripgrep exited with code {proc.returncode}: {error_msg}",
                mode=config.mode,
                command=config.command,
            )

    except FileNotFoundError:
        raise RipgrepUnavailableError(
            f"ripgrep (rg) is required for file search but could not be found. "
            f"{get_ripgrep_install_hint()}",
            mode=config.mode,
            command=config.command,
        )


async def rip_grep_stream(
    args: List[str],
    target: str,
    on_lines: Callable[[List[str]], None],
    abort_event: Optional[asyncio.Event] = None,
) -> None:
    """Stream ripgrep results as they arrive.

    Calls on_lines with batches of complete lines as they become available.
    """
    config = _get_ripgrep_config()
    full_args = [config.command] + config.args + args + [target]

    proc = await asyncio.create_subprocess_exec(
        *full_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    remainder = ""

    async for chunk in proc.stdout:  # type: ignore
        if abort_event and abort_event.is_set():
            proc.kill()
            break

        data = remainder + chunk.decode("utf-8", errors="replace")
        lines = data.split("\n")
        remainder = lines.pop() or ""

        if lines:
            on_lines([line.rstrip("\r") for line in lines])

    # Flush any remaining data
    if remainder:
        on_lines([remainder.rstrip("\r")])

    await proc.wait()


async def count_files_rounded_rg(
    dir_path: str,
    ignore_patterns: Optional[List[str]] = None,
    timeout_seconds: float = 30.0,
) -> Optional[int]:
    """Count files in a directory using ripgrep, rounded to nearest power of 10.

    More efficient than native methods for large directories.
    Returns None on error.
    """
    args = ["--files", "--hidden"]

    if ignore_patterns:
        for pattern in ignore_patterns:
            args.extend(["--glob", f"!{pattern}"])

    try:
        result = await rip_grep(args, dir_path, timeout_seconds)
        count = len(result)

        if count == 0:
            return 0

        # Round to nearest power of 10 for privacy
        import math

        magnitude = math.floor(math.log10(count))
        power = 10**magnitude
        return round(count / power) * power
    except (RipgrepTimeoutError, RipgrepUnavailableError):
        return None


def get_ripgrep_status() -> dict:
    """Get ripgrep status and configuration info."""
    config = _get_ripgrep_config()
    return {
        "mode": config.mode,
        "path": config.command,
        "available": shutil.which(config.command) is not None,
    }
