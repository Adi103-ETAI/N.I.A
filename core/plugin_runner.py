"""Lightweight safe plugin runner using subprocess isolation.

This module provides a simple subprocess-based execution helper for plugin files.
It executes the plugin file in a separate Python process and communicates via
JSON over stdin/stdout. This is not a full sandbox (use containers/VMs for
stronger isolation), but it reduces in-process blast radius and accidental
global state sharing.

API:
- execute_plugin_file(path, params=None, timeout=5) -> dict

The plugin file should accept JSON on stdin and print JSON to stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class PluginExecutionError(RuntimeError):
    pass


def execute_plugin_file(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 5) -> Dict[str, Any]:
    """Execute a plugin file in a subprocess and return parsed JSON result.

    Args:
        path: Path to the plugin Python file.
        params: JSON-serializable dict to send on stdin.
        timeout: Process timeout in seconds.

    Returns:
        Decoded JSON object from plugin stdout.

    Raises:
        PluginExecutionError: on non-zero exit code, invalid JSON output, or timeout.
    """
    params = params or {}
    path_obj = Path(path)
    if not path_obj.exists():
        raise PluginExecutionError(f"Plugin file not found: {path}")

    cmd = [sys.executable, str(path_obj.resolve())]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(params).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise PluginExecutionError(f"Plugin execution timed out: {exc}") from exc

    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        raise PluginExecutionError(f"Plugin exited with code {proc.returncode}: {stderr_text}")

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except Exception as exc:  # pragma: no cover - defensive
        raise PluginExecutionError(f"Failed to parse plugin JSON output: {exc}; output:{stdout!r}") from exc
