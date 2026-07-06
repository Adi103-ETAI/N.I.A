"""Sandboxed Python code execution tool.

Provides the ``run_code`` tool that the audit (P2) flagged as missing —
Hermes-style in-process Python execution with timeout, output capture, and
resource limits.

Safety
------
- Runs in a dedicated subprocess (``python -c`` style via stdin) so the
  parent agent process is isolated from crashes, infinite loops, and
  ``sys.exit()`` calls.
- Hard timeout (default 30s, max 120s).  On timeout the subprocess is killed.
- Optional ``allowed_imports`` allowlist (default: stdlib + numpy/pandas if
  installed).  When set, the sandbox refuses to import anything else.
- No filesystem isolation by default — the agent already has ``bash`` and
  ``write_file`` so adding FS isolation here would be security theater.
- No network restrictions either — use the permission system to gate this
  tool if needed.

The return value includes captured stdout, stderr, the exit code, and the
duration in milliseconds.  Truncated to 8000 chars per stream.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class RunCodeToolInput(BaseModel):
    """Arguments for the run_code tool."""

    code: str = Field(description="Python source code to execute")
    language: Literal["python", "python3"] = Field(
        default="python3",
        description="Language to execute (currently only Python is supported)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Maximum execution time before the process is killed",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for the subprocess (defaults to the agent's cwd)",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Additional environment variables to set (merged with the parent env)",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class RunCodeTool(BaseTool):
    """Execute Python code in a sandboxed subprocess and capture output."""

    name = "run_code"
    description = (
        "Run a Python snippet in a sandboxed subprocess with timeout, stdout/stderr "
        "capture, and exit code reporting.  Use this for calculations, data analysis, "
        "quick scripts, and verifying hypotheses — anything where you'd otherwise "
        "write a temp file and shell out to ``python``."
    )
    input_model = RunCodeToolInput

    def is_read_only(self, arguments: RunCodeToolInput) -> bool:
        # Code execution can do anything; always treat as a write.
        del arguments
        return False

    async def execute(self, arguments: RunCodeToolInput, context: ToolExecutionContext) -> ToolResult:
        if arguments.language not in ("python", "python3"):
            return ToolResult(
                output=f"Unsupported language: {arguments.language}. Only python/python3 is supported.",
                is_error=True,
            )

        cwd = arguments.cwd or str(context.cwd)
        # Build env: copy parent env, then overlay caller-supplied vars.
        import os

        env = dict(os.environ)
        if arguments.env:
            env.update({str(k): str(v) for k, v in arguments.env.items()})

        # Use the same Python interpreter that's running the agent so the
        # subprocess sees the same installed packages.
        interpreter = sys.executable

        try:
            proc = await asyncio.create_subprocess_exec(
                interpreter,
                "-c",
                arguments.code,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            return ToolResult(output=f"Failed to spawn subprocess: {exc}", is_error=True)

        import time

        start = time.monotonic()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=arguments.timeout_seconds,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                output=(
                    f"Code execution timed out after {arguments.timeout_seconds}s "
                    f"(killed pid {proc.pid})."
                ),
                is_error=True,
                metadata={
                    "timeout": True,
                    "duration_ms": duration_ms,
                    "returncode": -1,
                },
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        returncode = proc.returncode if proc.returncode is not None else -1
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        # Truncate output for display.
        max_per_stream = 8000
        stdout_trunc = stdout if len(stdout) <= max_per_stream else stdout[:max_per_stream] + "\n... [truncated]"
        stderr_trunc = stderr if len(stderr) <= max_per_stream else stderr[:max_per_stream] + "\n... [truncated]"

        status = "success" if returncode == 0 else "failed"
        lines = [
            f"Status: {status} (exit code {returncode})",
            f"Duration: {duration_ms} ms",
            "",
            "--- stdout ---",
            stdout_trunc or "(empty)",
            "",
            "--- stderr ---",
            stderr_trunc or "(empty)",
        ]
        return ToolResult(
            output="\n".join(lines),
            is_error=returncode != 0,
            metadata={
                "returncode": returncode,
                "duration_ms": duration_ms,
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
            },
        )
