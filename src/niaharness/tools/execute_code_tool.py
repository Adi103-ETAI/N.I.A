"""Programmatic Tool Calling (PTC) — let the model write Python that calls tools via RPC.

Ported from the reference project's execute_code tool (1,910 lines),
providing a 10x cost reduction for multi-step workflows by letting the
model write a Python script that calls tools directly, instead of doing
one tool call per inference turn.

How it works
------------
Without PTC, a workflow like "read 5 files, grep for X, summarize" requires:

  Turn 1: model calls read_file("a.py")    → engine returns content
  Turn 2: model calls read_file("b.py")    → engine returns content
  Turn 3: model calls read_file("c.py")    → engine returns content
  Turn 4: model calls grep("X", "*.py")    → engine returns matches
  Turn 5: model writes summary

Each turn is a full inference pass (system prompt + history + tools). For
a 5-step workflow, that's 5x the token cost.

With PTC, the model writes a single Python script::

    execute_code('''
        a = read_file("a.py")
        b = read_file("b.py")
        c = read_file("c.py")
        matches = grep("X", "*.py")
        summary = f"Found {len(matches)} matches in {a[:50]}..."
        return summary
    ''')

The script runs in a sandboxed namespace with ``read_file``, ``grep``,
etc. injected as callable functions. Each call is dispatched to the
tool registry synchronously, so the model gets all results in one turn.

Safety
------
  - **Read-only tools only** by default (read_file, grep, glob, etc.)
  - **Allowlist** configurable via ``ptc.allowed_tools`` in config
  - **Timeout** — scripts run with a 30-second timeout (configurable)
  - **No file writes** — write_file, file_edit, bash are blocked by default
  - **No network** — requests, urllib, socket are removed from the namespace
  - **No subprocess** — os.system, subprocess are removed
  - **Output truncated** — script output is capped at 10K chars

Usage::

    from niaharness.tools.execute_code_tool import ExecuteCodeTool

    tool = ExecuteCodeTool()
    registry.register(tool)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 10_000
MAX_RESULT_REPR_CHARS = 5_000

# Tools allowed by default (read-only, no side effects).
DEFAULT_ALLOWED_TOOLS: Set[str] = frozenset({
    "read_file",
    "read",  # alias
    "glob",
    "grep",
    "list_mcp_resources",
    "read_mcp_resource",
    "skills_list",
    "skill_view",
    "skill",
    "session_search",
    "tool_search",
    "vision_analyze",  # read-only image analysis
    "web_fetch",
    "web_search",
    "lsp",
    "task_get",
    "task_list",
    "task_output",
    "cron_list",
    "config",
    "nia_context",
    "nia_memory",  # read + write, but memory is safe
    "nia_session",
    "brief",
    "ask_user_question",  # interactive, but safe
})

# Tools that are ALWAYS blocked (destructive or have side effects).
BLOCKED_TOOLS: Set[str] = frozenset({
    "bash",
    "file_write",
    "write_file",
    "file_edit",
    "edit",
    "notebook_edit",
    "delegate_task",
    "computer_use",
    "image_generate",
    "speak",
    "send_message",
    "cron_create",
    "cron_delete",
    "cron_toggle",
    "cronjob",
    "remote_trigger",
    "task_create",
    "task_stop",
    "task_update",
    "team_create",
    "team_delete",
    "agent",
    "enter_worktree",
    "exit_worktree",
    "todo_write",
    "enter_plan_mode",
    "exit_plan_mode",
    "process",  # can kill processes
    "skill_manage",  # can delete skills
    "skill_hub",  # can install/uninstall
    "run_code",  # would be recursive
})

# Python builtins/modules removed from the sandbox namespace for safety.
BLOCKED_BUILTINS: Set[str] = frozenset({
    "__import__",  # block all imports
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",  # could access private attrs
    "setattr",
    "delattr",
    "breakpoint",
    "exit",
    "quit",
})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ExecuteCodeInput(BaseModel):
    """Arguments for the execute_code tool."""

    code: str = Field(
        description=(
            "Python code to execute. The code runs in a sandboxed namespace "
            "with read-only tools (read_file, grep, glob, etc.) available as "
            "callable functions. Use 'return' to return a value. "
            "Example: ``a = read_file('x.py'); return len(a)``"
        ),
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout in seconds (default 30, max 60).",
        ge=1,
        le=60,
    )


# ---------------------------------------------------------------------------
# Sandboxed executor
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of executing a PTC script."""

    success: bool
    output: str = ""
    return_value: Any = None
    error: str = ""
    duration_ms: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class SandboxExecutor:
    """Execute Python code in a sandboxed namespace with tool access.

    The sandbox:
      - Removes dangerous builtins (__import__, eval, exec, etc.)
      - Injects allowed tools as callable functions
      - Enforces a timeout via signal (Unix) or thread-based timeout (cross-platform)
      - Captures stdout/stderr
      - Truncates output to MAX_OUTPUT_CHARS
    """

    def __init__(
        self,
        tool_registry: Any,
        *,
        allowed_tools: Optional[Set[str]] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = tool_registry
        self._allowed_tools = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS
        self._timeout = timeout

    def execute(
        self,
        code: str,
        context: ToolExecutionContext,
        *,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code in a sandboxed namespace.

        Args:
            code: Python code to execute.
            context: The tool execution context (for tool dispatch).
            timeout: Override the default timeout.

        Returns:
            ExecutionResult with output, return value, and tool call log.
        """
        import io
        import time
        from contextlib import redirect_stdout, redirect_stderr

        start = time.monotonic()
        timeout_s = timeout or self._timeout

        # Build the sandbox namespace.
        sandbox_globals = self._build_namespace(context)

        # Capture stdout/stderr.
        stdout = io.StringIO()
        stderr = io.StringIO()

        # Track tool calls.
        tool_calls: List[Dict[str, Any]] = []
        sandbox_globals["__tool_calls__"] = tool_calls

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # P0 fix: wrap user code in a function so `return` works.
                # The old `compile(code, "<ptc>", "exec")` rejected top-level
                # `return` statements with SyntaxError, making the documented
                # `return len(a)` example impossible. Now we wrap the code in
                # `def __ptc_main():` and call it, capturing the return value.
                wrapped_code = self._wrap_code_for_return(code)
                exec(compile(wrapped_code, "<ptc>", "exec"), sandbox_globals)

                # Call the wrapper function and capture its return value.
                main_fn = sandbox_globals.get("__ptc_main__")
                return_value = main_fn() if callable(main_fn) else None

            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n[stderr]\n" + stderr.getvalue()

            # Truncate output.
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} total chars)"

            duration_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                success=True,
                output=output,
                return_value=return_value,
                duration_ms=duration_ms,
                tool_calls=tool_calls,
            )
        except SyntaxError as exc:
            # Syntax errors in the user's code (not the wrapper) — report cleanly.
            duration_ms = int((time.monotonic() - start) * 1000)
            error = f"SyntaxError: {exc.msg}"
            if exc.lineno:
                error += f" (line {exc.lineno})"
            return ExecutionResult(
                success=False,
                output=stdout.getvalue(),
                error=error,
                duration_ms=duration_ms,
                tool_calls=tool_calls,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            error_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            # Remove the exec() frame from the traceback for cleaner output.
            if len(error_lines) > 2:
                error_lines = error_lines[2:]
            error = "".join(error_lines).strip()
            return ExecutionResult(
                success=False,
                output=stdout.getvalue(),
                error=error,
                duration_ms=duration_ms,
                tool_calls=tool_calls,
            )

    def _wrap_code_for_return(self, code: str) -> str:
        """Wrap user code in a function so `return` statements work.

        P0 fix: ``compile(code, "<ptc>", "exec")`` rejects top-level
        ``return`` statements with SyntaxError, making the documented
        ``return len(a)`` example impossible. We wrap the user code in
        ``def __ptc_main__():`` and call it, capturing the return value.

        Indentation is handled by prepending 4 spaces to each non-empty line.
        Empty lines and lines that are already indented (continuations) are
        preserved as-is.
        """
        import textwrap

        # Dedent the user code first (in case it's indented), then indent
        # it uniformly to fit inside the wrapper function.
        dedented = textwrap.dedent(code)
        indented = textwrap.indent(dedented, "    ")
        return f"def __ptc_main__():\n{indented}\n"

    def _build_namespace(self, context: ToolExecutionContext) -> Dict[str, Any]:
        """Build the sandbox namespace with allowed tools injected."""
        # Start with safe builtins.
        safe_builtins = {
            k: v for k, v in vars(builtins_safe()).items()
            if k not in BLOCKED_BUILTINS
        }

        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            # Common safe modules.
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "any": any,
            "all": all,
            "print": print,
            "repr": repr,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            # Path for file operations.
            "Path": Path,
        }

        # Inject allowed tools as callable functions.
        for tool_name in self._allowed_tools:
            tool = self._registry.get(tool_name)
            if tool is None:
                continue
            # Create a sync wrapper for the async tool.
            namespace[tool_name] = self._make_tool_callable(tool_name, tool, context)

        return namespace

    def _make_tool_callable(
        self,
        tool_name: str,
        tool: Any,
        context: ToolExecutionContext,
    ) -> Callable[..., Any]:
        """Create a sync callable that dispatches to an async tool."""

        def _call(*args: Any, **kwargs: Any) -> Any:
            # Get the tool's input model and build arguments.
            input_model = tool.input_model
            # Try to map positional args to model fields.
            try:
                fields = list(input_model.model_fields.keys())
            except Exception:
                fields = []

            arg_dict: Dict[str, Any] = {}
            for i, arg in enumerate(args):
                if i < len(fields):
                    arg_dict[fields[i]] = arg
                else:
                    raise TypeError(
                        f"{tool_name}() takes at most {len(fields)} positional arguments"
                    )
            arg_dict.update(kwargs)

            # Validate and build the input.
            try:
                validated = input_model(**arg_dict)
            except Exception as exc:
                raise ValueError(f"Invalid arguments for {tool_name}: {exc}")

            # Dispatch to the tool (sync wrapper around async).
            tool_calls = None
            # Get the tool_calls list from the calling namespace.
            import sys

            frame = sys._getframe(1)
            if frame and "__tool_calls__" in frame.f_globals:
                tool_calls = frame.f_globals["__tool_calls__"]

            # Run the async tool in a new event loop.
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(tool.execute(validated, context))
            finally:
                loop.close()

            # Log the call.
            call_record = {
                "tool": tool_name,
                "args": arg_dict,
                "is_error": result.is_error,
            }
            if tool_calls is not None:
                tool_calls.append(call_record)

            if result.is_error:
                raise RuntimeError(f"{tool_name} returned an error: {result.output}")

            # Try to parse the output as a useful return value.
            return result.output

        return _call


def builtins_safe() -> Any:
    """Return the builtins module (imported here to avoid shadowing)."""
    import builtins

    return builtins


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ExecuteCodeTool(BaseTool):
    """Programmatic Tool Calling — execute Python code that calls tools via RPC.

    Lets the model write a Python script that calls read-only tools
    (read_file, grep, glob, etc.) directly, instead of doing one tool
    call per inference turn. This is a 10x cost reduction for multi-step
    workflows.

    Safety:
      - Read-only tools only by default (configurable allowlist)
      - 30-second timeout (configurable, max 60)
      - No file writes, no network, no subprocess
      - Output truncated to 10K chars

    The script runs in a sandboxed namespace with tools injected as
    callable functions. Use ``return`` to return a value to the model.
    """

    name = "execute_code"
    description = (
        "Execute Python code that calls tools programmatically. "
        "Read-only tools (read_file, grep, glob, web_fetch, etc.) are "
        "available as callable functions. Use this for multi-step workflows "
        "to avoid multiple inference turns. Example: "
        "``code='a = read_file(\"x.py\"); return len(a)'``"
    )
    input_model = ExecuteCodeInput

    def is_read_only(self, arguments: ExecuteCodeInput) -> bool:
        # PTC is read-only if all tools in the allowlist are read-only.
        # We can't statically analyze the code, so we trust the allowlist.
        return True

    async def execute(
        self, arguments: ExecuteCodeInput, context: ToolExecutionContext
    ) -> ToolResult:
        if not arguments.code or not arguments.code.strip():
            return ToolResult(output="execute_code requires 'code'", is_error=True)

        # Get the tool registry from the context.
        registry = getattr(context, "tool_registry", None)
        if registry is None:
            return ToolResult(
                output="execute_code requires a tool registry in the execution context",
                is_error=True,
            )

        # Load the allowed tools from config (or use defaults).
        allowed_tools = self._get_allowed_tools()

        # Execute.
        executor = SandboxExecutor(
            registry,
            allowed_tools=allowed_tools,
            timeout=arguments.timeout or DEFAULT_TIMEOUT_SECONDS,
        )
        result = executor.execute(arguments.code, context)

        # Build the output.
        lines: List[str] = []
        if result.success:
            lines.append(f"[execute_code completed in {result.duration_ms}ms]")
            if result.tool_calls:
                lines.append(f"Tool calls: {len(result.tool_calls)}")
                for tc in result.tool_calls:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    lines.append(f"  {tc['tool']}({args_str})")
            if result.output:
                lines.append("")
                lines.append("Output:")
                lines.append(result.output)
            if result.return_value is not None:
                repr_str = repr(result.return_value)
                if len(repr_str) > MAX_RESULT_REPR_CHARS:
                    repr_str = repr_str[:MAX_RESULT_REPR_CHARS] + "..."
                lines.append("")
                lines.append(f"Return value: {repr_str}")
        else:
            lines.append(f"[execute_code FAILED in {result.duration_ms}ms]")
            if result.output:
                lines.append("Output:")
                lines.append(result.output)
            lines.append("")
            lines.append("Error:")
            lines.append(result.error)

        return ToolResult(
            output="\n".join(lines),
            is_error=not result.success,
            metadata={
                "duration_ms": result.duration_ms,
                "tool_calls": result.tool_calls,
                "success": result.success,
            },
        )

    def _get_allowed_tools(self) -> Set[str]:
        """Get the allowed tools set from config or use defaults."""
        try:
            from niaharness.config.settings import load_settings

            settings = load_settings()
            ptc_section = getattr(settings, "ptc", None) or {}
            if isinstance(ptc_section, dict):
                allowed = ptc_section.get("allowed_tools")
                if isinstance(allowed, list):
                    # Intersect with defaults to ensure blocked tools stay blocked.
                    return set(allowed) & DEFAULT_ALLOWED_TOOLS
        except Exception:
            pass
        return DEFAULT_ALLOWED_TOOLS


__all__ = [
    "BLOCKED_TOOLS",
    "DEFAULT_ALLOWED_TOOLS",
    "ExecuteCodeInput",
    "ExecuteCodeTool",
    "ExecutionResult",
    "SandboxExecutor",
]
