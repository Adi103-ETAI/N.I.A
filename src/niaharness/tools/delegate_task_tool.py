"""Delegate task tool — spawn isolated subagents for parallel work.

Adapted from Hermes Agent's tools/delegate_tool.py.

1. Parent calls delegate_task with a goal + optional context.
2. A child QueryEngine is created with a FRESH conversation (no parent
   history) and a restricted toolset.
3. The child runs to completion (or until max_turns / timeout).
4. Only the child's final text result is returned to the parent — the
   parent never sees the child's intermediate tool calls or reasoning.
5. Supports batch mode: pass ``tasks`` array for parallel fan-out.

Blocked tools (children must never have access to) — mirrors Hermes's
DELEGATE_BLOCKED_TOOLS plus NIA-specific side-effect tools:
- delegate_task (no recursive delegation)
- ask_user_question (no user interaction from subagent)
- nia_memory (no writes to shared memory)
- nia_session (no session persistence from subagent)
- skill_manage (no skill mutations from subagent)
- cron_create / cron_delete / cron_toggle (no scheduling from subagent)
- send_message (no cross-platform side effects)
- run_code (children should reason step-by-step, not write scripts)
- agent / team_create / team_delete (no subprocess spawning / team ops)
- task_create / task_get / task_list / task_stop / task_output / task_update
- enter_worktree / exit_worktree (no workspace mutation)
- remote_trigger (no remote invocation)
- nia_context (no shared context mutation)
- nia_voice (no voice side effects)
- speak (no TTS side effects)
- config (no global settings mutation)

Safety (mirrors Hermes):
- Max delegation depth: 2 (configurable via NIA_DELEGATE_MAX_DEPTH env var).
- Max turns per child: 10 (configurable via NIA_DELEGATE_MAX_TURNS env var).
- Per-child timeout: 120s (configurable via NIA_DELEGATE_TIMEOUT env var).
- Subagent permission mode: auto-deny dangerous commands by default
  (configurable via NIA_DELEGATE_AUTO_APPROVE env var).
- Children inherit the parent's api_client, model, max_tokens, and
  tool_metadata (mcp_manager, bridge_manager, etc.).
- Depth is propagated to children via _delegate_depth in tool_metadata.

Reference: Hermes Agent's ``tools/delegate_tool.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_DEPTH = 2
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_CONCURRENT_CHILDREN = 3


# Tools that children must never have access to.
# Mirrors Hermes's DELEGATE_BLOCKED_TOOLS + NIA-specific side-effect tools.
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        # Hermes blocks these:
        "delegate_task",      # no recursive delegation
        "ask_user_question",  # no user interaction from subagent
        "nia_memory",         # no writes to shared memory
        "send_message",       # no cross-platform side effects
        "run_code",           # children should reason step-by-step
        "cron_create",        # no scheduling from subagent
        "cron_delete",
        "cron_toggle",
        # NIA-specific additions:
        "nia_session",        # no session persistence from subagent
        "skill_manage",       # no skill mutations from subagent
        "agent",              # no subprocess spawning
        "team_create",        # no team ops
        "team_delete",
        "task_create",        # no background task lifecycle
        "task_get",
        "task_list",
        "task_stop",
        "task_output",
        "task_update",
        "enter_worktree",     # no workspace mutation
        "exit_worktree",
        "remote_trigger",     # no remote invocation
        "nia_context",        # no shared context mutation
        "nia_voice",          # no voice side effects
        "speak",              # no TTS side effects
        "config",             # no global settings mutation
    ]
)


# ---------------------------------------------------------------------------
# Subagent approval callbacks (adapted from Hermes)
# ---------------------------------------------------------------------------


def _subagent_auto_deny(command: str, description: str, **kwargs) -> bool:
    """Auto-deny dangerous commands in subagent threads (safe default).

    Returns False so the subagent sees a refusal it can recover from,
    and never calls input() (which would deadlock the parent TUI).
    """
    logger.warning(
        "Subagent auto-denied dangerous command: %s (%s). "
        "Set NIA_DELEGATE_AUTO_APPROVE=1 to allow.",
        command,
        description,
    )
    return False


def _subagent_auto_approve(command: str, description: str, **kwargs) -> bool:
    """Auto-approve dangerous commands in subagent threads (opt-in YOLO).

    Only used when NIA_DELEGATE_AUTO_APPROVE=1. Returns True so the
    subagent proceeds without blocking the parent UI.
    """
    logger.warning(
        "Subagent auto-approved dangerous command: %s (%s)",
        command,
        description,
    )
    return True


def _get_subagent_approval_callback():
    """Return the approval callback for subagent permission prompts.

    Config: NIA_DELEGATE_AUTO_APPROVE env var (default: not set = deny).
    """
    if os.environ.get("NIA_DELEGATE_AUTO_APPROVE", "").lower() in ("1", "true", "yes"):
        return _subagent_auto_approve
    return _subagent_auto_deny


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_max_depth() -> int:
    return int(os.environ.get("NIA_DELEGATE_MAX_DEPTH", str(DEFAULT_MAX_DEPTH)))


def _get_max_turns() -> int:
    return int(os.environ.get("NIA_DELEGATE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))


def _get_timeout() -> float:
    return float(os.environ.get("NIA_DELEGATE_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))


def _get_max_concurrent() -> int:
    return int(os.environ.get("NIA_DELEGATE_MAX_CONCURRENT", str(MAX_CONCURRENT_CHILDREN)))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class DelegateTaskInput(BaseModel):
    """Arguments for the delegate_task tool.

    Two modes:
    - Single: pass ``goal`` (+ optional ``context``).
    - Batch: pass ``tasks`` array of {goal, context, tools} dicts.

    The parent blocks until all children complete. The parent's context
    only sees the delegation call and the structured result — never the
    child's intermediate tool calls.
    """

    goal: str | None = Field(
        default=None,
        description="The task for the subagent to accomplish (single mode).",
    )
    context: str | None = Field(
        default=None,
        description="Additional context for the subagent (background, constraints, relevant files).",
    )
    tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Batch mode: array of tasks to run in parallel (max "
            f"{MAX_CONCURRENT_CHILDREN} concurrent). Each task is a dict "
            "with 'goal' (required), 'context' (optional), 'tools' (optional "
            "comma-separated tool name list). When provided, 'goal' and "
            "'context' are ignored."
        ),
    )
    tools: str | None = Field(
        default=None,
        description=(
            "Comma-separated tool name whitelist for the subagent. "
            "If omitted, the subagent gets all tools except blocked ones."
        ),
    )
    max_turns: int = Field(
        default=DEFAULT_MAX_TURNS,
        ge=1,
        le=50,
        description=f"Maximum agentic turns per subagent (default {DEFAULT_MAX_TURNS}).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class DelegateTaskTool(BaseTool):
    """Spawn one or more isolated subagents to handle delegated tasks."""

    name = "delegate_task"
    description = (
        "Spawn isolated subagents for parallel work. Each subagent gets a "
        "fresh conversation (no parent history), a restricted toolset, and "
        "runs to completion with a timeout. Only the structured result is "
        "returned to the parent. Supports single-task and batch (parallel) "
        f"modes (max {_get_max_concurrent()} concurrent). The parent's context "
        "only sees the delegation call and the result — never the child's "
        "intermediate tool calls."
    )
    input_model = DelegateTaskInput

    def is_read_only(self, arguments: DelegateTaskInput) -> bool:
        del arguments
        return False

    async def execute(self, arguments: DelegateTaskInput, context: ToolExecutionContext) -> ToolResult:
        # Check delegation depth.
        current_depth = context.metadata.get("_delegate_depth", 0)
        max_depth = _get_max_depth()
        if current_depth >= max_depth:
            return ToolResult(
                output=json.dumps({
                    "error": (
                        f"Delegation depth limit reached (depth={current_depth}, "
                        f"max={max_depth}). Raise NIA_DELEGATE_MAX_DEPTH env var "
                        f"if deeper nesting is required."
                    ),
                }),
                is_error=True,
            )

        # Determine mode.
        if arguments.tasks:
            return await self._run_batch(arguments, context)
        if arguments.goal:
            return await self._run_single(arguments, context)
        return ToolResult(
            output="Provide either 'goal' (single mode) or 'tasks' (batch mode).",
            is_error=True,
        )

    # ---- single mode ---------------------------------------------------

    async def _run_single(self, arguments: DelegateTaskInput, context: ToolExecutionContext) -> ToolResult:
        """Run a single subagent and return its result."""
        result = await self._run_subagent(
            goal=arguments.goal or "",
            context_text=arguments.context,
            tools_whitelist=arguments.tools,
            max_turns=arguments.max_turns,
            parent_context=context,
            depth=context.metadata.get("_delegate_depth", 0),
        )
        if result.get("error"):
            return ToolResult(output=json.dumps(result), is_error=True)

        return ToolResult(
            output=self._format_single_result(result),
            metadata=result,
        )

    # ---- batch mode ----------------------------------------------------

    async def _run_batch(self, arguments: DelegateTaskInput, context: ToolExecutionContext) -> ToolResult:
        """Run multiple subagents in parallel and return all results."""
        tasks = arguments.tasks or []
        if not tasks:
            return ToolResult(output="No tasks provided.", is_error=True)

        max_concurrent = _get_max_concurrent()
        depth = context.metadata.get("_delegate_depth", 0)

        # Cap concurrent children.
        sem = asyncio.Semaphore(max_concurrent)

        async def _run_with_sem(task_args):
            async with sem:
                return await self._run_subagent(**task_args)

        # Build coroutines.
        coros = []
        for i, task in enumerate(tasks):
            goal = task.get("goal", "")
            if not goal:
                continue
            coros.append(
                _run_with_sem({
                    "goal": goal,
                    "context_text": task.get("context"),
                    "tools_whitelist": task.get("tools", arguments.tools),
                    "max_turns": task.get("max_turns", arguments.max_turns),
                    "parent_context": context,
                    "depth": depth,
                    "task_label": f"Task {i + 1}",
                })
            )

        if not coros:
            return ToolResult(output="No valid tasks (all missing 'goal').", is_error=True)

        results = await asyncio.gather(*coros, return_exceptions=True)

        return ToolResult(
            output=self._format_batch_results(results, len(coros)),
            metadata={
                "results": [
                    r if isinstance(r, dict) else {"error": str(r), "status": "failed"}
                    for r in results
                ],
                "total_duration_seconds": 0,  # filled below if we track it
            },
        )

    # ---- subagent runner -----------------------------------------------

    async def _run_subagent(
        self,
        *,
        goal: str,
        context_text: str | None,
        tools_whitelist: str | None,
        max_turns: int,
        parent_context: ToolExecutionContext,
        depth: int,
        task_label: str = "Task",
    ) -> dict[str, Any]:
        """Run a single subagent to completion with timeout. Returns a structured result dict."""
        start_time = time.monotonic()

        try:
            # Extract api_client and model from parent context.
            # These are injected by the runtime via tool_metadata (see fix below).
            api_client = parent_context.metadata.get("api_client")
            model = parent_context.metadata.get("model", "unknown")
            max_tokens = parent_context.metadata.get("max_tokens", 4096)

            if api_client is None:
                return {
                    "goal": goal,
                    "label": task_label,
                    "status": "failed",
                    "exit_reason": "no_api_client",
                    "result": "",
                    "error": (
                        "No API client available in parent context. "
                        "The runtime must inject 'api_client' into tool_metadata."
                    ),
                    "turns": 0,
                    "duration_seconds": 0,
                    "usage": {"input": 0, "output": 0},
                }

            # Build the subagent's tool registry with restricted tools.
            child_registry = self._build_child_registry(tools_whitelist, parent_context)

            # Build the subagent's system prompt.
            system_prompt = self._build_child_prompt(goal, context_text)

            # Build a child QueryEngine with fresh conversation.
            from niaharness.engine.query_engine import QueryEngine
            from niaharness.permissions.checker import PermissionChecker
            from niaharness.config.settings import PermissionSettings

            # Install subagent approval callback — auto-deny by default.
            approval_cb = _get_subagent_approval_callback()

            async def _subagent_permission_prompt(tool_name: str, description: str) -> bool:
                return approval_cb(description, tool_name)

            # Propagate parent's tool_metadata + depth + api_client to child.
            child_tool_metadata = dict(parent_context.metadata)
            child_tool_metadata["_delegate_depth"] = depth + 1
            # Remove api_client from tool_metadata to avoid confusion —
            # it's passed directly to QueryEngine, not via metadata.
            child_tool_metadata.pop("api_client", None)
            child_tool_metadata.pop("model", None)
            child_tool_metadata.pop("max_tokens", None)

            child_engine = QueryEngine(
                api_client=api_client,
                tool_registry=child_registry,
                permission_checker=PermissionChecker(PermissionSettings()),
                cwd=parent_context.cwd,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                max_turns=min(max_turns, _get_max_turns()),
                permission_prompt=_subagent_permission_prompt,
                tool_metadata=child_tool_metadata,
            )

            # Run the subagent with a timeout.
            last_text = ""
            total_usage = {"input": 0, "output": 0}
            turn_count = 0
            exit_reason = "completed"

            try:
                async with asyncio.timeout(_get_timeout()):
                    async for event in child_engine.submit_message(goal):
                        from niaharness.engine.stream_events import (
                            AssistantTextDelta,
                            AssistantTurnComplete,
                        )

                        if isinstance(event, AssistantTextDelta):
                            last_text += event.text
                        elif isinstance(event, AssistantTurnComplete):
                            if event.message.text:
                                last_text = event.message.text
                            if event.usage:
                                total_usage["input"] += event.usage.input_tokens
                                total_usage["output"] += event.usage.output_tokens
                            turn_count += 1

            except asyncio.TimeoutError:
                logger.warning("Subagent timed out after %ss: %s", _get_timeout(), goal[:80])
                exit_reason = "timeout"
                last_text = last_text or "(subagent timed out)"

            duration = time.monotonic() - start_time

            return {
                "goal": goal,
                "label": task_label,
                "status": "completed" if exit_reason == "completed" else "failed",
                "exit_reason": exit_reason,
                "result": last_text.strip() or "(no output)",
                "turns": turn_count,
                "duration_seconds": round(duration, 2),
                "usage": total_usage,
                "model": model,
                "error": None,
            }

        except Exception as exc:
            logger.exception("Subagent failed")
            duration = time.monotonic() - start_time
            return {
                "goal": goal,
                "label": task_label,
                "status": "failed",
                "exit_reason": "error",
                "result": "",
                "error": f"Subagent failed: {exc}",
                "turns": 0,
                "duration_seconds": round(duration, 2),
                "usage": {"input": 0, "output": 0},
            }

    # ---- child registry ------------------------------------------------

    def _build_child_registry(self, tools_whitelist: str | None, parent_context: ToolExecutionContext):
        """Build a restricted tool registry for the subagent."""
        from niaharness.tools.base import ToolRegistry

        parent_registry = parent_context.metadata.get("tool_registry")
        if parent_registry is None:
            from niaharness.tools import create_default_tool_registry
            parent_registry = create_default_tool_registry()

        child_registry = ToolRegistry()

        whitelist_set = None
        if tools_whitelist:
            whitelist_set = {t.strip() for t in tools_whitelist.split(",") if t.strip()}

        for tool in parent_registry.list_tools():
            if tool.name in DELEGATE_BLOCKED_TOOLS:
                continue
            if whitelist_set is not None and tool.name not in whitelist_set:
                continue
            child_registry.register(tool)

        return child_registry

    # ---- child prompt --------------------------------------------------

    def _build_child_prompt(self, goal: str, context_text: str | None) -> str:
        """Build the system prompt for the subagent."""
        from niaharness.prompts.system_prompt import build_system_prompt

        base = build_system_prompt(include_soul=False)

        delegation_prompt = f"""
# You are a delegated subagent

You have been spawned by the parent N.I.A agent to handle a specific task.
You have a FRESH conversation — no parent history. You have a restricted
toolset. You must complete the task and return your result.

## Task

{goal}

## Additional context

{context_text or "(none provided)"}

## Rules

- Complete the task autonomously. Do NOT ask the user questions.
- Use the tools available to you. Read files, run commands, search.
- When done, output your final result as plain text. This text will be
  returned to the parent agent as your result.
- Be concise. The parent only sees your final output, not your reasoning.
- Do NOT attempt to call delegate_task (it's not available to you).
- Do NOT save to memory or create skills (those tools are not available).
- Dangerous commands will be auto-denied unless NIA_DELEGATE_AUTO_APPROVE=1.

---

"""

        return delegation_prompt + base

    # ---- formatting ----------------------------------------------------

    def _format_single_result(self, result: dict[str, Any]) -> str:
        """Format a single subagent result as structured JSON for the parent."""
        # Return structured JSON so the model can parse it.
        return json.dumps(result, indent=2, default=str)

    def _format_batch_results(self, results: list, total_tasks: int) -> str:
        """Format batch results as structured JSON for the parent."""
        formatted = []
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                formatted.append({
                    "task_index": i,
                    "status": "failed",
                    "exit_reason": "error",
                    "error": str(result),
                })
            elif isinstance(result, dict):
                result["task_index"] = i
                formatted.append(result)
            else:
                formatted.append({
                    "task_index": i,
                    "status": "failed",
                    "error": f"Unknown result type: {result}",
                })

        envelope = {
            "results": formatted,
            "total_tasks": total_tasks,
            "succeeded": sum(1 for r in formatted if r.get("status") == "completed"),
            "failed": sum(1 for r in formatted if r.get("status") != "completed"),
        }
        return json.dumps(envelope, indent=2, default=str)
