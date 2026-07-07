"""Delegate task tool — spawn isolated subagents for parallel work.

The audit (P1 #8) flagged that NIA's `agent` + `team_create` tools are a
faint shadow of Hermes's `delegate_task`. This tool implements the
Hermes-style delegation pattern:

1. Parent calls delegate_task with a goal + optional context.
2. A child QueryEngine is created with a FRESH conversation (no parent
   history) and a restricted toolset.
3. The child runs to completion (or until max_turns).
4. Only the child's final text result is returned to the parent — the
   parent never sees the child's intermediate tool calls or reasoning.
5. Supports batch mode: pass ``tasks`` array for parallel fan-out.

Blocked tools (children must never have access to):
- ``delegate_task`` (no recursive delegation)
- ``ask_user_question`` (no user interaction from a subagent)
- ``nia_memory`` (no writes to shared memory)

Safety:
- Max delegation depth: 2 (configurable via NIA_DELEGATE_MAX_DEPTH env var).
- Max iterations per child: 10 (configurable via NIA_DELEGATE_MAX_TURNS).
- Children inherit the parent's provider/model/api_key.
- Children run in a thread pool (concurrent for batch mode).

Reference: Hermes Agent's ``tools/delegate_tool.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_DEPTH = 2

# Tools that children must never have access to.
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "ask_user_question",  # no user interaction from subagent
        "nia_memory",  # no writes to shared memory
        "nia_session",  # no session persistence from subagent
        "skill_manage",  # no skill mutations from subagent
        "cron_create",  # no scheduling from subagent
        "cron_delete",
        "cron_toggle",
    ]
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class DelegateTaskInput(BaseModel):
    """Arguments for the delegate_task tool.

    Two modes:
    - Single: pass ``goal`` (+ optional ``context``).
    - Batch: pass ``tasks`` array of {goal, context, tools} dicts.

    The parent blocks until all children complete. The parent's context
    only sees the delegation call and the summary result — never the
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
            "Batch mode: array of tasks to run in parallel. Each task is a dict "
            "with 'goal' (required), 'context' (optional), 'tools' (optional "
            "comma-separated tool name list). When provided, 'goal' and "
            "'context' are ignored."
        ),
    )
    tools: str | None = Field(
        default=None,
        description=(
            "Comma-separated tool name whitelist for the subagent. "
            "If omitted, the subagent gets all tools except blocked ones "
            "(delegate_task, ask_user_question, nia_memory, etc.)."
        ),
    )
    max_turns: int = Field(
        default=DEFAULT_MAX_TURNS,
        ge=1,
        le=50,
        description="Maximum agentic turns per subagent.",
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
        "runs to completion. Only the final result is returned to the parent. "
        "Supports single-task and batch (parallel) modes. The parent's context "
        "only sees the delegation call and the summary — never the child's "
        "intermediate tool calls."
    )
    input_model = DelegateTaskInput

    def is_read_only(self, arguments: DelegateTaskInput) -> bool:
        # Subagents can write files / run bash — treat as a write operation.
        del arguments
        return False

    async def execute(self, arguments: DelegateTaskInput, context: ToolExecutionContext) -> ToolResult:
        # Check delegation depth.
        current_depth = context.metadata.get("_delegate_depth", 0)
        max_depth = int(os.environ.get("NIA_DELEGATE_MAX_DEPTH", str(DEFAULT_MAX_DEPTH)))
        if current_depth >= max_depth:
            return ToolResult(
                output=(
                    f"Delegation depth limit reached (depth={current_depth}, "
                    f"max={max_depth}). Raise NIA_DELEGATE_MAX_DEPTH env var "
                    f"if deeper nesting is required."
                ),
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
            return ToolResult(output=result["error"], is_error=True)

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

        depth = context.metadata.get("_delegate_depth", 0)

        # Run all subagents concurrently.
        coros = []
        for i, task in enumerate(tasks):
            goal = task.get("goal", "")
            if not goal:
                continue
            ctx_text = task.get("context")
            tools = task.get("tools", arguments.tools)
            max_turns = task.get("max_turns", arguments.max_turns)
            coros.append(
                self._run_subagent(
                    goal=goal,
                    context_text=ctx_text,
                    tools_whitelist=tools,
                    max_turns=max_turns,
                    parent_context=context,
                    depth=depth,
                    task_label=f"Task {i + 1}",
                )
            )

        results = await asyncio.gather(*coros, return_exceptions=True)

        # Format results.
        return ToolResult(
            output=self._format_batch_results(results, len(tasks)),
            metadata={"results": [r if isinstance(r, dict) else {"error": str(r)} for r in results]},
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
        """Run a single subagent to completion. Returns a result dict."""
        try:
            # Build the subagent's tool registry with restricted tools.
            child_registry = self._build_child_registry(tools_whitelist, parent_context)

            if child_registry is None:
                return {"error": "Could not build child tool registry — no API client available."}

            # Build the subagent's system prompt.
            system_prompt = self._build_child_prompt(goal, context_text)

            # Get the API client from the parent context.
            api_client = parent_context.metadata.get("api_client")
            if api_client is None:
                return {"error": "No API client available in parent context."}

            # Build a child QueryEngine with fresh conversation.
            from niaharness.engine.query_engine import QueryEngine
            from niaharness.permissions.checker import PermissionChecker
            from niaharness.config.settings import PermissionSettings

            child_engine = QueryEngine(
                api_client=api_client,
                tool_registry=child_registry,
                permission_checker=PermissionChecker(PermissionSettings()),
                cwd=parent_context.cwd,
                model=parent_context.metadata.get("model", "unknown"),
                system_prompt=system_prompt,
                max_tokens=parent_context.metadata.get("max_tokens", 4096),
                max_turns=max_turns,
            )

            # Run the subagent.
            from niaharness.engine.messages import ConversationMessage

            last_text = ""
            total_usage = {"input": 0, "output": 0}
            turn_count = 0

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

            return {
                "goal": goal,
                "label": task_label,
                "result": last_text.strip() or "(no output)",
                "turns": turn_count,
                "usage": total_usage,
                "error": None,
            }

        except Exception as exc:
            logger.exception("Subagent failed")
            return {
                "goal": goal,
                "label": task_label,
                "result": "",
                "error": f"Subagent failed: {exc}",
            }

    # ---- child registry ------------------------------------------------

    def _build_child_registry(self, tools_whitelist: str | None, parent_context: ToolExecutionContext):
        """Build a restricted tool registry for the subagent.

        If ``tools_whitelist`` is given, only those tools are included
        (minus blocked tools). Otherwise, all tools except blocked ones.
        """
        from niaharness.tools.base import ToolRegistry

        # Get the parent's tool registry.
        parent_registry = parent_context.metadata.get("tool_registry")
        if parent_registry is None:
            # Fall back to creating a fresh registry.
            from niaharness.tools import create_default_tool_registry

            parent_registry = create_default_tool_registry()

        child_registry = ToolRegistry()

        # Parse whitelist if given.
        whitelist_set = None
        if tools_whitelist:
            whitelist_set = {t.strip() for t in tools_whitelist.split(",") if t.strip()}

        for tool in parent_registry.list_tools():
            # Skip blocked tools.
            if tool.name in DELEGATE_BLOCKED_TOOLS:
                continue
            # Skip if not in whitelist.
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

---

"""

        return delegation_prompt + base

    # ---- formatting ----------------------------------------------------

    def _format_single_result(self, result: dict[str, Any]) -> str:
        """Format a single subagent result for the parent."""
        lines = [
            f"Subagent result ({result.get('turns', 0)} turns):",
            "",
            result.get("result", "(no output)"),
        ]
        usage = result.get("usage", {})
        if usage:
            lines.append("")
            lines.append(
                f"Usage: {usage.get('input', 0)} input / {usage.get('output', 0)} output tokens"
            )
        return "\n".join(lines)

    def _format_batch_results(self, results: list, total_tasks: int) -> str:
        """Format batch results for the parent."""
        lines = [f"Batch delegation complete ({total_tasks} tasks):", ""]
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                lines.append(f"## Task {i}: FAILED")
                lines.append(f"Error: {result}")
            elif isinstance(result, dict):
                if result.get("error"):
                    lines.append(f"## Task {i}: ERROR")
                    lines.append(f"Goal: {result.get('goal', '?')}")
                    lines.append(f"Error: {result['error']}")
                else:
                    lines.append(f"## Task {i}: {result.get('label', f'Task {i}')}")
                    lines.append(f"Goal: {result.get('goal', '?')}")
                    lines.append(f"Result ({result.get('turns', 0)} turns):")
                    lines.append(result.get("result", "(no output)"))
                    usage = result.get("usage", {})
                    if usage:
                        lines.append(
                            f"Usage: {usage.get('input', 0)} in / {usage.get('output', 0)} out"
                        )
            else:
                lines.append(f"## Task {i}: UNKNOWN ({result})")
            lines.append("")
        return "\n".join(lines)
