"""TARA Graph Nodes — Tool Executor Node.

Node 2 of the TARA pipeline. Executes all pending tool calls from the
last AIMessage in parallel using asyncio.gather, then updates state
context from the results.

Security:
    - High-risk tools are cleared through the Warden before execution.
    - A SecurityError from the Warden produces a ToolMessage with a
      🚫 prefix and does NOT crash the node.

Memory:
    - Successful tool chains are stored in the procedural memory layer
      (NetworkX) so NIA can recall them on similar future queries.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from src.core.logger import setup_logger
from src.core.schema.states import safe_get_content
from src.agents.tara.graph.state import TaraState, TaraStateUpdate
from src.capabilities.interface import get_tara_tools
from src.core.security import get_warden, SecurityError

logger = setup_logger("TARA.Nodes.Executor")


# =============================================================================
# Context Extractor (helper used by tool_executor)
# =============================================================================

def _extract_context_from_results(tool_messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """Extract context state updates from tool result messages.

    Inspects tool outputs and populates relevant state keys so the
    next reasoner iteration has fresh context.

    Extracted keys:
        screen_context — from dump_ui_tree results
        active_app     — from launch_app results
        clipboard      — from get_clipboard results
        last_error     — from any ❌ / Error / failed result
    """
    updates: Dict[str, Any] = {}

    for msg in tool_messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = safe_get_content(msg)
        tool_name = getattr(msg, "name", "")

        if tool_name == "dump_ui_tree" or "UI Elements" in content:
            updates["screen_context"] = content

        if tool_name == "launch_app" and "launched" in content.lower():
            if ":" in content:
                updates["active_app"] = content.split(":")[1].strip().split()[0]

        if tool_name == "get_clipboard" or "Clipboard" in content:
            if ":" in content:
                updates["clipboard"] = content.split(":", 1)[1].strip()[:200]

        if content.startswith("❌") or "Error" in content or "failed" in content.lower():
            updates["last_error"] = content

    return updates


# =============================================================================
# Node 2: Tool Executor
# =============================================================================

async def tool_executor(state: TaraState) -> TaraStateUpdate:
    """Async Tool Executor Node — parallel native execution.

    Executes all pending tool calls from the last AIMessage concurrently
    via asyncio.gather. Uses ainvoke() polymorphically so both sync and
    async tools are dispatched correctly.

    Pipeline:
        1. Extract tool_calls from last message
        2. Build task list (_exec_single coroutines)
        3. asyncio.gather → parallel execution
        4. Learn successful sequences (procedural memory)
        5. Extract context updates from results

    Args:
        state: Current TaraState with tool calls pending.

    Returns:
        TaraStateUpdate with tool result ToolMessages and context updates.
    """
    tools = get_tara_tools()
    tools_map: Dict[str, Any] = {tool.name: tool for tool in tools}

    messages = state.get("messages", [])
    if not messages:
        logger.warning("[TOOL_EXECUTOR] No messages in state")
        return {"messages": [], "tool_calls_pending": False}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        logger.warning("[TOOL_EXECUTOR] No tool_calls in last message")
        return {"messages": [], "tool_calls_pending": False}

    logger.info(f"[TOOL_EXECUTOR] 🚀 Executing {len(tool_calls)} tool(s) in parallel")

    # -----------------------------------------------------------------------
    # Inner coroutine: execute a single tool call with security check
    # -----------------------------------------------------------------------
    async def _exec_single(t_call):
        t_name = t_call.get("name", "")
        t_args = t_call.get("args", {})
        t_id   = t_call.get("id", f"call_{t_name}")

        logger.info(f"[TOOL_EXECUTOR] → {t_name}({t_args})")

        tool = tools_map.get(t_name)
        if tool is None:
            err = f"❌ Tool not found: {t_name}"
            logger.error(err)
            return ToolMessage(content=err, tool_call_id=t_id, name=t_name)

        try:
            # Security gate: high-risk tools require Warden approval
            sec_level = tool.metadata.get("security_level", "host_standard")
            if sec_level == "high_risk":
                logger.warning(f"🛡️ Warden Intercept: Validating '{t_name}'...")
                get_warden().check_permission(t_name, t_args)
                logger.info(f"🛡️ Warden Approved: '{t_name}'")

            # Polymorphic dispatch: ainvoke() returns value or coroutine
            result = tool.ainvoke(t_args)
            if inspect.iscoroutine(result) or asyncio.isfuture(result):
                result = await result

            result_str = str(result) if result is not None else f"✅ {t_name} completed"
            logger.info(f"[TOOL_EXECUTOR] ✅ {t_name}: {result_str[:80]}...")
            return ToolMessage(content=result_str, tool_call_id=t_id, name=t_name)

        except SecurityError as se:
            err = f"🚫 SECURITY DENIED: {se}"
            logger.warning(f"[TOOL_EXECUTOR] {err}")
            return ToolMessage(content=err, tool_call_id=t_id, name=t_name)

        except Exception as e:
            err = f"❌ {t_name} failed: {e}"
            logger.error(f"[TOOL_EXECUTOR] {err}")
            return ToolMessage(content=err, tool_call_id=t_id, name=t_name)

    # -----------------------------------------------------------------------
    # Parallel execution
    # -----------------------------------------------------------------------
    tool_messages = await asyncio.gather(*[_exec_single(tc) for tc in tool_calls])

    logger.info(f"[TOOL_EXECUTOR] ✓ Completed {len(tool_messages)} tool execution(s)")

    # -----------------------------------------------------------------------
    # Procedural memory: store successful tool chains
    # -----------------------------------------------------------------------
    if tool_messages and all("❌" not in safe_get_content(m) for m in tool_messages):
        try:
            from src.core.di import ServiceRegistry
            memory_svc = ServiceRegistry.get("memory")

            user_goal = state.get("user_goal")
            if not user_goal:
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, HumanMessage):
                        user_goal = safe_get_content(msg)[:100]
                        break

            if memory_svc and user_goal:
                tool_names = [tc.get("name") for tc in tool_calls if tc.get("name")]
                if tool_names:
                    memory_svc.add_skill_path(user_goal, tool_names)
                    logger.info(f"🧠 [SKILL] Learned: '{user_goal[:50]}' → {tool_names}")
        except Exception as e:
            logger.warning(f"🧠 [SKILL] Failed to save skill path: {e}")

    # Extract context updates from results and return
    context_updates = _extract_context_from_results(tool_messages)
    return {
        "messages": list(tool_messages),
        "tool_calls_pending": False,
        **context_updates,
    }


__all__ = ["tool_executor", "_extract_context_from_results"]
