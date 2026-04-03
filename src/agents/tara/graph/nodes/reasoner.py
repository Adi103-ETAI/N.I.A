"""TARA Graph Nodes — Reasoner Node.

Node 1 of the TARA pipeline. Calls the LLM with the current state and
tool bindings to produce the next action (tool call or final answer).

Data flow:
    state → build_tara_context() → LLM.bind_tools() → ainvoke() → TaraStateUpdate
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage, SystemMessage

from src.core.logger import setup_logger
from src.core.config import get_settings
from src.core.schema.states import safe_get_content
from src.agents.tara.graph.state import TaraState, TaraStateUpdate
from src.agents.tara.graph.prompts import TARA_SYSTEM_PROMPT, build_tara_context
from src.capabilities.interface import get_tara_tools
from src.models.manager import get_smart_model

from src.core.utils.text_utils import _parse_llama_tool_calls, parse_tool_call_json

logger = setup_logger("TARA.Nodes.Reasoner")
settings = get_settings()


# =============================================================================
# LLM Factory
# =============================================================================

def _get_llm():
    """Return a configured LLM via ModelManager (hot-swap aware).

    Fetches the model fresh on every call so provider switches in
    ModelManager take effect immediately without a restart.
    """
    return get_smart_model(temperature=settings.LLM_TEMPERATURE)


# =============================================================================
# Node 1: Reasoner
# =============================================================================

async def reasoner(state: TaraState) -> TaraStateUpdate:
    """Main reasoning node — generates tool calls via dynamic context.

    Pipeline:
        1. Check iteration limit (hard stop)
        2. Build context string from current state
        3. Prepend system message with context
        4. Bind tools to LLM + async invoke
        5. Fallback: parse Llama <|python_tag|> format if bind_tools missed it

    Args:
        state: Current TaraState with conversation history and context.

    Returns:
        TaraStateUpdate with new messages, iteration_count, and
        tool_calls_pending flag.
    """
    iteration = state.get("iteration_count", 0)

    # --- Hard stop: prevent infinite loops ---
    if iteration >= settings.MAX_ITERATIONS:
        logger.warning(f"Max iterations ({settings.MAX_ITERATIONS}) reached")
        return {
            "messages": [],
            "final_response": "Maximum iterations reached. Task may be incomplete.",
            "tool_calls_pending": False,
        }

    logger.info(f"Reasoner iteration {iteration + 1}")

    try:
        # Build dynamic context and system message
        context_str = build_tara_context(state)
        system_msg = SystemMessage(content=f"{TARA_SYSTEM_PROMPT}\n\n{context_str}")

        messages: List[BaseMessage] = list(state.get("messages", []))
        full_messages = [system_msg] + messages

        # Bind tools and invoke asynchronously
        tools = get_tara_tools()
        llm = _get_llm()
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(full_messages)

        has_tool_calls = hasattr(response, "tool_calls") and len(response.tool_calls) > 0

        # Fallback 1: parse Llama 3.1 <|python_tag|> format
        content_str = safe_get_content(response)
        if not has_tool_calls and content_str and "<|python_tag|>" in content_str:
            logger.warning("[REASONER] LLM didn't parse tool calls, using fallback parser")
            parsed_calls = _parse_llama_tool_calls(content_str)
            if parsed_calls:
                response.tool_calls = parsed_calls
                has_tool_calls = True
                logger.debug(f"[REASONER] Fallback parser found {len(parsed_calls)} tool call(s)")

        # Fallback 2: parse plain JSON tool payloads
        if not has_tool_calls and content_str:
            parsed_json_calls = parse_tool_call_json(content_str)
            if parsed_json_calls:
                response.tool_calls = parsed_json_calls
                has_tool_calls = True
                logger.debug(
                    "[REASONER] JSON parser found %d tool call(s)",
                    len(parsed_json_calls),
                )

        logger.debug(f"LLM response: {content_str[:100] if content_str else 'Tool call'}...")
        logger.info(f"[REASONER] tool_calls_pending={has_tool_calls}")

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "tool_calls_pending": has_tool_calls,
        }

    except Exception as e:
        logger.error(f"Reasoner error: {e}")
        return {
            "messages": [],
            "last_error": str(e),
            "final_response": f"Reasoning error: {e}",
        }


__all__ = ["reasoner", "_get_llm"]
