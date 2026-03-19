"""NIA Graph Nodes — Agent Nodes.

Contains the core agent-facing node functions:

    supervisor_node   — Runs the NIA Supervisor (conversational LLM)
    iris_node         — Delegates to IRIS vision agent (sync → thread)
    general_assistant — Standalone chat node for non-automation queries
"""
from __future__ import annotations

import asyncio

from src.core.logger import setup_logger

try:
    from langchain_core.messages import SystemMessage, AIMessage
    _HAS_LANGCHAIN_MESSAGES = True
except ImportError:
    _HAS_LANGCHAIN_MESSAGES = False
    SystemMessage = None  # type: ignore
    AIMessage     = None  # type: ignore

from src.agents.nia.state import AgentState, safe_get_content
from .helpers import get_prompts, asummarize_oldest

logger = setup_logger("NIA.Nodes.Agents")


# =============================================================================
# Supervisor Node
# =============================================================================

async def supervisor_node(state: AgentState, supervisor, summarize_llm=None) -> AgentState:
    """Run the NIA Supervisor with smart context-window management.

    Steps:
        1. Inject the latest dynamic system prompt into the supervisor.
        2. Compress message history if it exceeds MAX_HISTORY.
        3. Delegate to supervisor.aprocess(state).

    Args:
        state:         Current AgentState.
        supervisor:    Supervisor instance (injected by builder).
        summarize_llm: Optional LLM to use for summarization;
                       lazy-loaded from ModelManager if None.
    """
    logger.debug("Executing supervisor node (Async)")

    # Refresh dynamic prompt (e.g. persona changes at runtime)
    prompts = get_prompts()
    if hasattr(supervisor, "system_prompt"):
        supervisor.system_prompt = prompts.get("supervisor", supervisor.system_prompt)

    # Compress history if needed
    current_messages = state.get("messages", [])
    clean_messages   = await asummarize_oldest(current_messages, summarize_llm)
    if len(clean_messages) < len(current_messages):
        state = {**state, "messages": clean_messages}

    return await supervisor.aprocess(state)


# =============================================================================
# IRIS Node
# =============================================================================

async def iris_node(state: AgentState, iris) -> AgentState:
    """Run IRIS asynchronously via its native aprocess() method.

    Uses IrisAgent.aprocess() which internally offloads the sync vision
    call to a thread pool.  This never blocks the event loop.

    Args:
        state: Current AgentState.
        iris:  IRIS agent instance (injected by builder).
    """
    logger.debug("Executing IRIS node (async via aprocess)")
    return await iris.aprocess(state)


# =============================================================================
# General Assistant Node
# =============================================================================

async def general_assistant(state: AgentState) -> AgentState:
    """Standalone chat node for non-automation, non-TARA queries.

    Uses the smart LLM with the unified NIA persona system prompt.
    If LLM is unavailable, returns a graceful error message.

    Args:
        state: Current AgentState.

    Returns:
        Updated AgentState with the AI response appended.
    """
    logger.info("💬 Routing to General Assistant (Chat)")

    try:
        from src.models.manager import ModelManager
        llm = ModelManager().get_smart_model()

        if not llm:
            if _HAS_LANGCHAIN_MESSAGES:
                err = AIMessage(content="I'm sorry, I couldn't connect to the AI service.")
                return {**state, "messages": state.get("messages", []) + [err]}
            return state

        messages = state.get("messages", [])

        # Inject unified persona as system message if not already present
        from src.persona.profile import get_system_prompt
        system_prompt_text = get_system_prompt()

        if _HAS_LANGCHAIN_MESSAGES:
            if not (messages and isinstance(messages[0], SystemMessage)):
                messages = [SystemMessage(content=system_prompt_text)] + messages
        else:
            if not (messages and isinstance(messages[0], dict) and messages[0].get("role") == "system"):
                messages = [{"role": "system", "content": system_prompt_text}] + messages

        response         = await llm.ainvoke(messages)
        response_content = safe_get_content(response)

        logger.debug(f"Chat response: {response_content[:100]}...")

        if _HAS_LANGCHAIN_MESSAGES:
            new_messages = messages + [AIMessage(content=response_content)]
        else:
            new_messages = messages + [{"role": "assistant", "content": response_content}]

        return {**state, "messages": new_messages, "final_response": response_content}

    except Exception as e:
        logger.error(f"General assistant error: {e}")
        err_text = f"I encountered an error: {e}"
        if _HAS_LANGCHAIN_MESSAGES:
            return {**state, "messages": state.get("messages", []) + [AIMessage(content=err_text)]}
        return state


__all__ = ["supervisor_node", "iris_node", "general_assistant"]
