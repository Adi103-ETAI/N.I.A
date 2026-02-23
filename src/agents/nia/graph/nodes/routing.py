"""NIA Graph Nodes — Routing Functions & AI Router Node.

Contains the conditional edge functions that drive the NIA LangGraph
state machine, plus the router_node that uses DecisionCore (LLM) to
decide which agent handles each user request.

    router_node      — AI-based routing: chat / system / swarm
    route_from_router — Conditional edge: reads state["next"]
    route_from_tara   — Conditional edge: handles TARA → supervisor loop
"""
from __future__ import annotations

from src.core.logger import setup_logger
from src.agents.nia.state import (
    AgentState,
    AGENT_SUPERVISOR,
    AGENT_END,
    safe_get_content,
)

try:
    from langchain_core.messages import HumanMessage
    _HAS_LANGCHAIN_MESSAGES = True
except ImportError:
    _HAS_LANGCHAIN_MESSAGES = False
    HumanMessage = None  # type: ignore

logger = setup_logger("NIA.Nodes.Routing")


# =============================================================================
# AI Router Node (Phase 3)
# =============================================================================

async def router_node(state: AgentState) -> AgentState:
    """AI Decision Node — routes user input to the correct agent.

    Uses DecisionCore (LLM) to classify the request and sets
    ``state["next"]`` to one of: ``supervisor``, ``tara``, ``docker``.

    Routing map:
        chat   → supervisor  (conversational)
        system → tara        (desktop automation)
        swarm  → docker      (Docker skill execution)

    Args:
        state: Current AgentState.

    Returns:
        State with ``next`` and ``metadata`` updated.
    """
    logger.info("🧭 Executing Router Node")

    from src.agents.nia.decision.router import DecisionCore

    # Get the user's raw input (prefer explicit field, fallback to last message)
    user_input = state.get("user_input", "")
    if not user_input:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                user_input = safe_get_content(msg)
                break

    if not user_input:
        logger.warning("Router received empty input — defaulting to Chat.")
        return {**state, "next": "supervisor"}

    # Consult DecisionCore
    router   = DecisionCore()
    decision = await router.aroute(user_input)

    target    = decision.target
    skill     = decision.skill
    reasoning = decision.reasoning

    logger.info(f"👉 Route: {target.upper()} | Skill: {skill} | Why: {reasoning}")

    # Update metadata
    new_meta = state.get("metadata", {}).copy()
    new_meta["routing_reason"] = reasoning
    if skill:
        new_meta["target_skill"] = skill
        new_meta["skill_query"]  = user_input

    # Map target → next node name
    next_node_map = {
        "swarm":  "docker",
        "system": "tara",
        "chat":   "supervisor",
    }
    next_node = next_node_map.get(target, "supervisor")

    return {**state, "next": next_node, "metadata": new_meta}


# =============================================================================
# Conditional Edge Functions
# =============================================================================

def route_from_router(state: AgentState) -> str:
    """Conditional edge: read the router's decision from state.

    Called by LangGraph after router_node to select the next graph node.

    Returns:
        Next node name (e.g. ``supervisor``, ``tara``, ``docker``).
    """
    next_node = state.get("next", "supervisor")
    logger.info(f"Highway Switch → {next_node}")
    return next_node


def route_from_tara(state: AgentState) -> str:
    """Conditional edge after TARA execution.

    Enables the Active Listening loop — TARA can request a return to the
    supervisor (e.g. after saving a preference) by setting ``next`` to
    ``supervisor`` in its return state.

    Returns:
        ``AGENT_SUPERVISOR`` if TARA requested continuation,
        ``AGENT_END`` otherwise.
    """
    next_agent = state.get("next", AGENT_END)

    if next_agent in ("supervisor", AGENT_SUPERVISOR):
        logger.debug("🔄 TARA → Supervisor: Active Listening loop")
        return AGENT_SUPERVISOR

    return AGENT_END


__all__ = ["router_node", "route_from_router", "route_from_tara"]
