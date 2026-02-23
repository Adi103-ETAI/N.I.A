"""NIA Graph Nodes — Docker & TARA 2.0 Execution Nodes.

Houses the two nodes that delegate work outside the NIA process:

    call_tara_2   — Hands off automation tasks to the TARA 2.0 subgraph.
    docker_node   — Routes swarm tasks to Docker via DockerBridge.

Both nodes sanitize the user goal before dispatch to strip memory-context
injection artifacts (``[MEMORY CONTEXT]`` / ``User Input:`` prefixes).
"""
from __future__ import annotations

from src.core.logger import setup_logger

try:
    from langchain_core.messages import HumanMessage, AIMessage
    _HAS_LANGCHAIN_MESSAGES = True
except ImportError:
    _HAS_LANGCHAIN_MESSAGES = False
    HumanMessage = None  # type: ignore
    AIMessage    = None  # type: ignore

from src.agents.nia.state import AgentState, AGENT_END, safe_get_content

logger = setup_logger("NIA.Nodes.Docker")

# ---------------------------------------------------------------------------
# TARA 2.0 graph — optional (graceful degradation if not available)
# ---------------------------------------------------------------------------
try:
    from src.agents.tara.graph import tara_app, create_initial_tara_state
    _HAS_TARA_2 = True
    logger.debug("✅ TARA 2.0 graph imported successfully")
except ImportError as e:
    _HAS_TARA_2 = False
    tara_app = None               # type: ignore
    create_initial_tara_state = None  # type: ignore
    logger.warning(f"⚠️ TARA 2.0 graph not available: {e}")


# =============================================================================
# Shared Helper: Goal Sanitizer
# =============================================================================

def _sanitize_goal(state: AgentState) -> str:
    """Extract and clean the user goal from state.

    Removes ``[MEMORY CONTEXT]`` and ``User Input:`` prefixes that the
    engine injects before handing off to agent nodes.

    Returns:
        Clean user goal string.
    """
    raw_content = ""

    # Source of truth: last HumanMessage in history
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            raw_content = safe_get_content(msg)
            break
        elif isinstance(msg, dict) and msg.get("type") == "human":
            raw_content = safe_get_content(msg)
            break

    if not raw_content:
        raw_content = state.get("user_input", "")

    # Strip memory-context injection artifacts
    if "User Input:" in raw_content:
        clean = raw_content.split("User Input:")[-1].strip()
        logger.info(f"🧼 Sanitized via Split: '{clean}'")
        return clean
    elif "[MEMORY CONTEXT]" in raw_content:
        clean = raw_content.replace("[MEMORY CONTEXT]", "").strip()
        logger.info(f"🧼 Sanitized via Replace: '{clean}'")
        return clean

    return raw_content.strip()


# =============================================================================
# TARA 2.0 Node
# =============================================================================

async def call_tara_2(state: AgentState) -> AgentState:
    """Hand off an automation task to the TARA 2.0 LangGraph subgraph.

    Flow:
        1. Extract + sanitize user goal from message history.
        2. Build TARA input state dict.
        3. Await tara_app.ainvoke(tara_input).
        4. Extract final_response from TARA result.
        5. Return updated NIA state with TARA's response appended.

    Falls back to a graceful error message if TARA 2.0 is unavailable.
    """
    if not _HAS_TARA_2 or not tara_app:
        logger.error("TARA 2.0 not available — returning error")
        if _HAS_LANGCHAIN_MESSAGES:
            err = AIMessage(content="I'm sorry, the automation system is not available right now.")
            return {**state, "messages": state.get("messages", []) + [err]}
        return state

    logger.info("🚀 Routing to TARA 2.0 SubGraph...")

    try:
        clean_goal = _sanitize_goal(state)

        tara_input = {
            "messages":         state.get("messages", []),
            "user_goal":        clean_goal,
            "screen_context":   None,
            "active_app":       None,
            "clipboard":        None,
            "last_error":       None,
            "tool_calls_pending": False,
            "iteration_count":  0,
            "final_response":   None,
            "metadata":         state.get("metadata", {}),
        }

        result         = await tara_app.ainvoke(tara_input)
        final_response = result.get("final_response", "")

        # Fallback: last AI message from TARA's history
        if not final_response:
            for msg in reversed(result.get("messages", [])):
                if getattr(msg, "type", None) == "ai" or msg.__class__.__name__ == "AIMessage":
                    final_response = safe_get_content(msg)
                    break

        final_response = final_response or "Task completed."
        logger.info(f"✅ TARA 2.0 done. Response: {final_response[:100]}...")

        if _HAS_LANGCHAIN_MESSAGES:
            new_messages = state.get("messages", []) + [AIMessage(content=final_response)]
        else:
            new_messages = state.get("messages", []) + [{"role": "assistant", "content": final_response}]

        return {**state, "messages": new_messages, "final_response": final_response, "next": AGENT_END}

    except Exception as e:
        logger.error(f"❌ TARA 2.0 failed: {e}")
        import traceback; traceback.print_exc()
        err_text = f"I encountered an error while automating: {e}"
        if _HAS_LANGCHAIN_MESSAGES:
            return {**state, "messages": state.get("messages", []) + [AIMessage(content=err_text)], "next": AGENT_END}
        return {**state, "next": AGENT_END}


# =============================================================================
# Docker Swarm Node
# =============================================================================

async def docker_node(state: AgentState) -> AgentState:
    """Route a swarm task to the Docker container engine via DockerBridge.

    Requires a skill name in ``state["metadata"]["target_skill"]`` — set
    by the router_node after DecisionCore selects a skill.

    Flow:
        1. Read skill name + query from metadata.
        2. Load skill metadata from the skill registry.
        3. Build MissionManifest with host workdir (The Wormhole).
        4. Execute via DockerBridge.execute_mission().
        5. Return result or error as an AIMessage.
    """
    import os
    import uuid
    from langchain_core.messages import AIMessage

    from src.infrastructure.container_engine.bridge import DockerBridge
    from src.infrastructure.container_engine.manager import DockerEngine
    from src.agents.soldiers.schemas import MissionManifest
    from src.core.skills.loader import load_docker_skills, get_skill_source_code

    logger.info("🐳 Routing to Docker Node")

    meta       = state.get("metadata", {})
    skill_name = meta.get("target_skill")
    query      = meta.get("skill_query")

    if not skill_name:
        return {**state, "messages": state.get("messages", []) + [AIMessage(content="Error: No skill specified.")]}

    all_skills   = load_docker_skills()
    skill_config = next((s for s in all_skills if s["name"] == skill_name), None)

    if not skill_config:
        return {**state, "messages": state.get("messages", []) + [AIMessage(content=f"Error: Skill '{skill_name}' not found.")]}

    host_workdir = meta.get("workdir", os.getcwd())

    manifest = MissionManifest(
        task_id      = str(uuid.uuid4()),
        soldier_type = "coding",
        objective    = query,
        code         = get_skill_source_code(skill_name),
        runtime      = skill_config.get("runtime", "python"),
        host_workdir = host_workdir,
        pty          = skill_config.get("pty", False),
        dependencies = skill_config.get("dependencies", []),
        user_query   = query,
    )

    try:
        engine = DockerEngine()
        bridge = DockerBridge(engine)
        result = bridge.execute_mission(manifest)

        response_text = result.output
        if result.error:
            response_text = f"Error: {result.error}\nOutput: {result.output}"

        logger.info(f"🐳 Docker complete. Result: {len(response_text)} chars")
        return {
            **state,
            "messages":       state.get("messages", []) + [AIMessage(content=response_text)],
            "final_response": response_text,
        }

    except Exception as e:
        logger.error(f"Docker execution failed: {e}")
        return {**state, "messages": state.get("messages", []) + [AIMessage(content=f"Docker execution failed: {e}")]}


__all__ = ["call_tara_2", "docker_node", "_HAS_TARA_2"]
