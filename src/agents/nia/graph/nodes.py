"""N.I.A. Graph Nodes - Node functions for LangGraph state machine.

This module contains the node functions that operate on AgentState:
- supervisor_node: Routes requests and manages conversation
- tara_node: Executes tools and automation
- iris_node: Handles vision tasks
- Routing functions for conditional edges
"""
from __future__ import annotations

from src.core.logger import setup_logger
from typing import TYPE_CHECKING, List, Optional
import asyncio

logger = setup_logger("NIA.Nodes")

# Import state types
from src.agents.nia.state import (
    AgentState,
    AGENT_SUPERVISOR,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_END,
)

# LangChain messages for summarization
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    _HAS_LANGCHAIN_MESSAGES = True
except ImportError:
    _HAS_LANGCHAIN_MESSAGES = False
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore

from src.core.config import settings

import json
from pathlib import Path

def _load_vision_config() -> dict:
    """Load vision configuration from JSON file."""
    config_path = Path(__file__).resolve().parents[4] / "config" / "iris" / "triggers.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load triggers.json: {e}. Using defaults.")
        return {
            "triggers": {
                "screen": ["screen", "screenshot", "window"],
                "camera": ["camera", "webcam", "photo"],
                "actions": ["look at", "what do you see", "vision"]
            }
        }

# Cache the vision config at module level
_VISION_CONFIG = None

def get_vision_keywords() -> list:
    """Get all vision trigger keywords."""
    global _VISION_CONFIG
    if _VISION_CONFIG is None:
        _VISION_CONFIG = _load_vision_config()
    triggers = _VISION_CONFIG.get("triggers", {})
    return (
        triggers.get("screen", []) +
        triggers.get("camera", []) +
        triggers.get("actions", [])
)

# Logger already initialized at top of file via setup_logger("NIA.Nodes")


# =============================================================================
# Routing Config Loader (Lazy Loading Pattern)
# =============================================================================

_ROUTING_CONFIG: Optional[dict] = None


def _load_routing_config() -> dict:
    """Load routing configuration from JSON file.
    
    Uses lazy loading pattern - only loads on first access.
    
    Returns:
        Dict with tara_keywords, iris_keywords, and general_fallback.
    """
    config_path = Path(__file__).resolve().parents[4] / "config" / "nia" / "routing.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load routing.json: {e}. Using defaults.")
        # Fallback defaults (same as original hardcoded values)
        return {
            "tara_keywords": {
                "categories": {
                    "app_control": ["open", "launch", "run", "start", "execute", "close", "kill", "terminate", "exit", "quit", "stop"],
                    "interaction": ["click", "type", "press", "select"],
                    "navigation": ["search", "browse", "google", "navigate"],
                    "verification": ["check", "verify", "test"]
                }
            },
            "iris_keywords": {
                "categories": {
                    "perception": ["look", "see", "watch", "describe"],
                    "screen": ["screen", "screenshot", "monitor"],
                    "phrase": ["what is on"]
                }
            },
            "general_fallback": {"default_route": "general"}
        }


def get_routing_keywords() -> tuple:
    """Get TARA and IRIS routing keywords from config.
    
    Returns:
        Tuple of (tara_keywords: list, iris_keywords: list)
    """
    global _ROUTING_CONFIG
    if _ROUTING_CONFIG is None:
        _ROUTING_CONFIG = _load_routing_config()
    
    # Flatten TARA categories into single list
    tara_config = _ROUTING_CONFIG.get("tara_keywords", {})
    tara_keywords = []
    for keywords in tara_config.get("categories", {}).values():
        tara_keywords.extend(keywords)
    
    # Flatten IRIS categories into single list
    iris_config = _ROUTING_CONFIG.get("iris_keywords", {})
    iris_keywords = []
    for keywords in iris_config.get("categories", {}).values():
        iris_keywords.extend(keywords)
    
    return (tara_keywords, iris_keywords)



# =============================================================================
# Prompts Config Loader (Lazy Loading Pattern)
# =============================================================================

_PROMPTS_CONFIG: Optional[dict] = None

def _load_prompts_config() -> dict:
    """Load prompts configuration from JSON file."""
    config_path = Path(__file__).resolve().parents[4] / "config" / "nia" / "prompts.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load prompts.json: {e}. Using defaults.")
        return {
            "identity": "You are N.I.A., a helpful AI assistant.",
            "supervisor": "You are the Supervisor. Route questions to workers."
        }

def get_prompts() -> dict:
    """Get all system prompts."""
    global _PROMPTS_CONFIG
    if _PROMPTS_CONFIG is None:
        _PROMPTS_CONFIG = _load_prompts_config()
    return _PROMPTS_CONFIG


# =============================================================================
# Summarization Helper
# =============================================================================

def summarize_oldest(messages: List, llm=None) -> List:
    """Compress oldest messages into a summary if limit exceeded.
    
    This prevents context window overflow by summarizing old messages
    into a concise summary paragraph.
    
    Args:
        messages: List of message objects.
        llm: LLM to use for summarization (optional - will lazy-load if None).
        
    Returns:
        Compressed message list if over limit, original otherwise.
    """
    max_history = settings.MAX_HISTORY
    prune_count = settings.PRUNE_COUNT
    
    if len(messages) <= max_history:
        return messages
    
    if not _HAS_LANGCHAIN_MESSAGES:
        # Fallback: just slice if LangChain not available
        logger.warning("LangChain messages not available for summarization, truncating")
        return messages[:max_history]
    
    logger.info(f"🧹 Pruning: History length {len(messages)} > {max_history}. Compressing...")
    
    if not llm:
        try:
            # Lazy import to prevent circular dependency
            from src.models.manager import ModelManager
            manager = ModelManager()
            # Try fast model first (cheaper/faster), fallback to smart
            llm = manager.get_fast_model()
            if not llm:
                llm = manager.get_smart_model()
            if llm:
                logger.debug("Loaded LLM for summarization via ModelManager")
        except Exception as e:
            logger.error(f"Failed to load LLM for summarization: {e}")
            # Fail safe: just truncate instead of crashing
            return messages[-max_history:]
    
    # If still no LLM, truncate and return
    if not llm:
        logger.warning("No LLM available for summarization, truncating to recent messages")
        return messages[-max_history:]
    
    # 1. Protect the System Prompt (Index 0)
    system_prompt = messages[0]
    
    # 2. Identify the chunk to summarize
    to_summarize = messages[1:1 + prune_count]
    remaining = messages[1 + prune_count:]
    
    # 3. Generate Summary using LLM
    summary_request = [
        SystemMessage(content=(
            "You are a helpful assistant. Summarize the following conversation "
            "into a concise context paragraph. Preserve key constraints, facts, "
            "and user goals. Keep it under 200 words."
        )),
        HumanMessage(content=f"Conversation to summarize:\n{to_summarize}")
    ]
    
    try:
        # 🌊 ASYNC SUPPORT: If llm has ainvoke, use it.
        if hasattr(llm, 'invoke'):
            response = llm.invoke(summary_request)
        else:
            # Fallback
            response = llm(summary_request)
            
        summary_text = response.content if hasattr(response, 'content') else str(response)
        
        # 4. Create the Summary Message
        summary_msg = SystemMessage(content=f"📝 [PREVIOUS CONTEXT SUMMARY]: {summary_text}")
        
        # 5. Reconstruct: [System Prompt] + [New Summary] + [Recent Messages]
        new_messages = [system_prompt, summary_msg] + remaining
        logger.info(f"🧹 Pruned {prune_count} messages into summary. New length: {len(new_messages)}")
        return new_messages
        
    except Exception as e:
        logger.error(f"❌ Summarization failed: {e}")
        return messages  # Fail safe: return original list


async def asummarize_oldest(messages: List, llm=None) -> List:
    """Async version of summarize_oldest."""
    return await asyncio.to_thread(summarize_oldest, messages, llm)


# =============================================================================
# Node Functions
# =============================================================================

async def supervisor_node(state: AgentState, supervisor, summarize_llm=None) -> AgentState:
    """Supervisor node function with smart summarization (Async)."""
    logger.debug("Executing supervisor node (Async)")
    
    # Inject dynamic system prompt (Root Fix)
    prompts = get_prompts()
    if hasattr(supervisor, 'system_prompt'):
        supervisor.system_prompt = prompts.get("supervisor", supervisor.system_prompt)
    
    # 1. Manage Memory (Summarize if needed)
    current_messages = state.get("messages", [])
    # Call async summarizer
    clean_messages = await asummarize_oldest(current_messages, summarize_llm)
    
    # Update state if messages were pruned
    if len(clean_messages) < len(current_messages):
        state = {**state, "messages": clean_messages}
    
    # 2. Proceed with Supervisor Logic (Async)
    return await supervisor.aprocess(state)


async def iris_node(state: AgentState, iris) -> AgentState:
    """IRIS node function (Async Wrapper)."""
    logger.debug("Executing IRIS node (Async Wrapper)")
    # Wrap Sync Iris in Thread to prevent blocking
    return await asyncio.to_thread(iris.process, state)



# =============================================================================
# General Assistant Node (For Chat/Non-Automation Queries)
# =============================================================================

async def general_assistant(state: AgentState) -> AgentState:
    """General assistant node (Async)."""
    logger.info("💬 Routing to General Assistant (Chat)")
    
    try:
        # Get LLM for chat response
        from src.models.manager import ModelManager
        manager = ModelManager()
        llm = manager.get_smart_model()
        
        if not llm:
            if _HAS_LANGCHAIN_MESSAGES:
                error_msg = AIMessage(content="I'm sorry, I couldn't connect to the AI service.")
                return {**state, "messages": state.get("messages", []) + [error_msg]}
            return state
        
        # Get messages
        messages = state.get("messages", [])
        
        # === IDENTITY INJECTION (UNIFIED PERSONA) ===
        # Use PersonaProfile for consistent identity across all nodes
        from src.persona.profile import get_system_prompt
        system_prompt_text = get_system_prompt()
        
        # Handle message list updates similar to sync...
        if _HAS_LANGCHAIN_MESSAGES:
            # Check if first message is SystemMessage
            has_system = messages and isinstance(messages[0], SystemMessage)
            if not has_system:
                messages = [SystemMessage(content=system_prompt_text)] + messages
        else:
            # Dict fallback
            has_system = messages and isinstance(messages[0], dict) and messages[0].get("role") == "system"
            if not has_system:
                messages = [{"role": "system", "content": system_prompt_text}] + messages
        
        # Invoke LLM (Async)
        response = await llm.ainvoke(messages)
        
        # Extract content
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        logger.debug(f"Chat response: {response_content[:100]}...")
        
        # Build new message
        if _HAS_LANGCHAIN_MESSAGES:
            response_msg = AIMessage(content=response_content)
            new_messages = messages + [response_msg]
        else:
            new_messages = messages + [{"role": "assistant", "content": response_content}]
        
        return {
            **state,
            "messages": new_messages,
            "final_response": response_content,
        }
        
    except Exception as e:
        logger.error(f"General assistant error: {e}")
        error_msg = f"I encountered an error: {e}"
        if _HAS_LANGCHAIN_MESSAGES:
            return {**state, "messages": state.get("messages", []) + [AIMessage(content=error_msg)]}
        return state


# =============================================================================
# TARA 2.0 Integration (New Graph-Based Node)
# =============================================================================

# Try to import TARA 2.0 compiled graph
try:
    from src.agents.tara.graph import tara_app, create_initial_tara_state
    _HAS_TARA_2 = True
    logger.debug("✅ TARA 2.0 graph imported successfully")
except ImportError as e:
    _HAS_TARA_2 = False
    tara_app = None  # type: ignore
    create_initial_tara_state = None  # type: ignore
    logger.warning(f"⚠️ TARA 2.0 graph not available: {e}")


async def call_tara_2(state: AgentState) -> AgentState:
    """TARA 2.0 Node (Async)."""
    if not _HAS_TARA_2 or not tara_app:
        logger.error("TARA 2.0 not available, falling back to error message")
        if _HAS_LANGCHAIN_MESSAGES:
            error_msg = AIMessage(content="I'm sorry, the automation system is not available right now.")
            return {**state, "messages": state.get("messages", []) + [error_msg]}
        return state
    
    logger.info("🚀 Routing to TARA 2.0 SubGraph...")
    
    try:
        # === STEP 1: RETRIEVE (Source of Truth Strategy) ===
        raw_content = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                raw_content = msg.content
                break
            # Handle dict-based messages (if serialization occurred)
            elif isinstance(msg, dict) and msg.get("type") == "human":
                raw_content = msg.get("content", "")
                break
        
        # Fallback: Use state input if history is empty
        if not raw_content:
            raw_content = state.get("user_input", "")
        
        # === STEP 2: SANITIZE (The Split Fix) ===
        # Remove "[MEMORY CONTEXT]... User Input:" injected by engine.py
        clean_goal = raw_content
        
        if "User Input:" in raw_content:
            # Split on "User Input:" and take the last part (the actual command)
            clean_goal = raw_content.split("User Input:")[-1].strip()
            logger.info(f"🧼 Sanitized via Split. Result: '{clean_goal}'")
        elif "[MEMORY CONTEXT]" in raw_content:
            # Fallback: Remove memory context header if present
            clean_goal = raw_content.replace("[MEMORY CONTEXT]", "").strip()
            logger.info(f"🧼 Sanitized via Replace. Result: '{clean_goal}'")
        else:
            clean_goal = raw_content.strip()
            logger.info(f"✅ Input was clean: '{clean_goal}'")
        
        # === STEP 3: HANDOFF (To TARA) ===
        tara_input = {
            "messages": state.get("messages", []),
            "user_goal": clean_goal,  # ← CLEAN GOAL
            "screen_context": None,
            "active_app": None,
            "clipboard": None,
            "last_error": None,
            "tool_calls_pending": False,
            "iteration_count": 0,
            "final_response": None,
            "metadata": state.get("metadata", {}),
        }
        
        # === INVOKE TARA 2.0 (ASYNC) ===
        result = await tara_app.ainvoke(tara_input)
        
        # === EXTRACT RESPONSE (TARA → NIA) ===
        final_response = result.get("final_response", "")
        
        # If no final_response, try to get from last AI message
        if not final_response:
            result_messages = result.get("messages", [])
            for msg in reversed(result_messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    final_response = msg.content
                    break
                elif hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                    final_response = msg.content
                    break
        
        if not final_response:
            final_response = "Task completed."
        
        logger.info(f"✅ TARA 2.0 completed. Response: {final_response[:100]}...")
        
        # Build response message
        if _HAS_LANGCHAIN_MESSAGES:
            response_msg = AIMessage(content=final_response)
            new_messages = state.get("messages", []) + [response_msg]
        else:
            new_messages = state.get("messages", []) + [{"role": "assistant", "content": final_response}]
        
        return {
            **state,
            "messages": new_messages,
            "final_response": final_response,
            "next": AGENT_END,
        }
        
    except Exception as e:
        logger.error(f"❌ TARA 2.0 execution failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = f"I encountered an error while automating: {str(e)}"
        if _HAS_LANGCHAIN_MESSAGES:
            error_msg = AIMessage(content=error_response)
            return {**state, "messages": state.get("messages", []) + [error_msg], "next": AGENT_END}
        return {**state, "next": AGENT_END}


# =============================================================================
# Routing Functions
# =============================================================================

def route_from_tara(state: AgentState) -> str:
    """Route from TARA based on the 'next' field.
    
    This enables Active Listening loop: TARA can return to supervisor
    to complete the user's original request after saving preferences.
    
    Args:
        state: Current agent state.
        
    Returns:
        Next node name.
    """
    next_agent = state.get("next", AGENT_END)
    
    # If TARA specifically requested to go back to supervisor (Active Listening)
    if next_agent == "supervisor" or next_agent == AGENT_SUPERVISOR:
        logger.debug("🔄 TARA -> Supervisor: Active Listening loop")
        return AGENT_SUPERVISOR
    
    # Default: End the turn
    return AGENT_END


def route_from_supervisor(state: AgentState) -> str:
    """
    Decides execution path based on the User's Intent (ignoring system logs).
    
    Uses "Double Defense" strategy:
    1. Check explicit 'user_input' field first (most accurate)
    2. Fallback: scan backwards for last HumanMessage (robust)
    """
    # STRATEGY 0: Respect the Supervisor's Explicit Instruction (The Brain's Decision)
    next_agent = state.get("next")
    if next_agent and next_agent in [AGENT_TARA, AGENT_IRIS]:
        logger.info(f"🧠 Supervisor explicitly routed to: {next_agent}")
        return next_agent

    # STRATEGY 1: Check the explicit 'user_input' field first (Most Accurate)
    user_text = state.get("user_input", "")

    # STRATEGY 2: If empty, scan history BACKWARDS for the last HumanMessage
    if not user_text:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            # Check for HumanMessage type (LangChain object or dict)
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                user_text = msg.content
                break
            elif isinstance(msg, dict) and msg.get("type") == "human":
                user_text = msg.get("content", "")
                break
    
    # Safety: If still empty, default to General
    if not user_text:
        logger.warning("⚠️ Router found no Human input. Defaulting to General.")
        return "general"

    # Normalize
    user_text_lower = user_text.lower().strip()
    logger.info(f"🔍 ROUTER LOCKED ON: '{user_text_lower}'")

    # --- KEYWORD MATCHING (loaded from config) ---
    tara_keywords, iris_keywords = get_routing_keywords()

    if any(kw in user_text_lower for kw in tara_keywords):
        logger.info(f"🛠️ Routing to TARA (Trigger: '{user_text_lower}')")
        return AGENT_TARA

    if any(kw in user_text_lower for kw in iris_keywords):
        logger.info(f"👁️ Routing to IRIS (Trigger: '{user_text_lower}')")
        return AGENT_IRIS

    logger.info("💬 No keywords found. Routing to General Assistant.")
    return "general"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "summarize_oldest",
    "supervisor_node",
    "iris_node",
    "general_assistant",  # Chat/non-automation queries
    "call_tara_2",        # TARA 2.0 automation
    "_HAS_TARA_2",        # Feature flag
    "route_from_tara",
    "route_from_supervisor",
]


