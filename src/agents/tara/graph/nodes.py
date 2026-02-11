"""
TARA 2.0 Graph Nodes.

v2.5.2 "Velocity" - Key Features:
    - Dynamic Provider Access: Uses `_get_llm()` function to fetch LLM on each
      call, enabling hot-swap provider switching without restart.
    - SafeLLM Integration: All LLM calls are wrapped with circuit breaker for
      automatic retry and fallback on 429/503 errors.
    - Unified Async Bridge: Tool executor uses `ainvoke()` polymorphically,
      Native Async: Executes tools in parallel via asyncio.gather.

Data Flow:
    Supervisor -> TARA Reasoner -> SafeLLM -> ModelManager -> Active Provider
                      |                           ^
                      v                           |__ Auto-fallback on 429
                 Tool Executor -> 50+ Tools

Architecture:
    ┌─────────────┐
    │  reasoner   │ ← Generates tool calls using dynamic context + SafeLLM
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │  tool_executor  │ ← Parallel Native Async
    └────────┬────────┘
             │
             ▼
    ┌──────────────────┐
    │ context_updater  │ ← Extracts results into state keys
    └──────────────────┘
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Literal, Sequence, TypedDict

from src.core.logger import setup_logger
from src.core.config import get_settings

from .state import TaraState, TaraNextStep
from .prompts import TARA_SYSTEM_PROMPT, build_tara_context


# =============================================================================
# TypedDict for Node Return Types
# =============================================================================

class TaraStateUpdate(TypedDict, total=False):
    """TypedDict for partial state updates returned by TARA nodes.
    
    This provides type safety for node return values.
    All fields are optional (total=False) since nodes return partial updates.
    """
    messages: Sequence[Any]  # BaseMessage or subclass
    user_goal: str
    screen_context: str | None
    active_app: str | None
    clipboard: str | None
    last_error: str | None
    tool_calls_pending: bool
    iteration_count: int
    final_response: str | None
    metadata: Dict[str, Any]

# =============================================================================
# Strict Dependency Guards
# =============================================================================

# LangChain imports - REQUIRED
try:
    from langchain_core.messages import (
        BaseMessage,
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    )
except ImportError as e:
    # FAIL FAST: Do not masquerade with dummy classes
    raise RuntimeError(
        f"""\n
╔══════════════════════════════════════════════════════════════════╗
║  TARA STARTUP FAILED - MISSING CORE DEPENDENCY                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Required package not installed: {str(e).split("'")[1] if "'" in str(e) else 'langchain'}           
║                                                                  ║
║  Fix: pip install langchain-core                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
    ) from e

# v3.0: ModelManager for dynamic provider switching
from src.models.manager import get_smart_model

# LangGraph tool node - REQUIRED for tool execution
try:
    from langgraph.prebuilt import ToolNode
except ImportError as e:
    raise RuntimeError(
        f"""\n
╔══════════════════════════════════════════════════════════════════╗
║  TARA STARTUP FAILED - MISSING LANGGRAPH                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Required package not installed: langgraph                       ║
║                                                                  ║
║  Fix: pip install langgraph                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    ) from e

# TARA tools
from src.capabilities.interface import get_tara_tools
from src.agents.tara.security import get_warden, SecurityError

logger = setup_logger("TARA.Nodes")
settings = get_settings()



# =============================================================================
# Robust JSON Sanitization Layer
# =============================================================================

def _sanitize_json_string(raw_json: str) -> str:
    """
    Sanitize messy LLM JSON output before parsing.
    
    Handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Trailing commas in objects/arrays
    - JavaScript-style comments (// and /* */)
    - Leading/trailing whitespace
    - Single quotes instead of double quotes
    
    Args:
        raw_json: Potentially malformed JSON string.
        
    Returns:
        Sanitized JSON string (best effort).
    """
    text = raw_json.strip()
    
    # 1. Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # 2. Remove JavaScript-style single-line comments (// ...)
    text = re.sub(r'//[^\n]*', '', text)
    
    # 3. Remove JavaScript-style multi-line comments (/* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # 4. Fix trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # 5. Fix single quotes (best effort - won't work for all cases)
    # Only replace single quotes that look like string delimiters
    # This is risky but helps with common LLM outputs
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    
    return text.strip()


def _extract_json_objects(text: str) -> List[str]:
    """
    Extract all JSON objects from text using bracket matching.
    
    More robust than regex - handles nested objects, whitespace, and newlines.
    Applies sanitization before extraction for reliable parsing.
    
    Args:
        text: Raw text containing potential JSON objects.
        
    Returns:
        List of sanitized JSON string candidates.
    """
    json_objects = []
    depth = 0
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                raw_json = text[start_idx:i + 1]
                # Apply sanitization to each extracted object
                sanitized = _sanitize_json_string(raw_json)
                json_objects.append(sanitized)
                start_idx = None
    
    return json_objects


def _parse_llama_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Parse Llama 3.1's native tool call format when ChatNVIDIA doesn't.
    
    🌊 MODERNIZED: Uses bracket-matching JSON extraction instead of fragile regex.
    
    Llama 3.1 outputs tool calls in various formats:
        <|python_tag|><function>tool_name</function>{"arg": "value"}</function>
        <|python_tag|>tool_name.call({"arg": "value"})
        <|python_tag|>tool_name {"arg": "value"}
    
    Returns:
        List of tool call dicts with 'name', 'args', and 'id' keys.
        Returns empty list [] on any parsing failure (safe fallback).
    """
    tool_calls = []
    
    if not content or "<|python_tag|>" not in content:
        return tool_calls
    
    logger.debug("[ROBUST PARSER] Detected <|python_tag|> in response, parsing...")
    
    # Extract all JSON objects from the content
    json_candidates = _extract_json_objects(content)
    
    if not json_candidates:
        logger.warning("[ROBUST PARSER] No JSON objects found in response")
        return tool_calls
    
    # Try to find function name near each JSON object
    for i, json_str in enumerate(json_candidates):
        try:
            # Parse the JSON arguments
            args = json.loads(json_str)
            
            # Find function name: look backwards from JSON position
            json_pos = content.find(json_str)
            prefix = content[:json_pos].strip()
            
            func_name = None
            
            # Strategy 1: <function>name</function> pattern
            if "</function>" in prefix:
                # Extract from <function>name</function>
                func_match = re.search(r'<function>(\w+)</function>\s*$', prefix)
                if func_match:
                    func_name = func_match.group(1)
            
            # Strategy 2: name.call( or name( pattern
            if not func_name:
                call_match = re.search(r'(\w+)(?:\.call)?\s*\(\s*$', prefix)
                if call_match:
                    func_name = call_match.group(1)
            
            # Strategy 3: Just a word before the JSON
            if not func_name:
                word_match = re.search(r'(\w+)\s*$', prefix)
                if word_match:
                    func_name = word_match.group(1)
            
            # Strategy 4: Check if args contains 'name' field (structured output)
            if not func_name and isinstance(args, dict) and 'name' in args:
                func_name = args.pop('name')  # Extract and remove from args
            
            if func_name:
                tool_calls.append({
                    "name": func_name,
                    "args": args if isinstance(args, dict) else {"value": args},
                    "id": f"call_{func_name}_{i}",
                })
                logger.info(f"[ROBUST PARSER] Extracted: {func_name}({args})")
            else:
                logger.warning(f"[ROBUST PARSER] Found JSON but no function name: {json_str[:50]}...")
                
        except json.JSONDecodeError as e:
            logger.debug(f"[ROBUST PARSER] Invalid JSON: {e}")
            continue
        except Exception as e:
            logger.debug(f"[ROBUST PARSER] Parse error: {e}")
            continue
    
    return tool_calls


# =============================================================================
# LLM Initialization (v3.0: via ModelManager)
# =============================================================================

def _get_llm():
    """Get configured LLM instance via ModelManager.
    
    v3.0: Uses ModelManager for dynamic provider switching.
    The temperature is set by the ModelManager based on settings.
    """
    return get_smart_model(temperature=settings.LLM_TEMPERATURE)


# =============================================================================
# Node 1: Reasoner
# =============================================================================

async def reasoner(state: TaraState) -> TaraStateUpdate:
    """
    Main reasoning node - generates tool calls using dynamic context.
    
    This node:
        1. Builds dynamic context from current state
        2. Injects context into system prompt
        3. Binds TARA tools to LLM
        4. Invokes LLM to decide next action (ASYNC)
    
    Args:
        state: Current TaraState with conversation history and context.
        
    Returns:
        TaraStateUpdate with new messages and iteration count.
    """
    # Check iteration limit
    iteration = state.get("iteration_count", 0)
    if iteration >= settings.MAX_ITERATIONS:
        logger.warning(f"Max iterations ({settings.MAX_ITERATIONS}) reached")
        return {
            "messages": [],
            "final_response": "Maximum iterations reached. Task may be incomplete.",
            "tool_calls_pending": False,
        }
    
    logger.info(f"Reasoner iteration {iteration + 1}")
    
    try:
        # Build dynamic context
        context_str = build_tara_context(state)
        
        # Create system message with context
        system_content = f"{TARA_SYSTEM_PROMPT}\n\n{context_str}"
        system_msg = SystemMessage(content=system_content)
        
        # Get conversation messages
        messages: List[BaseMessage] = list(state.get("messages", []))
        
        # Build full message list for LLM
        full_messages = [system_msg] + messages
        
        # Get tools and bind to LLM
        tools = get_tara_tools()
        llm = _get_llm()
        llm_with_tools = llm.bind_tools(tools)
        
        # 🚀 ASYNC INVOKE: Non-blocking LLM call
        response = await llm_with_tools.ainvoke(full_messages)
        
        # Check if tool calls were made
        has_tool_calls = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
        
        # =====================================================================
        # FALLBACK: Parse Llama 3.1 <|python_tag|> format if bind_tools failed
        # (Only applicable when using NVIDIA Llama models)
        # =====================================================================
        if not has_tool_calls and response.content and "<|python_tag|>" in response.content:
            logger.warning("[REASONER] LLM didn't parse tool calls, using fallback parser")
            
            parsed_calls = _parse_llama_tool_calls(response.content)
            
            if parsed_calls:
                # Inject parsed tool calls into the AIMessage
                response.tool_calls = parsed_calls
                has_tool_calls = True
                logger.debug(f"[REASONER] Fallback parser found {len(parsed_calls)} tool call(s)")
        
        logger.debug(f"LLM response: {response.content[:100] if response.content else 'Tool call'}...")
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


# =============================================================================
# Node 2: Tool Executor (Async Bridge Pattern)
# =============================================================================

async def tool_executor(state: TaraState) -> TaraStateUpdate:
    """
    Async Tool Executor Node (Parallel Native).
    
    Executes multiple tool calls in parallel using asyncio.gather.
    Leverages LangChain's ainvoke() which handles Sync/Async dispatch internally.
    
    Args:
        state: Current TaraState with tool calls pending.
        
    Returns:
        TaraStateUpdate with tool results and updated context.
    """
    # Get tools and build lookup map
    tools = get_tara_tools()
    tools_map: Dict[str, Any] = {tool.name: tool for tool in tools}
    
    # Extract tool calls from last message
    messages = state.get("messages", [])
    if not messages:
        logger.warning("[TOOL_EXECUTOR] No messages in state")
        return {"messages": [], "tool_calls_pending": False}
    
    last_message = messages[-1]
    
    # Check if last message has tool calls
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        logger.warning("[TOOL_EXECUTOR] No tool_calls in last message")
        return {"messages": [], "tool_calls_pending": False}
    
    logger.info(f"[TOOL_EXECUTOR] 🚀 Executing {len(tool_calls)} tool(s) in parallel")
    
    # Prepare tasks for parallel execution
    tasks = []
    
    async def _exec_single(t_call):
        t_name = t_call.get("name", "")
        t_args = t_call.get("args", {})
        t_id = t_call.get("id", f"call_{t_name}")
        
        logger.info(f"[TOOL_EXECUTOR] → {t_name}({t_args})")
        
        tool = tools_map.get(t_name)
        if tool is None:
            error_msg = f"❌ Tool not found: {t_name}"
            logger.error(error_msg)
            return ToolMessage(content=error_msg, tool_call_id=t_id, name=t_name)
        
        try:
            # 🛡️ SECURITY GATEKEEPER (Operation Iron Cage)
            # Check tool metadata for security level
            sec_level = tool.metadata.get("security_level", "host_standard")
            
            if sec_level == "high_risk":
                logger.warning(f"🛡️ Warden Intercept: Validating '{t_name}' permission...")
                
                # BLOCKING CHECK: Query Warden for permission
                warden = get_warden()
                warden.check_permission(t_name, t_args)
                
                # If check_permission doesn't raise, we are APPROVED.
                logger.info(f"🛡️ Warden Approved: Proceeding with '{t_name}'")
            
            # Polymorphic Dispatch (Async or ThreadPool for Sync)
            # LangChain's ainvoke handles this automatically
            result = await tool.ainvoke(t_args)
            
            result_str = str(result) if result is not None else f"✅ {t_name} completed"
            logger.info(f"[TOOL_EXECUTOR] ✅ {t_name}: {result_str[:80]}...")
            
            return ToolMessage(content=result_str, tool_call_id=t_id, name=t_name)

        except SecurityError as se:
            # Explicitly catch security denials
            error_msg = f"🚫 SECURITY DENIED: {se}"
            logger.warning(f"[TOOL_EXECUTOR] {error_msg}")
            return ToolMessage(content=error_msg, tool_call_id=t_id, name=t_name)
            
        except Exception as e:
            error_msg = f"❌ {t_name} failed: {e}"
            logger.error(f"[TOOL_EXECUTOR] {error_msg}")
            return ToolMessage(content=error_msg, tool_call_id=t_id, name=t_name)

    # Launch all tasks
    for tool_call in tool_calls:
        tasks.append(_exec_single(tool_call))
    
    # Wait for all to complete
    tool_messages = await asyncio.gather(*tasks)
    
    logger.info(f"[TOOL_EXECUTOR] ✓ Completed {len(tool_messages)} tool execution(s)")
    
    # 🧠 LAYER 3: Learn successful tool chains (Procedural Memory)
    # Check if tools actually ran and succeeded (no "❌" in output)
    if tool_messages and all("❌" not in getattr(m, 'content', '') for m in tool_messages):
        try:
            # RIPPLE FIX: Local import to avoid circular dependency issues
            from src.core.registry import ServiceRegistry
            from langchain_core.messages import HumanMessage  # RIPPLE: For type check

            memory_svc = ServiceRegistry.get("memory")
            
            # 🎯 GOAL FALLBACK: If user_goal is None, extract from last human message
            user_goal = state.get("user_goal")
            if not user_goal:
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_goal = getattr(msg, 'content', '')[:100]  # Truncate to 100 chars
                        break
            
            # Ensure memory service is active AND we have a derived goal
            if memory_svc and user_goal:
                # Extract tool names from the original tool_calls list
                tool_names = [tc.get("name") for tc in tool_calls if tc.get("name")]
                if tool_names:
                    memory_svc.add_skill_path(user_goal, tool_names)
                    logger.info(f"🧠 [SKILL] Learned sequence for '{user_goal[:50]}': {tool_names}")
        except Exception as e:
            logger.warning(f"🧠 [SKILL] Failed to save skill path: {e}")
    
    # Update context based on tool results
    context_updates = _extract_context_from_results(tool_messages)
    
    return {
        "messages": tool_messages,
        "tool_calls_pending": False,
        **context_updates,
    }


def _extract_context_from_results(tool_messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """
    Extract context updates from tool results.
    
    Inspects tool outputs and updates relevant state keys
    for the next reasoning cycle.
    """
    updates: Dict[str, Any] = {}
    
    for msg in tool_messages:
        if not isinstance(msg, ToolMessage):
            continue
        
        content = str(msg.content)
        tool_name = getattr(msg, "name", "")
        
        # Update screen_context from UI tree dumps
        if tool_name == "dump_ui_tree" or "UI Elements" in content:
            updates["screen_context"] = content
        
        # Update active_app from launch results
        if tool_name == "launch_app" and "launched" in content.lower():
            # Extract app name from result
            if ":" in content:
                parts = content.split(":")
                if len(parts) > 1:
                    updates["active_app"] = parts[1].strip().split()[0]
        
        # Update clipboard from get_clipboard
        if tool_name == "get_clipboard" or "Clipboard" in content:
            if ":" in content:
                updates["clipboard"] = content.split(":", 1)[1].strip()[:200]
        
        # Track errors
        if content.startswith("❌") or "Error" in content or "failed" in content.lower():
            updates["last_error"] = content
    
    return updates


# =============================================================================
# Node 3: Response Formatter
# =============================================================================

def response_formatter(state: TaraState) -> Dict[str, Any]:
    """
    Format final response for return to NIA.
    
    Extracts the last meaningful response from the message history.
    """
    messages = state.get("messages", [])
    
    # Find last AI message without tool calls
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                if msg.content:
                    return {"final_response": msg.content}
    
    # Fallback
    return {"final_response": "Task completed."}


# =============================================================================
# Routing Logic
# =============================================================================

def should_continue(state: TaraState) -> TaraNextStep:
    """
    Determine next step in the TARA graph.
    
    Args:
        state: Current TaraState.
    
    Returns:
        TaraNextStep literal - one of tool_executor, reasoner, or __end__.
    """
    # Check for final response
    if state.get("final_response"):
        return "__end__"
    
    # Check for pending tool calls
    if state.get("tool_calls_pending"):
        return "tool_executor"
    
    # Check iteration limit
    if state.get("iteration_count", 0) >= settings.MAX_ITERATIONS:
        return "__end__"
    
    # Check last message for tool calls
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tool_executor"
            elif last_msg.content and not state.get("tool_calls_pending"):
                # AI responded without tools - might be done
                return "__end__"
    
    # Default: continue reasoning
    return "reasoner"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "reasoner",
    "tool_executor",
    "response_formatter",
    "should_continue",
]
