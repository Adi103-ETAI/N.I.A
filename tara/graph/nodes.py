"""
TARA 2.0 Graph Nodes.

Implements the reasoning and tool execution nodes for the TARA SubGraph.
These nodes form the core thinking loop: Reason → Execute → Update Context.

Architecture:
    ┌─────────────┐
    │  reasoner   │ ← Generates tool calls using dynamic context
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │  tool_executor  │ ← Executes tools and updates state
    └────────┬────────┘
             │
             ▼
    ┌──────────────────┐
    │ context_updater  │ ← Extracts results into state keys
    └──────────────────┘
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any, Dict, List, Literal, Sequence, TypedDict

from core.logger import setup_logger
from core.config import get_settings

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
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError as e:
    # FAIL FAST: Do not masquerade with dummy classes
    raise RuntimeError(
        f"""\n
╔══════════════════════════════════════════════════════════════════╗
║  TARA STARTUP FAILED - MISSING CORE DEPENDENCY                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Required package not installed: {str(e).split("'")[1] if "'" in str(e) else 'langchain'}           
║                                                                  ║
║  Fix: pip install langchain-core langchain-nvidia-ai-endpoints   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    ) from e

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
from tara.tools.interface import get_tara_tools

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
    
    logger.info("[ROBUST PARSER] Detected <|python_tag|> in response, parsing...")
    
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
# LLM Initialization
# =============================================================================

def _get_llm():
    """Get configured LLM instance with NVIDIA settings."""
    return ChatNVIDIA(
        model=settings.LLM_MODEL,
        nvidia_api_key=settings.NVIDIA_API_KEY.get_secret_value(),
        base_url=settings.NVIDIA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
    )


# =============================================================================
# Node 1: Reasoner
# =============================================================================

def reasoner(state: TaraState) -> TaraStateUpdate:
    """
    Main reasoning node - generates tool calls using dynamic context.
    
    This node:
        1. Builds dynamic context from current state
        2. Injects context into system prompt
        3. Binds TARA tools to LLM
        4. Invokes LLM to decide next action
    
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
        
        # Invoke LLM
        response = llm_with_tools.invoke(full_messages)
        
        # Check if tool calls were made
        has_tool_calls = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
        
        # =====================================================================
        # FALLBACK: Parse Llama 3.1 <|python_tag|> format if bind_tools failed
        # =====================================================================
        if not has_tool_calls and response.content and "<|python_tag|>" in response.content:
            logger.warning("[REASONER] ChatNVIDIA didn't parse tool calls, using fallback parser")
            
            parsed_calls = _parse_llama_tool_calls(response.content)
            
            if parsed_calls:
                # Inject parsed tool calls into the AIMessage
                response.tool_calls = parsed_calls
                has_tool_calls = True
                logger.info(f"[REASONER] Fallback parser found {len(parsed_calls)} tool call(s)")
        
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

async def _async_tool_executor(state: TaraState) -> TaraStateUpdate:
    """
    Internal async worker for tool execution.
    
    Uses LangChain's polymorphic `ainvoke()` which automatically handles:
        - Async tools: Awaits the coroutine directly
        - Sync tools: Wraps in thread executor for non-blocking I/O
    
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
    
    logger.info(f"[TOOL_EXECUTOR] 🚀 Executing {len(tool_calls)} tool(s) via unified async")
    
    # Execute each tool call with UNIFIED AWAIT
    tool_messages: List[ToolMessage] = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", f"call_{tool_name}")
        
        logger.info(f"[TOOL_EXECUTOR] → {tool_name}({tool_args})")
        
        # Lookup tool
        tool = tools_map.get(tool_name)
        if tool is None:
            error_msg = f"❌ Tool not found: {tool_name}"
            logger.error(error_msg)
            tool_messages.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name,
            ))
            continue
        
        # Execute with try/except for robustness
        try:
            # ═══════════════════════════════════════════════════════════════
            # 
            # LangChain's ainvoke() is polymorphic:
            # - For async tools: awaits the coroutine
            # - For sync tools: runs in thread pool (non-blocking)
            # 
            # This single line handles ALL tool types correctly.
            # ═══════════════════════════════════════════════════════════════
            result = await tool.ainvoke(tool_args)
            
            # Ensure result is string
            result_str = str(result) if result is not None else f"✅ {tool_name} completed"
            
            logger.info(f"[TOOL_EXECUTOR] ✅ {tool_name}: {result_str[:80]}...")
            
            tool_messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_id,
                name=tool_name,
            ))
            
        except Exception as e:
            # Per-tool error handling (don't crash the loop)
            error_msg = f"❌ {tool_name} failed: {e}"
            logger.error(f"[TOOL_EXECUTOR] {error_msg}")
            tool_messages.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name,
            ))
    
    logger.info(f"[TOOL_EXECUTOR] ✓ Completed {len(tool_messages)} tool execution(s)")
    
    # Update context based on tool results
    context_updates = _extract_context_from_results(tool_messages)
    
    return {
        "messages": tool_messages,
        "tool_calls_pending": False,
        **context_updates,
    }


def tool_executor(state: TaraState) -> TaraStateUpdate:
    """
    Sync-to-async bridge for LangGraph tool execution.
    
    LangGraph's parent NIA graph runs synchronously, but TARA's browser tools
    are async (Playwright). This wrapper safely bridges the two execution contexts.
    
    Threading Strategy:
        1. If NO event loop is running: Use `asyncio.run()` directly.
        2. If loop IS running (nested): Use ThreadPoolExecutor to spawn
           a fresh event loop in a separate thread, avoiding "nested loop" errors.
    
    This pattern is essential for integrating async Playwright operations
    with LangGraph's synchronous node execution model.
    
    Args:
        state: Current TaraState with tool calls pending.
        
    Returns:
        TaraStateUpdate with tool results and updated context.
    """
    import concurrent.futures
    
    try:
        # Check if we're already in an event loop (Safety Check)
        loop = asyncio.get_running_loop()
        # We ARE in a running loop - use thread pool to avoid nesting
        logger.debug("Running async tools in thread pool (loop already running)")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _async_tool_executor(state))
            return future.result()
    except RuntimeError:
        # No loop running - this is the expected path for N.I.A.
        pass
    
    # The Standard Bridge: Run async code synchronously
    return asyncio.run(_async_tool_executor(state))


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
