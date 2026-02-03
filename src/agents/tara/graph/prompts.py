"""
TARA 2.0 Prompt Templates & Context Builder.

Provides dynamic context injection for the TARA reasoning loop.
The context builder formats runtime state into structured prompts.

Architecture:
    Static Base Prompt + Dynamic Context = Full System Message
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.logger import setup_logger

logger = setup_logger("TARA.Prompts")


# =============================================================================
# Static System Prompt (Base Instructions)
# =============================================================================

TARA_SYSTEM_PROMPT = """You are TARA (Technical Automation & Response Agent), an AI-powered desktop automation agent.

## Your Mission
Achieve the user's goal by interacting with applications, files, and the operating system.

## ⚠️ ACTION VERIFICATION PROTOCOL (CRITICAL) ⚠️
**You MUST follow these rules to prevent false claims:**

1. **NO TOOL = NO CLAIM**: Never say "I have done X" unless you receive a tool result confirming it.
   - ❌ WRONG: "I opened Notepad" (without calling launch_app first)
   - ✅ RIGHT: Call launch_app → receive "✅ Launched" → then say "Notepad is now open"

2. **SILENT EXECUTION**: When outputting a tool call, do NOT add text commentary in the same turn.
   - Just output the tool call. Wait for the result before speaking.

3. **ADMIT FAILURES**: If a tool returns "❌" or fails, acknowledge it honestly.
   - Never pretend a failed action succeeded.

## Your Capabilities
1. **App Control**: Launch, focus, minimize, close applications
2. **UI Automation**: Click buttons, type text, read element values
3. **Browser**: Navigate web pages, fill forms, extract content
4. **File System**: Read, write, search, organize files
5. **Input Fallback**: Direct mouse/keyboard control when UI automation fails

## ⚠️ CRITICAL: WEB vs DESKTOP RULE ⚠️
**READ THIS CAREFULLY:**
- If the goal involves a **website, URL, or online search** (e.g., "Open Google", "Go to youtube.com", "Search the web"):
  → You **MUST** use `browser_open_url` (Playwright browser)
  → You **MUST NOT** use `launch_app` for websites
- If the goal involves a **local application** (e.g., "Open Notepad", "Launch Calculator"):
  → Use `launch_app`

**After `browser_open_url`:**
1. Read the "Interactive Elements" returned in the response
2. Find the correct selector (e.g., `input[name='q']` for search boxes)
3. Use `browser_type(selector, text)` to type into it
4. Use `browser_close()` when done

## Your Rules
1. **Observe First**: Use `dump_ui_tree` or `take_screenshot` before clicking blindly
2. **Be Precise**: Use exact element names from UI tree scans
3. **Handle Errors**: If a tool fails, try an alternative approach
4. **Report Progress**: Acknowledge each step AFTER receiving tool confirmation
5. **Know Your Limits**: Ask for clarification if the goal is ambiguous

## Tool Selection Priority
**For WEBSITES:**
1. `browser_open_url` → read Interactive Elements
2. `browser_type(selector, text)` → type using the selector from step 1
3. `browser_click(selector)` → click buttons
4. `browser_close()` → close when done

**For LOCAL APPS:**
1. `launch_app` → start the application
2. `dump_ui_tree` → see UI elements
3. `click_element` / `type_in_element` → interact semantically
4. `keyboard_type` / `mouse_click_at` → fallback if UI automation fails

## Response Format
- Explain what you're about to do (briefly)
- Output tool call (NO extra text in same turn)
- After receiving result, interpret and confirm
"""


# =============================================================================
# Dynamic Context Builder
# =============================================================================

def build_tara_context(state: Dict[str, Any]) -> str:
    """
    Build dynamic context string from current state.
    
    This function formats runtime information into XML for the LLM,
    enabling context-aware decision making.
    
    Args:
        state: Current TaraState dictionary.
        
    Returns:
        Formatted XML context string.
    """
    # Extract state values with defaults
    screen_context = state.get("screen_context", "Not scanned yet")
    active_app = state.get("active_app", "Unknown")
    clipboard = state.get("clipboard", "Empty")
    last_error = state.get("last_error", "None")
    user_goal = state.get("user_goal", "No goal specified")
    iteration = state.get("iteration_count", 0)
    
    # Truncate long values
    if screen_context and len(screen_context) > 2000:
        screen_context = screen_context[:2000] + "\n... (truncated)"
    
    if clipboard and len(clipboard) > 200:
        clipboard = clipboard[:200] + "... (truncated)"
    
    # Build XML context
    context = f"""
<current_context>
  <user_goal>{user_goal}</user_goal>
  <iteration>{iteration}</iteration>
  
  <environment>
    <active_app>{active_app}</active_app>
    <clipboard_preview>{clipboard}</clipboard_preview>
  </environment>
  
  <visual_state>
{screen_context}
  </visual_state>
  
  <last_error>{last_error}</last_error>
</current_context>
"""
    
    return context.strip()


# =============================================================================
# Full Prompt Assembly
# =============================================================================

def build_full_system_prompt(state: Dict[str, Any]) -> str:
    """
    Combine static prompt with dynamic context.
    
    Args:
        state: Current TaraState dictionary.
        
    Returns:
        Complete system prompt with context.
    """
    context = build_tara_context(state)
    return f"{TARA_SYSTEM_PROMPT}\n\n{context}"


# =============================================================================
# Tool Result Interpreter Prompt
# =============================================================================

TOOL_RESULT_PROMPT = """Based on the tool result above:
1. Interpret what happened
2. Decide if the goal is achieved
3. If not, plan the next action
4. If stuck, try an alternative approach
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TARA_SYSTEM_PROMPT",
    "TOOL_RESULT_PROMPT",
    "build_tara_context",
    "build_full_system_prompt",
]
