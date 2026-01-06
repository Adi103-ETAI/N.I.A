"""TARA Preferences Unit - User Preference Management Tools.

CRITICAL: These tools MUST be executed when the user mentions saving/remembering.
The LLM has NO internal memory. Tool execution is MANDATORY.
"""
from __future__ import annotations

import json

from tara.protocols import tara_tool
from core.memory import get_memory_manager


# =============================================================================
# Preference Tools (MANDATORY EXECUTION - NO CONVERSATIONAL RESPONSES)
# =============================================================================

@tara_tool(
    name="save_user_preference",
    category="memory",
    description=(
        "CRITICAL: YOU MUST USE THIS TOOL when the user implies saving a fact, "
        "preference, or setting. Do NOT just reply with text like 'I will remember that'. "
        "Actually CALL this function to persist the data. "
        "Trigger words: remember, save, note, set, store, prefer, like, hate, love, "
        "don't forget, I am a, my name is."
    )
)
def save_user_preference(key: str, value: str) -> str:
    """
    CRITICAL: YOU MUST USE THIS TOOL when the user implies saving a fact, preference, or setting.
    Do NOT just reply with text like "I will remember that".
    Actually CALL this function to persist the data.
    
    TRIGGER PHRASES (REQUIRE THIS TOOL):
    - "remember that I..."
    - "save this..."
    - "note that..."
    - "don't forget..."
    - "I prefer..."
    - "I like/hate/love..."
    - "I am a..."
    - "my name is..."
    - "set my..."
    
    Args:
        key: Short category (e.g., 'nickname', 'theme', 'language', 'expertise').
        value: The preference details (e.g., 'Python expert, hates Java').
    
    Returns:
        Confirmation that preference was persisted to database.
    """
    try:
        memory = get_memory_manager()
        success = memory.set_preference(key, value)
        
        if success:
            return f"✅ SAVED TO DATABASE: {key} = {value}"
        else:
            return f"⚠️ FAILED TO SAVE: {key}"
            
    except Exception as e:
        return f"❌ DATABASE ERROR: {e}"


@tara_tool(
    name="list_user_preferences",
    category="memory",
    description=(
        "MUST BE CALLED when user asks 'what do you know about me', 'my preferences', "
        "'what have I told you'. You have NO internal memory - this is the ONLY way."
    )
)
def list_user_preferences() -> str:
    """
    CRITICAL: You have NO memory. You MUST call this tool to know anything about the user.
    
    Returns:
        JSON-formatted string of all key-value preferences.
    """
    try:
        memory = get_memory_manager()
        prefs = memory.get_all_preferences()
        
        if not prefs:
            return "📋 DATABASE EMPTY: No preferences saved yet."
        
        formatted = json.dumps(prefs, indent=2, ensure_ascii=False)
        return f"📋 USER PREFERENCES FROM DATABASE:\n{formatted}"
        
    except Exception as e:
        return f"❌ DATABASE ERROR: {e}"


@tara_tool(
    name="get_user_preference",
    category="memory",
    description="Get a specific user preference by key from database."
)
def get_user_preference(key: str) -> str:
    """Get a specific user preference from database.
    
    Args:
        key: The preference key to look up.
    
    Returns:
        The preference value, or a message if not found.
    """
    try:
        memory = get_memory_manager()
        value = memory.get_preference(key)
        
        if value is not None:
            return f"📋 DATABASE RESULT: {key} = {value}"
        else:
            return f"📋 NOT FOUND IN DATABASE: '{key}'"
            
    except Exception as e:
        return f"❌ DATABASE ERROR: {e}"
