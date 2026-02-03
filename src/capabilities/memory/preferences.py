"""
TARA Memory Tools - Preference Management.

This module provides tools for saving user preferences and facts to persistent
memory. Preferences are stored in SQLite and used to personalize responses.

Usage:
    remember_preference("user_name", "Aditya")
    remember_preference("coding_style", "prefers concise code")
"""
from src.core.registry import ServiceRegistry
from src.core.logger import setup_logger

logger = setup_logger("TARA.Memory")


def remember_preference(key: str, value: str) -> str:
    """
    Save a user preference or fact permanently.
    
    Use this to remember user facts, preferences, or settings that should
    persist across sessions. Examples:
    - "user_name" -> "Aditya"
    - "preferred_language" -> "Python"
    - "response_style" -> "concise and technical"
    - "favorite_browser" -> "Chrome"
    
    Args:
        key: The preference name (e.g., "user_name", "coding_style").
        value: The preference value to store.
        
    Returns:
        Confirmation message indicating success or failure.
    """
    try:
        memory = ServiceRegistry.get("memory")
        
        if memory is None:
            logger.error("Memory service not available")
            return "❌ Memory service not available. Preference not saved."
        
        success = memory.set_preference(key, value, category="user")
        
        if success:
            logger.info(f"Saved preference: {key} = {value}")
            return f"✅ Remembered: {key} = {value}"
        else:
            logger.error(f"Failed to save preference: {key}")
            return f"❌ Failed to save preference: {key}"
            
    except Exception as e:
        logger.error(f"remember_preference error: {e}")
        return f"❌ Error saving preference: {e}"


def recall_preference(key: str) -> str:
    """
    Recall a previously saved user preference.
    
    Use this to retrieve user facts or preferences that were previously saved.
    
    Args:
        key: The preference name to look up.
        
    Returns:
        The preference value, or a message if not found.
    """
    try:
        memory = ServiceRegistry.get("memory")
        
        if memory is None:
            return "❌ Memory service not available."
        
        value = memory.get_preference(key)
        
        if value:
            return f"{key} = {value}"
        else:
            return f"No preference found for '{key}'"
            
    except Exception as e:
        logger.error(f"recall_preference error: {e}")
        return f"❌ Error recalling preference: {e}"


def list_preferences() -> str:
    """
    List all saved user preferences.
    
    Returns a summary of all stored user preferences and facts.
    
    Returns:
        Formatted list of all preferences, or message if empty.
    """
    try:
        memory = ServiceRegistry.get("memory")
        
        if memory is None:
            return "❌ Memory service not available."
        
        prefs = memory.get_all_preferences()
        
        if not prefs:
            return "No preferences saved yet."
        
        lines = [f"📋 Saved Preferences ({len(prefs)} total):"]
        for key, value in prefs.items():
            lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)
            
    except Exception as e:
        logger.error(f"list_preferences error: {e}")
        return f"❌ Error listing preferences: {e}"
