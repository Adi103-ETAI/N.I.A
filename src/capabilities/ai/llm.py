"""TARA 2.0 LLM Operations.

Provides tools for runtime LLM provider management.

v3.0 Feature: Multi-Provider LLM Switching.
"""
from __future__ import annotations

from src.core.logger import setup_logger
from src.models.manager import get_model_manager, VALID_PROVIDERS

logger = setup_logger("TARA.Tools.LLMOps")


def switch_llm_provider(provider: str) -> str:
    """Switch the active AI model provider at runtime.
    
    Allows changing between LLM providers (nvidia, openai, groq, ollama)
    without restarting the application. All agents (NIA, TARA, IRIS) will
    use the new provider for subsequent requests.
    
    Args:
        provider: The provider to switch to.
                  Valid options: 'nvidia', 'openai', 'groq', 'ollama'.
    
    Returns:
        Success message or error description.
    
    Examples:
        - "Switch to OpenAI" → switch_llm_provider("openai")
        - "Use Groq for faster responses" → switch_llm_provider("groq")
        - "Go back to NVIDIA" → switch_llm_provider("nvidia")
    """
    provider_clean = provider.lower().strip()
    
    logger.info(f"Attempting to switch LLM provider to: {provider_clean}")
    
    try:
        mm = get_model_manager()
        old_provider = mm.get_active_provider()
        
        mm.set_active_provider(provider_clean)
        
        logger.info(f"Successfully switched provider: {old_provider} → {provider_clean}")
        return f"✅ Successfully switched active LLM provider from '{old_provider}' to '{provider_clean}'."
        
    except ValueError as e:
        # Expected error: missing API key or invalid provider
        error_msg = str(e)
        logger.warning(f"Provider switch failed: {error_msg}")
        return f"❌ Cannot switch to '{provider_clean}': {error_msg}"
        
    except Exception as e:
        # Unexpected error
        error_msg = f"Unexpected error switching provider: {e}"
        logger.exception(error_msg)
        return f"❌ {error_msg}"


def get_current_provider() -> str:
    """Get the name of the currently active LLM provider.
    
    Returns the provider being used by all agents (NIA, TARA, IRIS).
    
    Returns:
        Current provider name (e.g., 'nvidia', 'openai').
    
    Examples:
        - "What AI provider are we using?" → get_current_provider()
        - "Which model is active?" → get_current_provider()
    """
    try:
        mm = get_model_manager()
        current = mm.get_active_provider()
        return f"🧠 Current active LLM provider: '{current}'"
    except Exception as e:
        return f"❌ Error getting current provider: {e}"


def list_available_providers() -> str:
    """List all available LLM providers that can be used.
    
    Returns the list of providers that are installed and can be
    switched to. Note: Some providers may require API keys to
    actually use.
    
    Returns:
        List of available provider names.
    
    Examples:
        - "What AI providers are available?" → list_available_providers()
        - "Which models can I switch to?" → list_available_providers()
    """
    try:
        mm = get_model_manager()
        available = mm.factory.get_available_providers()
        current = mm.get_active_provider()
        
        provider_list = ", ".join(sorted(available))
        return f"📋 Available providers: {provider_list}\n🧠 Currently using: '{current}'"
    except Exception as e:
        return f"❌ Error listing providers: {e}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "switch_llm_provider",
    "get_current_provider",
    "list_available_providers",
]
