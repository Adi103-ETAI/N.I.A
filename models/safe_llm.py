"""SafeLLM - Circuit Breaker Wrapper for LangChain LLMs.

Provides transparent error handling, retry logic, and automatic provider
fallback for LLM invocations. Acts as a protective shield around any
LangChain ChatModel.

v2.5.2: Centralized Circuit Breaker Pattern with Multi-Provider Fallback.

Usage:
    # Typically applied automatically by ModelManager
    from models.safe_llm import SafeLLM
    
    safe_model = SafeLLM(raw_model, manager=model_manager)
    response = safe_model.invoke(messages)  # Auto-retry on 429
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from core.logger import setup_logger

if TYPE_CHECKING:
    from models.model_manager import ModelManager

logger = setup_logger("SafeLLM")

# v2.5.2: Fallback notice template (injected into response when switch occurs)
FALLBACK_NOTICE_TEMPLATE = (
    "\n\n*(⚡ SYSTEM NOTICE: Connection to {original_provider} failed "
    "(rate limit/quota exceeded). Automatically switched to {fallback_provider}. "
    "Provider change is now active.)*"
)


# =============================================================================
# Error Detection Helpers
# =============================================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a rate limit (429) error.
    
    Handles various error formats from different providers:
    - OpenAI: openai.RateLimitError
    - NVIDIA/httpx: httpx.HTTPStatusError with status 429
    - Generic: Exception message contains "429" or "rate limit"
    """
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__
    
    # Check exception type names
    if exc_type in ("RateLimitError", "APIStatusError"):
        return True
    
    # Check for httpx status errors
    if exc_type == "HTTPStatusError":
        if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
            return exc.response.status_code == 429
    
    # Check message content
    if "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str:
        return True
    
    if "exceeded" in exc_str and ("request" in exc_str or "limit" in exc_str):
        return True
    
    return False


def _is_service_unavailable(exc: Exception) -> bool:
    """Check if exception is a service unavailable (503) error."""
    exc_str = str(exc).lower()
    
    if "503" in exc_str or "service unavailable" in exc_str:
        return True
    
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code in (503, 502, 504)
    
    return False


# =============================================================================
# SafeLLM Wrapper Class
# =============================================================================

class SafeLLM:
    """Circuit Breaker wrapper for LangChain LLMs.
    
    Transparently wraps any LangChain ChatModel to provide:
    - Automatic retry on rate limit (429) errors
    - Fallback to alternative provider on failure
    - Exponential backoff with jitter
    - Pass-through for all LangChain features (bind_tools, etc.)
    
    The wrapper is transparent - it looks and acts exactly like the
    underlying LangChain model to consuming code.
    
    Attributes:
        primary_model: The wrapped LangChain ChatModel.
        manager: Reference to ModelManager for fallback switching.
        fallback_provider: Provider to switch to on failure (default: "nvidia").
        max_retries: Maximum retry attempts before giving up.
        
    Example:
        # Direct usage (typically done by ModelManager)
        safe = SafeLLM(ChatNVIDIA(...), manager=mm)
        response = safe.invoke([HumanMessage(content="Hello")])
        
        # With tools (TARA pattern)
        safe_with_tools = safe.bind_tools(tools)
        response = safe_with_tools.invoke(messages)
    """
    
    def __init__(
        self,
        primary_model: Any,
        manager: Optional["ModelManager"] = None,
        fallback_provider: str = "nvidia",
        max_retries: int = 2,
    ) -> None:
        """Initialize SafeLLM wrapper.
        
        Args:
            primary_model: LangChain ChatModel to wrap.
            manager: ModelManager instance for provider switching.
            fallback_provider: Provider to fall back to (default: nvidia).
            max_retries: Max retry attempts on transient errors.
        """
        self._primary_model = primary_model
        self._manager = manager
        self._fallback_provider = fallback_provider
        self._max_retries = max_retries
        self._circuit_open = False  # Track if we've already failed over
        self._original_provider: Optional[str] = None  # Track provider before switch
    
    # =========================================================================
    # Core Invocation with Circuit Breaker
    # =========================================================================
    
    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """Invoke the LLM with circuit breaker protection.
        
        Attempts to call the primary model. On rate limit or service errors,
        switches to fallback provider and retries.
        
        Args:
            input: Messages or prompt to send to the LLM.
            **kwargs: Additional arguments passed to the model.
            
        Returns:
            LLM response (AIMessage or similar).
            
        Raises:
            Exception: If all retries exhausted and fallback fails.
        """
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                return self._primary_model.invoke(input, **kwargs)
                
            except Exception as exc:
                last_exception = exc
                
                # Check if this is a recoverable error
                is_rate_limit = _is_rate_limit_error(exc)
                is_unavailable = _is_service_unavailable(exc)
                
                if is_rate_limit or is_unavailable:
                    error_type = "Rate limit (429)" if is_rate_limit else "Service unavailable"
                    logger.warning(
                        f"🔄 [{error_type}] Primary provider failed (attempt {attempt + 1}/{self._max_retries + 1}): {exc}"
                    )
                    
                    # Try fallback if we have a manager and haven't already failed over
                    if self._manager and not self._circuit_open:
                        fallback_result = self._engage_circuit_breaker(input, **kwargs)
                        if fallback_result is not None:
                            return fallback_result
                    
                    # Exponential backoff before retry
                    if attempt < self._max_retries:
                        delay = min(2 ** attempt, 8)  # Cap at 8 seconds
                        logger.info(f"💤 Backing off {delay}s before retry...")
                        time.sleep(delay)
                else:
                    # Non-recoverable error - don't retry
                    logger.error(f"❌ LLM invocation failed (non-recoverable): {exc}")
                    raise
        
        # All retries exhausted
        logger.error(f"❌ All {self._max_retries + 1} attempts failed. Last error: {last_exception}")
        raise last_exception

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        """Async version of invoke with circuit breaker protection (Fixes 'The Imposter')."""
        import asyncio
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                return await self._primary_model.ainvoke(input, **kwargs)
                
            except Exception as exc:
                last_exception = exc
                
                # Check if this is a recoverable error
                is_rate_limit = _is_rate_limit_error(exc)
                is_unavailable = _is_service_unavailable(exc)
                
                if is_rate_limit or is_unavailable:
                    error_type = "Rate limit (429)" if is_rate_limit else "Service unavailable"
                    logger.warning(
                        f"🔄 [ASYNC] [{error_type}] Primary provider failed (attempt {attempt + 1}/{self._max_retries + 1}): {exc}"
                    )
                    
                    # Try fallback if we have a manager and haven't already failed over
                    if self._manager and not self._circuit_open:
                        fallback_result = await self._engage_circuit_breaker_async(input, **kwargs)
                        if fallback_result is not None:
                            return fallback_result
                    
                    # Exponential backoff before retry
                    if attempt < self._max_retries:
                        delay = min(2 ** attempt, 8)  # Cap at 8 seconds
                        logger.info(f"💤 [ASYNC] Backing off {delay}s before retry...")
                        await asyncio.sleep(delay)
                else:
                    # Non-recoverable error - don't retry
                    logger.error(f"❌ [ASYNC] LLM invocation failed (non-recoverable): {exc}")
                    raise
        
        # All retries exhausted
        logger.error(f"❌ [ASYNC] All {self._max_retries + 1} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def _engage_circuit_breaker(self, input: Any, **kwargs: Any) -> Optional[Any]:
        """Engage the circuit breaker - switch provider and retry.
        
        v2.5.2 FIX: Injects a system notice into the response content so the
        agent's memory stays synced with reality about which provider is active.
        
        Args:
            input: Original input to retry.
            **kwargs: Original kwargs.
            
        Returns:
            Response from fallback provider (with notice injected), or None if fallback failed.
        """
        # Track the original provider for the notice
        self._original_provider = self._manager.get_active_provider() if self._manager else "unknown"
        
        logger.warning(f"⚡ CIRCUIT BREAKER: Switching from '{self._original_provider}' to fallback '{self._fallback_provider}'")
        
        try:
            # Switch the global provider state
            self._manager.set_active_provider(self._fallback_provider)
            self._circuit_open = True  # Mark that we've failed over
            
            # Get a fresh model from the new provider
            fallback_model = self._get_fallback_model()
            
            if fallback_model:
                logger.info(f"🔄 Retrying with fallback provider '{self._fallback_provider}'...")
                response = fallback_model.invoke(input, **kwargs)
                
                # v2.5.2: Inject notice into response so agent knows about the switch
                return self._inject_fallback_notice(response)
            
        except Exception as fallback_exc:
            logger.error(f"❌ Fallback provider also failed: {fallback_exc}")
        
        return None

    async def _engage_circuit_breaker_async(self, input: Any, **kwargs: Any) -> Optional[Any]:
        """Async version of circuit breaker engagement."""
        # Track the original provider for the notice
        self._original_provider = self._manager.get_active_provider() if self._manager else "unknown"
        
        logger.warning(f"⚡ [ASYNC] CIRCUIT BREAKER: Switching from '{self._original_provider}' to fallback '{self._fallback_provider}'")
        
        try:
            # Switch the global provider state
            self._manager.set_active_provider(self._fallback_provider)
            self._circuit_open = True  # Mark that we've failed over
            
            # Get a fresh model from the new provider
            fallback_model = self._get_fallback_model()
            
            if fallback_model:
                logger.info(f"🔄 [ASYNC] Retrying with fallback provider '{self._fallback_provider}'...")
                # await the fallback model's ainvoke
                response = await fallback_model.ainvoke(input, **kwargs)
                
                # v2.5.2: Inject notice into response so agent knows about the switch
                return self._inject_fallback_notice(response)
            
        except Exception as fallback_exc:
            logger.error(f"❌ [ASYNC] Fallback provider also failed: {fallback_exc}")
        
        return None
    
    def _inject_fallback_notice(self, response: Any) -> Any:
        """Inject a system notice into the response about the provider switch.
        
        This makes the circuit breaker "Loud" - the agent's conversation
        history will include the notice, keeping its memory synced.
        
        Args:
            response: The AIMessage or response from the fallback model.
            
        Returns:
            Modified response with notice appended to content.
        """
        notice = FALLBACK_NOTICE_TEMPLATE.format(
            original_provider=self._original_provider.upper() if self._original_provider else "PRIMARY",
            fallback_provider=self._fallback_provider.upper(),
        )
        
        try:
            # Handle AIMessage objects (most common case)
            if hasattr(response, 'content'):
                original_content = response.content
                
                # Handle string content
                if isinstance(original_content, str):
                    response.content = original_content + notice
                    logger.info("📢 Circuit breaker notice injected into response")
                
                # Handle list content (multimodal - text + images)
                elif isinstance(original_content, list):
                    # Find the last text block and append notice
                    for i in range(len(original_content) - 1, -1, -1):
                        item = original_content[i]
                        if isinstance(item, dict) and item.get("type") == "text":
                            item["text"] = item.get("text", "") + notice
                            logger.info("📢 Circuit breaker notice injected into multimodal response")
                            break
                    else:
                        # No text block found, append as new text block
                        original_content.append({"type": "text", "text": notice})
                        logger.info("📢 Circuit breaker notice appended as new text block")
                
                # Handle other content types (unlikely but safe)
                else:
                    response.content = str(original_content) + notice
                    logger.info("📢 Circuit breaker notice injected (converted to string)")
            
            # Handle string responses directly
            elif isinstance(response, str):
                response = response + notice
                logger.info("📢 Circuit breaker notice appended to string response")
            
            # Handle dict responses
            elif isinstance(response, dict) and "content" in response:
                response["content"] = str(response["content"]) + notice
                logger.info("📢 Circuit breaker notice injected into dict response")
            
        except Exception as e:
            # Don't crash on notice injection failure
            logger.warning(f"Failed to inject circuit breaker notice: {e}")
        
        return response
    
    def _get_fallback_model(self) -> Optional[Any]:
        """Get a fresh model instance from the fallback provider.
        
        Returns:
            New model instance or None if unavailable.
        """
        try:
            # Import here to avoid circular imports
            from models.model_manager import get_smart_model
            
            # Get a fresh model (will use the new active provider)
            # Return the raw model, not wrapped (to avoid infinite recursion)
            model = get_smart_model()
            
            # If it's a SafeLLM, unwrap it
            if isinstance(model, SafeLLM):
                return model._primary_model
            
            return model
            
        except Exception as exc:
            logger.error(f"Failed to get fallback model: {exc}")
            return None
    
    # =========================================================================
    # LangChain Compatibility - Pass-through Methods
    # =========================================================================
    
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "SafeLLM":
        """Bind tools to the model, maintaining SafeLLM protection.
        
        When tools are bound, LangChain returns a new Runnable.
        We wrap that new Runnable in SafeLLM to maintain protection.
        
        Args:
            tools: List of tools to bind.
            **kwargs: Additional bind_tools arguments.
            
        Returns:
            New SafeLLM instance wrapping the tool-bound model.
        """
        bound_model = self._primary_model.bind_tools(tools, **kwargs)
        
        # Return a new SafeLLM wrapping the bound model
        return SafeLLM(
            primary_model=bound_model,
            manager=self._manager,
            fallback_provider=self._fallback_provider,
            max_retries=self._max_retries,
        )
    
    def with_structured_output(self, schema: Any, **kwargs: Any) -> "SafeLLM":
        """Apply structured output, maintaining SafeLLM protection.
        
        Args:
            schema: Output schema (Pydantic model or dict).
            **kwargs: Additional arguments.
            
        Returns:
            New SafeLLM instance with structured output.
        """
        structured_model = self._primary_model.with_structured_output(schema, **kwargs)
        
        return SafeLLM(
            primary_model=structured_model,
            manager=self._manager,
            fallback_provider=self._fallback_provider,
            max_retries=self._max_retries,
        )
    
    def __getattr__(self, name: str) -> Any:
        """Pass through attribute access to the underlying model.
        
        This ensures SafeLLM is fully transparent - any attribute
        not defined on SafeLLM is forwarded to the primary model.
        
        Args:
            name: Attribute name.
            
        Returns:
            Attribute from the primary model.
        """
        return getattr(self._primary_model, name)
    
    def __repr__(self) -> str:
        """String representation."""
        model_type = type(self._primary_model).__name__
        return f"SafeLLM({model_type}, fallback={self._fallback_provider})"


# =============================================================================
# Factory Function
# =============================================================================

def wrap_with_safety(
    model: Any,
    manager: Optional["ModelManager"] = None,
    fallback_provider: str = "nvidia",
) -> SafeLLM:
    """Wrap a LangChain model with SafeLLM protection.
    
    Convenience function for applying the circuit breaker pattern.
    
    Args:
        model: LangChain ChatModel to wrap.
        manager: ModelManager for provider switching.
        fallback_provider: Provider to fall back to.
        
    Returns:
        SafeLLM wrapper instance.
    """
    return SafeLLM(
        primary_model=model,
        manager=manager,
        fallback_provider=fallback_provider,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SafeLLM",
    "wrap_with_safety",
]
