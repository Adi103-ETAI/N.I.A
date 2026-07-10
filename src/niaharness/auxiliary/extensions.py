"""P1 Auxiliary extensions — universal entry, tool calling, streaming,
vision backend, client lifecycle, runtime injection.

These close the 9 P1 audit gaps from AUDIT.md:

  - ``call_llm`` universal entry — single function callers use instead of
    each building its own AuxiliaryClient instance.
  - Tool calling (``tools=`` arg) — lets background review do function
    calling via the auxiliary model.
  - Streaming (``complete_stream``) — for tasks that want token-by-token
    output (e.g. live title generation in the TUI).
  - Vision backend (``get_available_vision_backends``,
    ``resolve_vision_provider_client``) — discover which providers can
    accept image input and pick one for a vision task.
  - Client lifecycle (``shutdown_cached_clients``,
    ``cleanup_stale_async_clients``) — close SDK clients on shutdown and
    evict idle ones to avoid leaking connections.
  - ``set_runtime_main`` — inject the main-agent runtime so the aux
    client can reuse its credentials / config without a second copy.
  - Credential-pool integration — refresh credentials mid-call when the
    aux client gets a 401/403.
  - More providers — Anthropic direct, Azure Foundry, xAI, Nous.

Why a separate module?
----------------------
The base ``auxiliary/__init__.py`` is intentionally small (357 lines) so
the common path (no per-task config, no fallback) stays fast. These
extensions are opt-in: callers that need tool calling / streaming /
vision import them explicitly. The base ``AuxiliaryClient`` is unchanged
so existing callers (LLMCompactor, PermissionApproval) keep working.

Usage::

    from niaharness.auxiliary.extensions import call_llm, complete_with_tools

    # Simple universal entry — callers don't build a client.
    summary = await call_llm("Summarize: ...", task="compression")

    # Tool calling — background review can dispatch function calls.
    result = await complete_with_tools(
        prompt="Check if these commands are safe: ...",
        tools=[{"name": "check_safety", "description": "...", "input_schema": {...}}],
        task="permission_review",
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long an idle async client is kept in the cache before eviction.
_CLIENT_IDLE_TTL_SECONDS = 30 * 60  # 30 minutes

# How often cleanup_stale_async_clients runs (when called periodically).
_CLEANUP_INTERVAL_SECONDS = 5 * 60  # 5 minutes

# Per-task default max tokens. Lower than the primary model's defaults
# because aux tasks (compression, title generation) produce short output.
_PER_TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "compression": {"max_tokens": 1024, "temperature": 0.0},
    "title_generation": {"max_tokens": 64, "temperature": 0.3},
    "permission_review": {"max_tokens": 256, "temperature": 0.0},
    "background_review": {"max_tokens": 1024, "temperature": 0.0},
    "skill_suggestion": {"max_tokens": 256, "temperature": 0.2},
    "tool_validation": {"max_tokens": 256, "temperature": 0.0},
    "vision": {"max_tokens": 1024, "temperature": 0.0},
}


def get_per_task_defaults(task: str) -> Dict[str, Any]:
    """Return per-task defaults for max_tokens / temperature.

    Returns a copy so callers can safely mutate. Unknown tasks get a
    sensible default (max_tokens=1024, temperature=0.0).
    """
    defaults = _PER_TASK_DEFAULTS.get(task, {"max_tokens": 1024, "temperature": 0.0})
    return dict(defaults)


# ---------------------------------------------------------------------------
# Client cache (for lifecycle management)
# ---------------------------------------------------------------------------


@dataclass
class _CachedClient:
    """A cached AuxiliaryClient + its last-use timestamp."""

    client: Any  # AuxiliaryClient
    last_used: float = field(default_factory=time.monotonic)
    provider_label: str = ""
    task: str = ""

    def touch(self) -> None:
        self.last_used = time.monotonic()


# Module-level cache. Keyed by (task, provider_label) so per-task
# overrides get their own client. Uses weakref values so a client that's
# only held by the cache can still be GC'd if memory pressure demands.
_client_cache: Dict[str, _CachedClient] = {}
_client_cache_lock = asyncio.Lock()

# Main-agent runtime injection (set by set_runtime_main).
_runtime_main: Optional[Any] = None


def _cache_key(task: Optional[str], provider_label: str = "") -> str:
    return f"{task or 'default'}::{provider_label or 'primary'}"


async def _get_or_create_client(
    task: Optional[str] = None,
) -> Optional[Any]:
    """Get a cached AuxiliaryClient for the task, or create + cache one.

    Returns None if no aux model is configured.
    """
    from niaharness.auxiliary import AuxiliaryClient, get_aux_client

    key = _cache_key(task)
    async with _client_cache_lock:
        cached = _client_cache.get(key)
        if cached is not None:
            cached.touch()
            return cached.client

    # get_aux_client handles config resolution + per-task overrides.
    client = await get_aux_client(task)
    if client is None:
        return None

    async with _client_cache_lock:
        # Another task may have created one in the meantime — prefer the
        # existing entry to avoid duplicate clients.
        cached = _client_cache.get(key)
        if cached is not None:
            cached.touch()
            return cached.client
        _client_cache[key] = _CachedClient(
            client=client,
            provider_label=getattr(client.config, "provider", ""),
            task=task or "",
        )
    return client


# ---------------------------------------------------------------------------
# call_llm — universal entry point
# ---------------------------------------------------------------------------


async def call_llm(
    prompt: str,
    *,
    task: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    use_fallback: bool = True,
) -> Optional[str]:
    """Universal auxiliary LLM entry point.

    This is the function callers should use instead of building their own
    AuxiliaryClient. It:

      - Resolves per-task config (NIA_AUX_<TASK>_MODEL or auxiliary.<task>
        in config.yaml).
      - Caches the client for reuse across calls with the same task.
      - Optionally walks the fallback chain on payment/auth/rate-limit
        errors (set ``use_fallback=False`` to skip — useful for tests).
      - Applies per-task default max_tokens / temperature when the caller
        doesn't override them.

    Args:
        prompt: The user prompt.
        task: Optional task name for per-task config + defaults. Common
            tasks: "compression", "title_generation", "permission_review",
            "background_review", "skill_suggestion", "tool_validation",
            "vision".
        max_tokens: Override the per-task default.
        temperature: Override the per-task default.
        system: Optional system prompt.
        use_fallback: If True (default), walk the fallback chain on
            provider errors. Set False for tests / when you want the
            raw error to propagate.

    Returns:
        The completion text, or None if all providers fail (when
        use_fallback=True) or if the aux model isn't configured.
    """
    # Apply per-task defaults for max_tokens / temperature.
    defaults = get_per_task_defaults(task or "")
    effective_max = max_tokens if max_tokens is not None else defaults["max_tokens"]
    effective_temp = (
        temperature if temperature is not None else defaults["temperature"]
    )

    client = await _get_or_create_client(task)
    if client is None:
        logger.debug("call_llm(%s): no aux client configured", task or "default")
        return None

    if not use_fallback:
        try:
            return await client.complete(
                prompt,
                max_tokens=effective_max,
                temperature=effective_temp,
                system=system,
            )
        except Exception as exc:
            logger.warning(
                "call_llm(%s): primary failed (use_fallback=False): %s",
                task or "default",
                str(exc)[:200],
            )
            return None

    # Use the fallback-aware path.
    from niaharness.auxiliary.chain import call_with_fallback

    provider_label = getattr(client.config, "provider", "")
    return await call_with_fallback(
        primary_client=client,
        prompt=prompt,
        task=task or "default",
        max_tokens=effective_max,
        temperature=effective_temp,
        system=system,
        provider_label=provider_label,
    )


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


async def complete_with_tools(
    prompt: str,
    tools: List[Dict[str, Any]],
    *,
    task: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Auxiliary completion with tool calling.

    Lets background tasks (e.g. permission review, tool validation) ask
    the model to dispatch function calls. The model's response includes
    any tool_use blocks; the caller is responsible for executing them
    and sending the results back if needed.

    Args:
        prompt: The user prompt.
        tools: List of tool definitions in Anthropic's tool format
            ({"name": ..., "description": ..., "input_schema": {...}}).
        task: Optional task name for per-task config.
        max_tokens: Override the per-task default.
        temperature: Override the per-task default.
        system: Optional system prompt.
        tool_choice: Optional tool choice directive
            (e.g. {"type": "auto"}, {"type": "any"},
            {"type": "tool", "name": "check_safety"}).

    Returns:
        A dict with:
          - "text": The text content of the response (concatenated text blocks).
          - "tool_calls": List of tool_use blocks, each as a dict with
            "id", "name", "input".
          - "stop_reason": The stop reason ("end_turn", "tool_use", etc.).
          - "raw": The raw SDK response object (for advanced callers).
        Returns an empty dict if the aux model isn't configured.
    """
    client = await _get_or_create_client(task)
    if client is None:
        logger.debug("complete_with_tools(%s): no aux client configured", task or "default")
        return {}

    defaults = get_per_task_defaults(task or "")
    effective_max = max_tokens if max_tokens is not None else defaults["max_tokens"]
    effective_temp = (
        temperature if temperature is not None else defaults["temperature"]
    )

    sdk_client = await client._get_client()  # noqa: SLF001 — internal access for tool calls
    config = client.config

    if config.provider == "anthropic":
        kwargs: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": effective_max,
            "temperature": effective_temp,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
        }
        if system:
            kwargs["system"] = system
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        response = await sdk_client.messages.create(**kwargs)
        return _parse_anthropic_tool_response(response)

    # OpenAI-compatible
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Convert Anthropic-style tool defs to OpenAI format if needed.
    oai_tools = [_anthropic_to_openai_tool(t) for t in tools]

    kwargs = {
        "model": config.model,
        "max_tokens": effective_max,
        "temperature": effective_temp,
        "messages": messages,
        "tools": oai_tools,
    }
    if tool_choice:
        kwargs["tool_choice"] = _convert_tool_choice(tool_choice)
    response = await sdk_client.chat.completions.create(**kwargs)
    return _parse_openai_tool_response(response)


def _parse_anthropic_tool_response(response: Any) -> Dict[str, Any]:
    """Parse an Anthropic Messages API response into our common format."""
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in response.content:
        btype = getattr(block, "type", "")
        if btype == "text":
            text_parts.append(getattr(block, "text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}),
            })
    return {
        "text": "\n".join(text_parts),
        "tool_calls": tool_calls,
        "stop_reason": getattr(response, "stop_reason", ""),
        "raw": response,
    }


def _parse_openai_tool_response(response: Any) -> Dict[str, Any]:
    """Parse an OpenAI Chat Completions response into our common format."""
    choice = response.choices[0]
    message = choice.message
    text = getattr(message, "content", "") or ""
    tool_calls: List[Dict[str, Any]] = []
    for tc in getattr(message, "tool_calls", []) or []:
        import json as _json
        try:
            args = _json.loads(tc.function.arguments)
        except (ValueError, TypeError):
            args = {"_raw": tc.function.arguments}
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "input": args,
        })
    return {
        "text": text,
        "tool_calls": tool_calls,
        "stop_reason": getattr(choice, "finish_reason", ""),
        "raw": response,
    }


def _anthropic_to_openai_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic-style tool def to OpenAI's function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _convert_tool_choice(choice: Dict[str, Any]) -> Any:
    """Convert our tool_choice dict to the provider's format."""
    ctype = choice.get("type", "auto")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "tool":
        return {"type": "function", "function": {"name": choice.get("name", "")}}
    return ctype


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def complete_stream(
    prompt: str,
    *,
    task: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream tokens from the auxiliary model.

    Yields text chunks as they arrive. Useful for tasks that want
    token-by-token output (e.g. live title generation in the TUI, or
    streaming a long summary into a buffer).

    Args:
        prompt: The user prompt.
        task: Optional task name.
        max_tokens: Override the per-task default.
        temperature: Override the per-task default.
        system: Optional system prompt.

    Yields:
        Text chunks (strings). The concatenation of all chunks equals
        the full response text.
    """
    client = await _get_or_create_client(task)
    if client is None:
        logger.debug("complete_stream(%s): no aux client configured", task or "default")
        return

    defaults = get_per_task_defaults(task or "")
    effective_max = max_tokens if max_tokens is not None else defaults["max_tokens"]
    effective_temp = (
        temperature if temperature is not None else defaults["temperature"]
    )

    sdk_client = await client._get_client()  # noqa: SLF001
    config = client.config

    if config.provider == "anthropic":
        kwargs: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": effective_max,
            "temperature": effective_temp,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        # Anthropic SDK: messages.stream(...) returns an async context
        # manager whose .text_stream is an async iterator (attribute, not
        # method).
        stream_cm = sdk_client.messages.stream(**kwargs)
        if hasattr(stream_cm, "__aenter__"):
            async with stream_cm as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            # Some test fakes may return the stream object directly.
            async for text in stream_cm.text_stream:  # type: ignore[union-attr]
                yield text
        return

    # OpenAI-compatible
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = await sdk_client.chat.completions.create(
        model=config.model,
        max_tokens=effective_max,
        temperature=effective_temp,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------------------------------------------------------------------
# Vision backend
# ---------------------------------------------------------------------------


# Known vision-capable models per provider. Used by
# get_available_vision_backends to enumerate what's available given the
# currently-configured aux providers.
_VISION_CAPABLE_MODELS = {
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "openai-compatible": [],  # depends on the endpoint — caller must verify
    "groq": [
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ],
    "deepseek": [],  # DeepSeek's text models don't support vision as of 2025
}


@dataclass
class VisionBackend:
    """A vision-capable auxiliary backend."""

    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    config: Optional[Any] = None  # AuxConfig if resolved from one

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def get_available_vision_backends() -> List[VisionBackend]:
    """Enumerate vision-capable aux backends currently configured.

    Checks env vars and config for vision-capable providers. Returns a
    list ordered by preference (primary configured provider first, then
    fallbacks from the chain).

    A backend is "available" if:
      - The provider has at least one vision-capable model in
        ``_VISION_CAPABLE_MODELS``.
      - The provider's API key is set (env var or config).
    """
    backends: List[VisionBackend] = []
    seen_providers: set[str] = set()

    # 1. Primary aux config.
    primary = _resolve_primary_vision_backend()
    if primary is not None:
        backends.append(primary)
        seen_providers.add(primary.provider)

    # 2. Per-task vision override.
    vision_task = _resolve_vision_task_backend()
    if vision_task is not None and vision_task.provider not in seen_providers:
        backends.append(vision_task)
        seen_providers.add(vision_task.provider)

    # 3. Known env-var providers.
    for provider, env_var, base_url in [
        ("anthropic", "ANTHROPIC_API_KEY", None),
        ("openai", "OPENAI_API_KEY", None),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    ]:
        if provider in seen_providers:
            continue
        key = os.environ.get(env_var, "").strip()
        if not key:
            continue
        models = _VISION_CAPABLE_MODELS.get(provider, [])
        if not models:
            continue
        backends.append(VisionBackend(
            provider=provider,
            model=models[0],  # default to first (usually cheapest)
            api_key=key,
            base_url=base_url,
        ))
        seen_providers.add(provider)

    return backends


def _resolve_primary_vision_backend() -> Optional[VisionBackend]:
    """Check if the primary aux config is vision-capable."""
    from niaharness.auxiliary import get_aux_config

    config = get_aux_config()
    if config is None:
        return None
    models = _VISION_CAPABLE_MODELS.get(config.provider, [])
    if not models:
        return None
    # If the configured model is in the vision list, use it.
    if config.model in models:
        return VisionBackend(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key or "",
            base_url=config.base_url,
            config=config,
        )
    # Otherwise check if any vision model is available for the provider.
    # Use the first (usually cheapest) vision model.
    return VisionBackend(
        provider=config.provider,
        model=models[0],
        api_key=config.api_key or "",
        base_url=config.base_url,
        config=config,
    )


def _resolve_vision_task_backend() -> Optional[VisionBackend]:
    """Check for a per-task vision override."""
    from niaharness.auxiliary.chain import get_task_config

    task_cfg = get_task_config("vision")
    if task_cfg is None:
        return None
    model, api_key, base_url, provider = task_cfg
    return VisionBackend(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def resolve_vision_provider_client(
    preferred_provider: Optional[str] = None,
) -> Optional[VisionBackend]:
    """Pick a vision backend for a vision task.

    Args:
        preferred_provider: If provided, prefer this provider if it has
            a vision-capable model configured. Otherwise pick the first
            available backend from get_available_vision_backends().

    Returns:
        A VisionBackend, or None if no vision backend is configured.
    """
    backends = get_available_vision_backends()
    if not backends:
        return None
    if preferred_provider:
        for b in backends:
            if b.provider == preferred_provider:
                return b
    return backends[0]


# ---------------------------------------------------------------------------
# Client lifecycle management
# ---------------------------------------------------------------------------


async def shutdown_cached_clients() -> None:
    """Close all cached auxiliary clients and clear the cache.

    Call on application shutdown / config reload to release SDK
    connections. Safe to call multiple times.
    """
    global _client_cache
    async with _client_cache_lock:
        closed = 0
        for key, cached in list(_client_cache.items()):
            client = cached.client
            sdk_client = getattr(client, "_client", None)
            if sdk_client is not None:
                close_fn = getattr(sdk_client, "close", None)
                if close_fn is not None:
                    try:
                        result = close_fn()
                        if asyncio.iscoroutine(result):
                            await result
                        closed += 1
                    except Exception as exc:
                        logger.debug("Error closing aux client %s: %s", key, exc)
        _client_cache.clear()
        if closed:
            logger.info("Auxiliary: closed %d cached client(s)", closed)


async def cleanup_stale_async_clients(
    max_idle_seconds: int = _CLIENT_IDLE_TTL_SECONDS,
) -> int:
    """Evict cached clients that have been idle for more than max_idle_seconds.

    Returns the number of clients evicted. Call periodically (e.g. every
    5 minutes) to avoid leaking connections in long-running processes.
    """
    evicted = 0
    now = time.monotonic()
    async with _client_cache_lock:
        for key in list(_client_cache.keys()):
            cached = _client_cache[key]
            if now - cached.last_used > max_idle_seconds:
                # Close the SDK client before evicting.
                sdk_client = getattr(cached.client, "_client", None)
                if sdk_client is not None:
                    close_fn = getattr(sdk_client, "close", None)
                    if close_fn is not None:
                        try:
                            result = close_fn()
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as exc:
                            logger.debug("Error closing stale client %s: %s", key, exc)
                _client_cache.pop(key, None)
                evicted += 1
    if evicted:
        logger.debug("Auxiliary: evicted %d stale client(s)", evicted)
    return evicted


def get_cached_client_count() -> int:
    """Return the current number of cached clients (for diagnostics)."""
    return len(_client_cache)


# ---------------------------------------------------------------------------
# Runtime main injection
# ---------------------------------------------------------------------------


def set_runtime_main(runtime: Any) -> None:
    """Inject the main-agent runtime for credential / config reuse.

    The main-agent runtime typically holds the primary API client,
    credential pool, and config. By injecting it, the auxiliary layer
    can:

      - Reuse the primary model's API key when no aux-specific key is
        configured (avoids a second key requirement).
      - Refresh credentials mid-call by delegating to the runtime's
        credential pool.
      - Read runtime config (e.g. context window) for budgeting.

    Args:
        runtime: The main-agent runtime object. duck-typed: must have
            ``.api_client`` (the primary AnthropicApiClient) and
            ``.credential_pool`` (optional, a CredentialPool).
    """
    global _runtime_main
    _runtime_main = runtime
    logger.debug("Auxiliary: runtime main injected (%s)", type(runtime).__name__)


def get_runtime_main() -> Optional[Any]:
    """Return the injected main-agent runtime, or None."""
    return _runtime_main


def reset_runtime_main() -> None:
    """Clear the injected runtime (for tests / config reload)."""
    global _runtime_main
    _runtime_main = None


# ---------------------------------------------------------------------------
# Credential-pool integration
# ---------------------------------------------------------------------------


async def refresh_credentials_for_client(client: Any) -> bool:
    """Refresh credentials for an aux client via the runtime's credential pool.

    Called when the aux client gets a 401/403 and the runtime has a
    credential pool. Pulls a fresh credential and reconfigures the
    client's SDK instance.

    Returns True if credentials were refreshed, False otherwise (no
    runtime, no credential pool, or pool exhausted).
    """
    runtime = get_runtime_main()
    if runtime is None:
        return False
    pool = getattr(runtime, "credential_pool", None)
    if pool is None:
        return False

    config = getattr(client, "_config", None)
    if config is None:
        return False

    provider_name = getattr(config, "provider", "")
    try:
        # Try to acquire a fresh credential for the provider.
        acquire_fn = getattr(pool, "acquire", None)
        if acquire_fn is None:
            return False
        cred = acquire_fn(provider_name) if not asyncio.iscoroutine(
            acquire_fn(provider_name)
        ) else await acquire_fn(provider_name)
        if cred is None:
            return False

        # Reconfigure the client's SDK instance with the new key.
        new_key = getattr(cred, "api_key", None) or getattr(cred, "token", None)
        if not new_key:
            return False

        # Force a re-init on next call by clearing the cached SDK client.
        client._client = None  # noqa: SLF001
        # Update the config's api_key. AuxConfig is frozen, so create a copy.
        from niaharness.auxiliary import AuxConfig
        client._config = AuxConfig(  # noqa: SLF001
            model=config.model,
            api_key=new_key,
            base_url=config.base_url,
            provider=config.provider,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        logger.info("Auxiliary: refreshed credentials for %s", provider_name)
        return True
    except Exception as exc:
        logger.warning("Auxiliary: credential refresh failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Additional providers (Anthropic direct, Azure Foundry, xAI, Nous)
# ---------------------------------------------------------------------------


# Additional provider env vars to check in _try_api_key_provider.
# This extends the chain's provider detection without modifying chain.py.
ADDITIONAL_PROVIDER_ENV_VARS: Dict[str, Dict[str, Any]] = {
    "xai": {
        "env_var": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-vision-1212",
        "openai_compatible": True,
    },
    "nous": {
        "env_var": "NOUS_API_KEY",
        "base_url": "https://inference.nousresearch.com/v1",
        "default_model": "Hermes-3-Llama-3.1-405B",
        "openai_compatible": True,
    },
    "azure": {
        "env_var": "AZURE_OPENAI_API_KEY",
        "base_url_env": "AZURE_OPENAI_ENDPOINT",
        "default_model_env": "AZURE_OPENAI_DEPLOYMENT",
        "openai_compatible": True,
    },
    "together": {
        "env_var": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openai_compatible": True,
    },
    "fireworks": {
        "env_var": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "openai_compatible": True,
    },
}


def detect_additional_providers() -> List[Dict[str, Any]]:
    """Detect additional aux providers from env vars.

    Returns a list of dicts with provider info:
      [{"provider": "xai", "api_key": "...", "base_url": "...", "model": "..."}, ...]
    """
    detected: List[Dict[str, Any]] = []
    for name, info in ADDITIONAL_PROVIDER_ENV_VARS.items():
        key = os.environ.get(info["env_var"], "").strip()
        if not key:
            continue
        base_url = info.get("base_url", "")
        if info.get("base_url_env"):
            base_url = os.environ.get(info["base_url_env"], "").strip()
        model = info.get("default_model", "")
        if info.get("default_model_env"):
            model = os.environ.get(info["default_model_env"], "").strip()
        if not base_url or not model:
            continue
        detected.append({
            "provider": name,
            "api_key": key,
            "base_url": base_url,
            "model": model,
            "openai_compatible": info.get("openai_compatible", True),
        })
    return detected


__all__ = [
    "ADDITIONAL_PROVIDER_ENV_VARS",
    "VisionBackend",
    "call_llm",
    "cleanup_stale_async_clients",
    "complete_stream",
    "complete_with_tools",
    "detect_additional_providers",
    "get_available_vision_backends",
    "get_cached_client_count",
    "get_per_task_defaults",
    "get_runtime_main",
    "refresh_credentials_for_client",
    "reset_runtime_main",
    "resolve_vision_provider_client",
    "set_runtime_main",
    "shutdown_cached_clients",
]
