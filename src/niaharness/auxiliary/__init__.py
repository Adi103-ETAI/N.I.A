"""Auxiliary model client — small, fast model for background tasks.

Ported from the reference project's agent/auxiliary_client.py (8,121 lines),
providing a separate LLM client for background tasks that don't need the
full power (or cost) of the primary model:

  - **Context compaction** — summarize old messages (LLMCompactor uses this)
  - **Permission smart approval** — LLM-based dangerous-command judgment
  - **Background review** — periodic self-improvement review of recent turns
  - **Skill suggestions** — recommend skills based on conversation context
  - **Title generation** — auto-generate session titles
  - **Tool call validation** — sanity-check tool arguments before execution

Why a separate client?
----------------------
Using the primary model (e.g. Claude 4 Opus at $15/$75 per 1M tokens) for
background tasks wastes money. The auxiliary model (e.g. Claude 3 Haiku at
$0.25/$1.25 per 1M tokens) is 60x cheaper and fast enough for these tasks.

Configuration
-------------
The aux model is configured via ``NIA_AUX_MODEL`` and ``NIA_AUX_API_KEY``
env vars, or via ``auxiliary.model`` / ``auxiliary.api_key`` in config.yaml.
Defaults to the primary model if not configured (so existing setups keep
working).

Usage::

    from niaharness.auxiliary import get_aux_client

    client = get_aux_client()
    if client is not None:
        summary = await client.complete("Summarize: ...", max_tokens=1024)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxConfig:
    """Configuration for the auxiliary model client.

    Attributes:
        model: The model name (e.g. 'claude-3-haiku-20240307').
        api_key: The API key (or None to use the primary model's key).
        base_url: Optional base URL override (for self-hosted/OpenAI-compatible endpoints).
        provider: The provider name (e.g. 'anthropic', 'openai').
        max_tokens: Default max tokens for completions.
        temperature: Default temperature (0.0 = deterministic, good for summaries).
    """

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    provider: str = "anthropic"
    max_tokens: int = 1024
    temperature: float = 0.0


def get_aux_config(task: Optional[str] = None) -> Optional[AuxConfig]:
    """Return the auxiliary model configuration, or None if not configured.

    Resolution order:
      1. Per-task override: ``NIA_AUX_<TASK>_MODEL`` + ``NIA_AUX_<TASK>_API_KEY``
         env vars or ``auxiliary.<task>.model`` in config.yaml
      2. ``NIA_AUX_MODEL`` + ``NIA_AUX_API_KEY`` env vars
      3. ``auxiliary.model`` + ``auxiliary.api_key`` in config.yaml
      4. None (aux model disabled — primary model will be used)

    Args:
        task: Optional task name for per-task overrides (e.g. "compression",
              "vision", "title_generation", "permission").
    """
    # 0. Per-task override.
    if task:
        from niaharness.auxiliary.chain import get_task_config
        task_config = get_task_config(task)
        if task_config is not None:
            model, api_key, base_url, provider = task_config
            return AuxConfig(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
            )

    # 1. Environment variables.
    model = os.environ.get("NIA_AUX_MODEL", "").strip()
    api_key = os.environ.get("NIA_AUX_API_KEY", "").strip()
    base_url = os.environ.get("NIA_AUX_BASE_URL", "").strip() or None
    provider = os.environ.get("NIA_AUX_PROVIDER", "anthropic").strip()

    if model and api_key:
        return AuxConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
        )

    # 2. Config file.
    try:
        from niaharness.config.settings import load_settings

        settings = load_settings()
        aux_section = getattr(settings, "auxiliary", None) or {}
        if isinstance(aux_section, dict):
            model = aux_section.get("model", "").strip()
            api_key = aux_section.get("api_key", "").strip()
            base_url = aux_section.get("base_url", "").strip() or None
            provider = aux_section.get("provider", "anthropic").strip()
            if model and api_key:
                return AuxConfig(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    provider=provider,
                )
    except Exception as exc:
        logger.debug("Could not load auxiliary config from settings: %s", exc)

    # 3. Not configured.
    return None


# ---------------------------------------------------------------------------
# Auxiliary client
# ---------------------------------------------------------------------------


class AuxiliaryClient:
    """LLM client for background tasks (compaction, review, titles, etc.).

    Wraps the Anthropic SDK (or OpenAI SDK for OpenAI-compatible providers)
    with a simple ``complete(prompt) -> str`` interface. The client is
    separate from the primary API client so it can use a different (cheaper,
    faster) model.

    All completions are non-streaming (background tasks don't need streaming)
    and use temperature=0.0 by default (deterministic, good for summaries).
    """

    def __init__(self, config: AuxConfig) -> None:
        self._config = config
        self._client: Any = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> AuxConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    async def _get_client(self) -> Any:
        """Lazily initialize the SDK client."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            if self._config.provider == "anthropic":
                from anthropic import AsyncAnthropic

                kwargs: Dict[str, Any] = {"api_key": self._config.api_key}
                if self._config.base_url:
                    kwargs["base_url"] = self._config.base_url
                self._client = AsyncAnthropic(**kwargs)
            elif self._config.provider in ("openai", "openai-compatible"):
                from openai import AsyncOpenAI

                kwargs = {"api_key": self._config.api_key}
                if self._config.base_url:
                    kwargs["base_url"] = self._config.base_url
                self._client = AsyncOpenAI(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported auxiliary provider: {self._config.provider}. "
                    "Supported: anthropic, openai, openai-compatible."
                )
            return self._client

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> str:
        """Generate a completion for the prompt. Returns the text response.

        Args:
            prompt: The user prompt.
            max_tokens: Max tokens for the response (defaults to config.max_tokens).
            temperature: Temperature (defaults to config.temperature).
            system: Optional system prompt.

        Returns:
            The generated text.
        """
        client = await self._get_client()
        max_t = max_tokens or self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature

        if self._config.provider == "anthropic":
            kwargs: Dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": max_t,
                "temperature": temp,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = await client.messages.create(**kwargs)
            # Extract text from response.
            parts: List[str] = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        else:  # openai-compatible
            kwargs = {
                "model": self._config.model,
                "max_tokens": max_t,
                "temperature": temp,
                "messages": [],
            }
            if system:
                kwargs["messages"].append({"role": "system", "content": system})
            kwargs["messages"].append({"role": "user", "content": prompt})
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a completion and parse it as JSON.

        The prompt should ask the model to produce valid JSON. The response
        is parsed with ``json.loads``; if parsing fails, an empty dict is
        returned.
        """
        import json

        text = await self.complete(prompt, max_tokens=max_tokens, system=system)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from markdown code blocks.
            import re

            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, TypeError):
                    pass
            logger.warning("Auxiliary complete_json: failed to parse response as JSON")
            return {}


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------


_aux_client: Optional[AuxiliaryClient] = None
_aux_client_lock = asyncio.Lock()


async def get_aux_client(task: Optional[str] = None) -> Optional[AuxiliaryClient]:
    """Return the process-wide AuxiliaryClient, or None if not configured.

    The client is lazily initialized on first call. If a task name is
    provided, per-task config overrides are checked first.

    Args:
        task: Optional task name for per-task overrides (e.g. "compression").

    Returns:
        The AuxiliaryClient, or None if no aux model is configured.
    """
    global _aux_client
    if _aux_client is not None and task is None:
        return _aux_client
    async with _aux_client_lock:
        if _aux_client is not None and task is None:
            return _aux_client
        config = get_aux_config(task)
        if config is None:
            return None
        if task is None:
            _aux_client = AuxiliaryClient(config)
            return _aux_client
        else:
            # Return a per-task client (not cached)
            return AuxiliaryClient(config)


def reset_aux_client() -> None:
    """Reset the singleton client (useful for config changes / tests)."""
    global _aux_client
    _aux_client = None


# Re-export chain functions for convenience
from niaharness.auxiliary.chain import (  # noqa: E402
    call_with_fallback,
    is_auth_error,
    is_connection_error,
    is_model_not_found_error,
    is_payment_error,
    is_provider_unhealthy,
    is_rate_limit_error,
    is_transient_transport_error,
    mark_provider_unhealthy,
    reset_unhealthy_cache,
    try_payment_fallback,
)

# Re-export P1 extension functions
from niaharness.auxiliary.extensions import (  # noqa: E402
    ADDITIONAL_PROVIDER_ENV_VARS,
    VisionBackend,
    call_llm,
    cleanup_stale_async_clients,
    complete_stream,
    complete_with_tools,
    detect_additional_providers,
    get_auxiliary_extra_body,
    get_available_vision_backends,
    get_cached_client_count,
    get_per_task_defaults,
    get_runtime_main,
    get_text_auxiliary_client,
    refresh_credentials_for_client,
    reset_runtime_main,
    resolve_vision_provider_client,
    set_runtime_main,
    shutdown_cached_clients,
)


__all__ = [
    "AuxConfig",
    "AuxiliaryClient",
    "get_aux_client",
    "get_aux_config",
    "reset_aux_client",
    # Re-export chain functions for convenience
    "call_with_fallback",
    "is_payment_error",
    "is_auth_error",
    "is_rate_limit_error",
    "is_connection_error",
    "mark_provider_unhealthy",
    "is_provider_unhealthy",
    "try_payment_fallback",
    "reset_unhealthy_cache",
    # P1 extensions
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
