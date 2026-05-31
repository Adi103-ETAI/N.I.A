"""OpenAI-compatible provider.

Works with OpenAI, and any OpenAI-compatible API (DashScope, GitHub Models, etc.)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import (
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ModelInfo,
    ProviderInfo,
    ProviderStatus,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"

BUILTIN_MODELS = [
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider_id="openai",
        context_window=128000,
        max_output=16384,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.STREAMING],
        cost_input=2.50,
        cost_output=10.0,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider_id="openai",
        context_window=128000,
        max_output=16384,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION],
        cost_input=0.15,
        cost_output=0.60,
    ),
    ModelInfo(
        id="o3-mini",
        name="o3-mini",
        provider_id="openai",
        context_window=200000,
        max_output=100000,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.REASONING],
        cost_input=1.10,
        cost_output=4.40,
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider_id="openai",
        context_window=128000,
        max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION],
        cost_input=10.0,
        cost_output=30.0,
    ),
]


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider.

    Works with OpenAI API and any compatible endpoint.
    Fetches models from /v1/models endpoint when available.
    """

    def __init__(self) -> None:
        self._api_key: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None
        self._cached_models: list[ModelInfo] | None = None

    @property
    def id(self) -> str:
        return "openai"

    @property
    def name(self) -> str:
        return "OpenAI"

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        super().configure(api_key, base_url, **kwargs)
        if base_url:
            self._base_url = base_url
        if api_key:
            self._api_key = api_key
            self._client = None
            self._cached_models = None  # Reset cache on reconfigure

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url.rstrip("/") + "/",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            name=self.name,
            description="OpenAI GPT models and OpenAI-compatible APIs",
            status=ProviderStatus.CONFIGURED if self.is_configured() else ProviderStatus.UNKNOWN,
            api_key_configured=self.is_configured(),
            base_url=self._base_url,
            models=self.list_models(),
        )

    def list_models(self) -> list[ModelInfo]:
        """Return cached models or hardcoded fallback."""
        return self._cached_models or BUILTIN_MODELS

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from /v1/models endpoint.

        Falls back to hardcoded list if API call fails.
        Caches results for subsequent list_models() calls.
        """
        if not self.is_configured():
            return BUILTIN_MODELS

        try:
            client = self._get_client()
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("data", []):
                model_id = item.get("id", "")
                if not model_id:
                    continue

                # Skip embedding models and other non-chat models
                if any(skip in model_id.lower() for skip in ["embedding", "embed", "moderation", "whisper", "tts", "dall-e"]):
                    continue

                models.append(ModelInfo(
                    id=model_id,
                    name=model_id.split("/")[-1],  # Use last part of ID as name
                    provider_id=self.id,
                    context_window=128000,  # Default, providers may not report this
                    max_output=4096,
                    capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS],
                ))

            if models:
                self._cached_models = models
                logger.info(f"Fetched {len(models)} models from {self.name}")
                return models

        except Exception as e:
            logger.debug(f"Failed to fetch models from {self.name}: {e}")

        # Fall back to hardcoded
        return BUILTIN_MODELS

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        messages = list(request.messages)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        content = choice["message"].get("content", "") or ""
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=request.model,
            provider=self.id,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()

        messages = list(request.messages)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
