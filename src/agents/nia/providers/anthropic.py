"""Anthropic Claude provider."""

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

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

BUILTIN_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider_id="anthropic",
        context_window=200000,
        max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.REASONING],
        cost_input=3.0,
        cost_output=15.0,
    ),
    ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider_id="anthropic",
        context_window=200000,
        max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.REASONING],
        cost_input=15.0,
        cost_output=75.0,
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider_id="anthropic",
        context_window=200000,
        max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION],
        cost_input=0.80,
        cost_output=4.0,
    ),
]


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using native API."""

    def __init__(self) -> None:
        self._api_key: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None

    @property
    def id(self) -> str:
        return "anthropic"

    @property
    def name(self) -> str:
        return "Anthropic"

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        super().configure(api_key, base_url, **kwargs)
        if base_url:
            self._base_url = base_url
        if api_key:
            self._api_key = api_key
            self._client = None  # Reset client

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            name=self.name,
            description="Anthropic's Claude models",
            status=ProviderStatus.CONFIGURED if self.is_configured() else ProviderStatus.UNKNOWN,
            api_key_configured=self.is_configured(),
            base_url=self._base_url,
            models=self.list_models(),
        )

    def list_models(self) -> list[ModelInfo]:
        return BUILTIN_MODELS

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        messages = [m for m in request.messages if m.get("role") != "system"]
        system = request.system or ""
        if not system:
            for m in request.messages:
                if m.get("role") == "system":
                    system = m.get("content", "")
                    break

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        response = await client.post("/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=request.model,
            provider=self.id,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()

        messages = [m for m in request.messages if m.get("role") != "system"]
        system = request.system or ""

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with client.stream("POST", "/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except json.JSONDecodeError:
                        continue
