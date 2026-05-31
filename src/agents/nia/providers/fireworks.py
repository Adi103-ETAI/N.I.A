"""Fireworks AI provider."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import (
    LLMRequest, LLMResponse, ModelCapability, ModelInfo, ProviderInfo, ProviderStatus,
)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

BUILTIN_MODELS = [
    ModelInfo(id="accounts/fireworks/models/llama-v3p3-70b-instruct", name="Llama 3.3 70B", provider_id="fireworks", context_window=128000, max_output=8192, capabilities=[ModelCapability.CHAT]),
    ModelInfo(id="accounts/fireworks/models/llama-v3p1-8b-instruct", name="Llama 3.1 8B", provider_id="fireworks", context_window=128000, max_output=8192, capabilities=[ModelCapability.CHAT]),
    ModelInfo(id="accounts/fireworks/models/mixtral-8x22b-instruct", name="Mixtral 8x22B", provider_id="fireworks", context_window=65536, max_output=8192, capabilities=[ModelCapability.CHAT]),
]


class FireworksProvider(LLMProvider):
    def __init__(self) -> None:
        self._api_key: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None

    @property
    def id(self) -> str: return "fireworks"

    @property
    def name(self) -> str: return "Fireworks AI"

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        super().configure(api_key, base_url, **kwargs)
        if base_url: self._base_url = base_url
        if api_key: self._api_key = api_key; self._client = None

    def is_configured(self) -> bool: return bool(self._api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url.rstrip("/") + "/", headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}, timeout=60.0)
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(id=self.id, name=self.name, description="Fireworks AI inference", status=ProviderStatus.CONFIGURED if self.is_configured() else ProviderStatus.UNKNOWN, api_key_configured=self.is_configured(), base_url=self._base_url, models=self.list_models())

    def list_models(self) -> list[ModelInfo]: return BUILTIN_MODELS

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        messages = list(request.messages)
        if request.system: messages.insert(0, {"role": "system", "content": request.system})
        response = await client.post("/chat/completions", json={"model": request.model, "messages": messages, "max_tokens": request.max_tokens, "temperature": request.temperature})
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(content=choice["message"].get("content", "") or "", model=request.model, provider=self.id, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), finish_reason=choice.get("finish_reason", "stop"))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()
        messages = list(request.messages)
        if request.system: messages.insert(0, {"role": "system", "content": request.system})
        async with client.stream("POST", "/chat/completions", json={"model": request.model, "messages": messages, "max_tokens": request.max_tokens, "stream": True}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        content = event["choices"][0].get("delta", {}).get("content")
                        if content: yield content
                    except (json.JSONDecodeError, KeyError, IndexError): continue
