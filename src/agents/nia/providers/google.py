"""Google Gemini provider."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import (
    LLMRequest, LLMResponse, ModelCapability, ModelInfo, ProviderInfo, ProviderStatus,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

BUILTIN_MODELS = [
    ModelInfo(id="gemini-2.0-flash", name="Gemini 2.0 Flash", provider_id="google", context_window=1048576, max_output=8192, capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION]),
    ModelInfo(id="gemini-2.5-pro-preview-05-06", name="Gemini 2.5 Pro", provider_id="google", context_window=1048576, max_output=65536, capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.REASONING]),
    ModelInfo(id="gemini-1.5-pro", name="Gemini 1.5 Pro", provider_id="google", context_window=2097152, max_output=8192, capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS, ModelCapability.VISION]),
]


class GoogleProvider(LLMProvider):
    def __init__(self) -> None:
        self._api_key: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None

    @property
    def id(self) -> str: return "google"

    @property
    def name(self) -> str: return "Google Gemini"

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        super().configure(api_key, base_url, **kwargs)
        if base_url: self._base_url = base_url
        if api_key: self._api_key = api_key; self._client = None

    def is_configured(self) -> bool: return bool(self._api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(id=self.id, name=self.name, description="Google Gemini models", status=ProviderStatus.CONFIGURED if self.is_configured() else ProviderStatus.UNKNOWN, api_key_configured=self.is_configured(), base_url=self._base_url, models=self.list_models())

    def list_models(self) -> list[ModelInfo]: return BUILTIN_MODELS

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        model_path = f"models/{request.model}:generateContent"

        contents = []
        for m in request.messages:
            role = "user" if m.get("role") in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        response = await client.post(f"/{model_path}?key={self._api_key}", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        usage = data.get("usageMetadata", {})

        return LLMResponse(
            content=text,
            model=request.model,
            provider=self.id,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            finish_reason="stop",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()
        model_path = f"models/{request.model}:streamGenerateContent"

        contents = []
        for m in request.messages:
            role = "user" if m.get("role") in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        async with client.stream("POST", f"/{model_path}?key={self._api_key}&alt=sse", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        text = event.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text: yield text
                    except (json.JSONDecodeError, KeyError, IndexError): continue
