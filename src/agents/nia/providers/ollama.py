"""Ollama provider for local models."""

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

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self) -> None:
        self._api_key: str = ""  # Not needed for Ollama
        self._base_url: str = DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None
        self._cached_models: list[ModelInfo] | None = None

    @property
    def id(self) -> str:
        return "ollama"

    @property
    def name(self) -> str:
        return "Ollama (Local)"

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        if base_url:
            self._base_url = base_url
        self._client = None
        self._cached_models = None

    def is_configured(self) -> bool:
        # Ollama is always "configured" if running locally
        return True

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Ensure base_url ends with / for proper path joining
            base_url = self._base_url.rstrip("/") + "/"
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 min for LLM responses
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            name=self.name,
            description="Local models via Ollama",
            status=ProviderStatus.CONFIGURED,
            api_key_configured=True,
            base_url=self._base_url,
            models=self.list_models(),
        )

    def list_models(self) -> list[ModelInfo]:
        """Return cached models. Use fetch_models() to refresh from API."""
        if self._cached_models is not None:
            return self._cached_models
        # Return empty list if not yet cached
        return []

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from Ollama API and cache them."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("models", []):
                models.append(ModelInfo(
                    id=model["name"],
                    name=model.get("name", model["name"]),
                    provider_id=self.id,
                    context_window=4096,  # Default, varies by model
                    max_output=2048,
                    capabilities=[ModelCapability.CHAT, ModelCapability.TOOLS],
                ))

            self._cached_models = models
            return models
        except Exception:
            return self._cached_models or []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        # Build prompt from messages
        prompt_parts = []
        if request.system:
            prompt_parts.append(f"System: {request.system}")
        for m in request.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max(request.max_tokens, 2048),  # Ensure minimum tokens
                "temperature": request.temperature,
            },
        }

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        content = data.get("response", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return LLMResponse(
            content=content,
            model=request.model,
            provider=self.id,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            finish_reason="stop",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()

        # Build prompt from messages
        prompt_parts = []
        if request.system:
            prompt_parts.append(f"System: {request.system}")
        for m in request.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        async with client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                try:
                    event = json.loads(line)
                    content = event.get("response", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
