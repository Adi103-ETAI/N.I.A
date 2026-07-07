"""OpenAI-compatible provider (includes Ollama, OpenRouter, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from niaharness.providers.base import (
    AuthMode,
    LLMProvider,
    ProviderAuthConfig,
    ProviderCapabilities,
    ProviderCategory,
    ProviderConfig,
    ProviderModel,
)


class OpenAIProvider(LLMProvider):
    """OpenAI and OpenAI-compatible providers.

    Supports: OpenAI, Ollama, OpenRouter, Groq, Together, DeepSeek, etc.
    Fetches models from /v1/models endpoint when API key is configured.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="openai",
            label="OpenAI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["OPENAI_API_KEY"],
                base_url_env_vars=["OPENAI_BASE_URL"],
                model_env_vars=["OPENAI_MODEL"],
                default_base_url="https://api.openai.com/v1",
                default_model="gpt-4o",
            ),
            models=[
                # Hardcoded fallbacks - actual models fetched from API
                ProviderModel(id="gpt-4o", label="GPT-4o", context_window=128000, max_output_tokens=16384),
                ProviderModel(id="gpt-4o-mini", label="GPT-4o Mini", context_window=128000, max_output_tokens=16384),
                ProviderModel(id="o3-mini", label="o3-mini", context_window=200000, max_output_tokens=100000),
            ],
            is_first_party=True,
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx
        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))
        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
            timeout=60.0,
        )

    async def fetch_models(self) -> list[ProviderModel]:
        """Fetch models from /v1/models endpoint."""
        api_key = self.resolve_api_key()
        if not api_key:
            return self.config.models

        try:
            import httpx
            base_url = self.resolve_base_url()
            client = httpx.AsyncClient(
                base_url=base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("data", []):
                model_id = item.get("id", "")
                if not model_id:
                    continue
                # Skip embedding/moderation/tts models
                skip = ["embedding", "embed", "moderation", "whisper", "tts", "dall-e"]
                if any(s in model_id.lower() for s in skip):
                    continue
                models.append(ProviderModel(
                    id=model_id,
                    label=model_id.split("/")[-1],
                    context_window=128000,
                    max_output_tokens=4096,
                ))

            if models:
                self._fetched_models = models
                logger.info(f"Fetched {len(models)} models from OpenAI")
                return models

        except Exception as e:
            logger.debug(f"Failed to fetch models from OpenAI: {e}")

        return self.config.models


class OllamaProvider(LLMProvider):
    """Ollama local model provider.

    Dynamically fetches models from Ollama API.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="ollama",
            label="Ollama (Local)",
            category=ProviderCategory.LOCAL,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                default_base_url="http://localhost:11434",
                default_model="",
            ),
            models=[],  # Dynamically loaded
            supports_model_routing=False,
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx
        base_url = self.resolve_base_url(kwargs.get("base_url"))
        return httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=120.0)

    def is_configured(self) -> bool:
        """Check if Ollama is running locally."""
        import httpx
        try:
            client = httpx.Client(base_url="http://localhost:11434/", timeout=2.0)
            response = client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def fetch_models(self) -> list[ProviderModel]:
        """Fetch models from Ollama API."""
        try:
            import httpx
            base_url = self.resolve_base_url()
            client = httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=10.0)
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("models", []):
                models.append(ProviderModel(
                    id=m["name"],
                    label=m["name"],
                    context_window=4096,
                    max_output_tokens=2048,
                ))

            if models:
                self._fetched_models = models
                logger.info(f"Fetched {len(models)} models from Ollama")
                return models

        except Exception as e:
            logger.debug(f"Failed to fetch models from Ollama: {e}")

        return []


class OpenRouterProvider(LLMProvider):
    """OpenRouter multi-provider gateway."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="openrouter",
            label="OpenRouter",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["OPENROUTER_API_KEY"],
                default_base_url="https://openrouter.ai/api/v1",
                default_model="anthropic/claude-sonnet-4",
            ),
            models=[
                ProviderModel(
                    id="anthropic/claude-sonnet-4",
                    label="Claude Sonnet 4 (via OpenRouter)",
                    context_window=200000,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="openai/gpt-4o",
                    label="GPT-4o (via OpenRouter)",
                    context_window=128000,
                    max_output_tokens=16384,
                ),
                ProviderModel(
                    id="google/gemini-2.0-flash-001",
                    label="Gemini 2.0 Flash (via OpenRouter)",
                    context_window=1048576,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="meta-llama/llama-3.3-70b-instruct",
                    label="Llama 3.3 70B (via OpenRouter)",
                    context_window=128000,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/nia",
                "X-Title": "N.I.A",
            },
            timeout=60.0,
        )


class GroqProvider(LLMProvider):
    """Groq fast inference provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="groq",
            label="Groq",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["GROQ_API_KEY"],
                default_base_url="https://api.groq.com/openai/v1",
                default_model="llama-3.3-70b-versatile",
            ),
            models=[
                ProviderModel(
                    id="llama-3.3-70b-versatile",
                    label="Llama 3.3 70B",
                    context_window=128000,
                    max_output_tokens=32768,
                ),
                ProviderModel(
                    id="mixtral-8x7b-32768",
                    label="Mixtral 8x7B",
                    context_window=32768,
                    max_output_tokens=32768,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class TogetherProvider(LLMProvider):
    """Together AI provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="together",
            label="Together AI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["TOGETHER_API_KEY"],
                default_base_url="https://api.together.xyz/v1",
                default_model="meta-llama/Llama-3-70b-chat-hf",
            ),
            models=[
                ProviderModel(
                    id="meta-llama/Llama-3-70b-chat-hf",
                    label="Llama 3 70B",
                    context_window=8192,
                    max_output_tokens=4096,
                ),
                ProviderModel(
                    id="Qwen/Qwen2.5-72B-Instruct-Turbo",
                    label="Qwen 2.5 72B",
                    context_window=32768,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class DeepSeekProvider(LLMProvider):
    """DeepSeek AI provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="deepseek",
            label="DeepSeek",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["DEEPSEEK_API_KEY"],
                default_base_url="https://api.deepseek.com/v1",
                default_model="deepseek-chat",
            ),
            models=[
                ProviderModel(
                    id="deepseek-chat",
                    label="DeepSeek V3",
                    context_window=65536,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="deepseek-reasoner",
                    label="DeepSeek R1",
                    context_window=65536,
                    max_output_tokens=8192,
                    capabilities=ProviderCapabilities(supports_reasoning=True),
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="google",
            label="Google Gemini",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["GOOGLE_API_KEY"],
                default_base_url="https://generativelanguage.googleapis.com/v1beta",
                default_model="gemini-2.0-flash",
            ),
            models=[
                ProviderModel(
                    id="gemini-2.0-flash",
                    label="Gemini 2.0 Flash",
                    context_window=1048576,
                    max_output_tokens=8192,
                    capabilities=ProviderCapabilities(supports_vision=True),
                ),
                ProviderModel(
                    id="gemini-2.5-pro-preview-05-06",
                    label="Gemini 2.5 Pro",
                    context_window=1048576,
                    max_output_tokens=65536,
                    capabilities=ProviderCapabilities(
                        supports_vision=True,
                        supports_reasoning=True,
                    ),
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=60.0,
        )


class NVIDIAProvider(LLMProvider):
    """NVIDIA NIM provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="nvidia",
            label="NVIDIA NIM",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["NVIDIA_API_KEY"],
                default_base_url="https://integrate.api.nvidia.com/v1",
                default_model="meta/llama-3.1-405b-instruct",
            ),
            models=[
                ProviderModel(
                    id="meta/llama-3.1-405b-instruct",
                    label="Llama 3.1 405B",
                    context_window=131072,
                    max_output_tokens=4096,
                ),
                ProviderModel(
                    id="meta/llama-3.1-70b-instruct",
                    label="Llama 3.1 70B",
                    context_window=131072,
                    max_output_tokens=4096,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class CerebrasProvider(LLMProvider):
    """Cerebras fast inference provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="cerebras",
            label="Cerebras",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["CEREBRAS_API_KEY"],
                default_base_url="https://api.cerebras.ai/v1",
                default_model="llama-3.3-70b",
            ),
            models=[
                ProviderModel(
                    id="llama-3.3-70b",
                    label="Llama 3.3 70B",
                    context_window=128000,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class FireworksProvider(LLMProvider):
    """Fireworks AI provider."""

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="fireworks",
            label="Fireworks AI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["FIREWORKS_API_KEY"],
                default_base_url="https://api.fireworks.ai/inference/v1",
                default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            ),
            models=[
                ProviderModel(
                    id="accounts/fireworks/models/llama-v3p3-70b-instruct",
                    label="Llama 3.3 70B",
                    context_window=128000,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class OpenCodeProvider(LLMProvider):
    """OpenCode Zen — curated AI model gateway by the OpenCode team.

    OpenAI-compatible API. Get an API key at https://opencode.ai/zen.
    Models use the 'opencode/' prefix (e.g. opencode/gpt-4o).
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="opencode",
            label="OpenCode Zen",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["OPENCODE_API_KEY", "OPENCODE_ZEN_API_KEY"],
                base_url_env_vars=["OPENCODE_BASE_URL"],
                model_env_vars=["OPENCODE_MODEL"],
                default_base_url="https://opencode.ai/zen/v1",
                default_model="opencode/gpt-4o",
            ),
            models=[
                ProviderModel(
                    id="opencode/gpt-4o",
                    label="GPT-4o (via Zen)",
                    context_window=128000,
                    max_output_tokens=16384,
                ),
                ProviderModel(
                    id="opencode/claude-sonnet-4",
                    label="Claude Sonnet 4 (via Zen)",
                    context_window=200000,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class XAIProvider(LLMProvider):
    """xAI — Grok models (Grok-3, Grok-4, etc.).

    OpenAI-compatible API. Get an API key at https://x.ai.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="xai",
            label="xAI (Grok)",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["XAI_API_KEY", "GROK_API_KEY"],
                base_url_env_vars=["XAI_BASE_URL"],
                model_env_vars=["XAI_MODEL"],
                default_base_url="https://api.x.ai/v1",
                default_model="grok-4",
            ),
            models=[
                ProviderModel(
                    id="grok-4",
                    label="Grok 4",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="grok-3",
                    label="Grok 3",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="grok-3-mini",
                    label="Grok 3 Mini",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class PerplexityProvider(LLMProvider):
    """Perplexity — online models with built-in web search.

    OpenAI-compatible API. Get an API key at https://perplexity.ai.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="perplexity",
            label="Perplexity",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["PERPLEXITY_API_KEY"],
                base_url_env_vars=["PERPLEXITY_BASE_URL"],
                model_env_vars=["PERPLEXITY_MODEL"],
                default_base_url="https://api.perplexity.ai",
                default_model="sonar-pro",
            ),
            models=[
                ProviderModel(
                    id="sonar-pro",
                    label="Sonar Pro",
                    context_window=200000,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="sonar",
                    label="Sonar",
                    context_window=127072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="sonar-reasoning",
                    label="Sonar Reasoning",
                    context_window=127072,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class DeepInfraProvider(LLMProvider):
    """DeepInfra — fast inference for open-source models.

    OpenAI-compatible API. Get an API key at https://deepinfra.com.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="deepinfra",
            label="DeepInfra",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["DEEPINFRA_API_KEY"],
                base_url_env_vars=["DEEPINFRA_BASE_URL"],
                model_env_vars=["DEEPINFRA_MODEL"],
                default_base_url="https://api.deepinfra.com/v1/openai",
                default_model="meta-llama/Llama-3.3-70B-Instruct",
            ),
            models=[
                ProviderModel(
                    id="meta-llama/Llama-3.3-70B-Instruct",
                    label="Llama 3.3 70B",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="meta-llama/Meta-Llama-3.1-405B-Instruct",
                    label="Llama 3.1 405B",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="deepseek-ai/DeepSeek-R1",
                    label="DeepSeek R1",
                    context_window=131072,
                    max_output_tokens=32768,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API — 200K+ open-source models.

    OpenAI-compatible API. Get an API key at https://huggingface.co/settings/tokens.
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="huggingface",
            label="Hugging Face",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["HF_API_KEY", "HUGGINGFACE_API_KEY"],
                base_url_env_vars=["HF_BASE_URL"],
                model_env_vars=["HF_MODEL"],
                default_base_url="https://api-inference.huggingface.co/v1",
                default_model="meta-llama/Llama-3.3-70B-Instruct",
            ),
            models=[
                ProviderModel(
                    id="meta-llama/Llama-3.3-70B-Instruct",
                    label="Llama 3.3 70B",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
                ProviderModel(
                    id="meta-llama/Llama-3.1-8B-Instruct",
                    label="Llama 3.1 8B",
                    context_window=131072,
                    max_output_tokens=8192,
                ),
            ],
        )

    def get_client(self, api_key: str | None = None, **kwargs: Any) -> Any:
        import httpx

        resolved_key = self.resolve_api_key(api_key)
        base_url = self.resolve_base_url(kwargs.get("base_url"))

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
