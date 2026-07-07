"""Tests for the new LLM providers added in the P1 provider expansion."""

from __future__ import annotations

import pytest

from niaharness.providers.openai import (
    OpenCodeProvider,
    XAIProvider,
    PerplexityProvider,
    DeepInfraProvider,
    HuggingFaceProvider,
)
from niaharness.providers.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# New provider config tests
# ---------------------------------------------------------------------------


class TestOpenCodeProvider:
    def test_config_name(self):
        p = OpenCodeProvider()
        assert p.config.name == "opencode"
        assert p.config.label == "OpenCode Zen"

    def test_base_url(self):
        p = OpenCodeProvider()
        assert p.config.auth.default_base_url == "https://opencode.ai/zen/v1"

    def test_env_vars(self):
        p = OpenCodeProvider()
        assert "OPENCODE_API_KEY" in p.config.auth.api_key_env_vars
        assert "OPENCODE_ZEN_API_KEY" in p.config.auth.api_key_env_vars

    def test_default_model(self):
        p = OpenCodeProvider()
        assert p.config.auth.default_model == "opencode/gpt-4o"

    def test_models_nonempty(self):
        p = OpenCodeProvider()
        assert len(p.config.models) >= 2

    def test_get_client(self):
        p = OpenCodeProvider()
        client = p.get_client(api_key="test-key")
        assert client is not None


class TestXAIProvider:
    def test_config_name(self):
        p = XAIProvider()
        assert p.config.name == "xai"
        assert p.config.label == "xAI (Grok)"

    def test_base_url(self):
        p = XAIProvider()
        assert p.config.auth.default_base_url == "https://api.x.ai/v1"

    def test_env_vars(self):
        p = XAIProvider()
        assert "XAI_API_KEY" in p.config.auth.api_key_env_vars
        assert "GROK_API_KEY" in p.config.auth.api_key_env_vars

    def test_default_model(self):
        p = XAIProvider()
        assert p.config.auth.default_model == "grok-4"

    def test_models_include_grok3(self):
        p = XAIProvider()
        model_ids = [m.id for m in p.config.models]
        assert "grok-3" in model_ids


class TestPerplexityProvider:
    def test_config_name(self):
        p = PerplexityProvider()
        assert p.config.name == "perplexity"

    def test_base_url(self):
        p = PerplexityProvider()
        assert p.config.auth.default_base_url == "https://api.perplexity.ai"

    def test_env_vars(self):
        p = PerplexityProvider()
        assert "PERPLEXITY_API_KEY" in p.config.auth.api_key_env_vars

    def test_default_model(self):
        p = PerplexityProvider()
        assert p.config.auth.default_model == "sonar-pro"


class TestDeepInfraProvider:
    def test_config_name(self):
        p = DeepInfraProvider()
        assert p.config.name == "deepinfra"

    def test_base_url(self):
        p = DeepInfraProvider()
        assert p.config.auth.default_base_url == "https://api.deepinfra.com/v1/openai"

    def test_env_vars(self):
        p = DeepInfraProvider()
        assert "DEEPINFRA_API_KEY" in p.config.auth.api_key_env_vars


class TestHuggingFaceProvider:
    def test_config_name(self):
        p = HuggingFaceProvider()
        assert p.config.name == "huggingface"

    def test_base_url(self):
        p = HuggingFaceProvider()
        assert p.config.auth.default_base_url == "https://api-inference.huggingface.co/v1"

    def test_env_vars(self):
        p = HuggingFaceProvider()
        assert "HF_API_KEY" in p.config.auth.api_key_env_vars
        assert "HUGGINGFACE_API_KEY" in p.config.auth.api_key_env_vars


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistryIncludesNewProviders:
    def test_registry_registers_all_20_providers(self):
        """The registry should now have 20 providers (15 original + 5 new)."""
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert len(registry._providers) == 20

    def test_registry_has_opencode(self):
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert "opencode" in registry._providers
        assert isinstance(registry._providers["opencode"], OpenCodeProvider)

    def test_registry_has_xai(self):
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert "xai" in registry._providers
        assert isinstance(registry._providers["xai"], XAIProvider)

    def test_registry_has_perplexity(self):
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert "perplexity" in registry._providers

    def test_registry_has_deepinfra(self):
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert "deepinfra" in registry._providers

    def test_registry_has_huggingface(self):
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        assert "huggingface" in registry._providers

    def test_all_new_providers_have_valid_configs(self):
        """Every new provider must have a non-empty name, label, and base_url."""
        registry = ProviderRegistry()
        registry._register_builtin_providers()
        for name in ("opencode", "xai", "perplexity", "deepinfra", "huggingface"):
            prov = registry._providers[name]
            cfg = prov.config
            assert cfg.name == name, f"Provider {name} has wrong config.name: {cfg.name}"
            assert cfg.label, f"Provider {name} has empty label"
            assert cfg.auth.default_base_url, f"Provider {name} has empty base_url"
            assert cfg.auth.default_model, f"Provider {name} has empty default_model"
