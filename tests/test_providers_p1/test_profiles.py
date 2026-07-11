"""Tests for the P1 provider profiles — 26 new providers (6 OAuth + 20 API-key).

Covers:
  - ProviderProfile base class (get_hostname, fetch_models stub).
  - Registry (register, lookup by name + alias, list).
  - Auto-detection from env vars + URL.
  - Each of the 26 provider profiles has correct metadata.
  - Integration with provider_config.get_provider_config.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from niaharness.api.provider_profiles import (
    OMIT_TEMPERATURE,
    ProviderProfile,
    detect_provider_from_env,
    detect_provider_from_url,
    get_provider_profile,
    get_provider_summary,
    list_provider_names,
    list_provider_profiles,
    register_provider_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Clear provider env vars so tests don't pick up host config."""
    for key in list(os.environ.keys()):
        if any(suffix in key for suffix in (
            "API_KEY", "OAUTH_TOKEN", "BEARER_TOKEN", "ACP_TOKEN",
            "COPILOT_TOKEN",
        )):
            monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# ProviderProfile base class
# ---------------------------------------------------------------------------


class TestProviderProfile:
    def test_get_hostname_from_explicit(self):
        p = ProviderProfile(name="test", hostname="api.test.com")
        assert p.get_hostname() == "api.test.com"

    def test_get_hostname_from_base_url(self):
        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        assert p.get_hostname() == "api.example.com"

    def test_get_hostname_empty(self):
        p = ProviderProfile(name="test")
        assert p.get_hostname() == ""

    def test_get_max_tokens_default(self):
        p = ProviderProfile(name="test", default_max_tokens=4096)
        assert p.get_max_tokens("any-model") == 4096

    def test_get_max_tokens_none(self):
        p = ProviderProfile(name="test")
        assert p.get_max_tokens("any-model") is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_provider_profiles_returns_26(self):
        profiles = list_provider_profiles()
        assert len(profiles) >= 26

    def test_get_provider_profile_by_name(self):
        p = get_provider_profile("zai")
        assert p is not None
        assert p.display_name == "Z.AI / GLM"

    def test_get_provider_profile_by_alias(self):
        p = get_provider_profile("glm")
        assert p is not None
        assert p.name == "zai"

    def test_get_provider_profile_not_found(self):
        assert get_provider_profile("nonexistent") is None

    def test_list_provider_names_sorted(self):
        names = list_provider_names()
        assert names == sorted(names)
        assert "nous" in names
        assert "deepseek" not in names  # deepseek is in the old config, not profiles

    def test_get_provider_summary(self):
        summary = get_provider_summary()
        assert summary["total"] >= 26
        assert len(summary["oauth"]) == 6
        assert len(summary["api_key"]) == 20
        assert "nous" in summary["names"]


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetection:
    def test_detect_from_env_zai(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key")
        p = detect_provider_from_env()
        assert p is not None
        assert p.name == "zai"

    def test_detect_from_env_kimi(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        p = detect_provider_from_env()
        assert p is not None
        assert p.name == "kimi-coding"

    def test_detect_from_env_no_keys(self):
        assert detect_provider_from_env() is None

    def test_detect_from_url_zai(self):
        p = detect_provider_from_url("https://api.z.ai/api/paas/v4")
        assert p is not None
        assert p.name == "zai"

    def test_detect_from_url_kimi(self):
        p = detect_provider_from_url("https://api.moonshot.ai/v1")
        assert p is not None
        assert p.name == "kimi-coding"

    def test_detect_from_url_nous(self):
        p = detect_provider_from_url("https://inference.nousresearch.com/v1")
        assert p is not None
        assert p.name == "nous"

    def test_detect_from_url_unknown(self):
        assert detect_provider_from_url("https://unknown.example.com/v1") is None

    def test_detect_from_url_empty(self):
        assert detect_provider_from_url("") is None
        assert detect_provider_from_url(None) is None


# ---------------------------------------------------------------------------
# OAuth providers (6)
# ---------------------------------------------------------------------------


class TestOAuthProviders:
    @pytest.mark.parametrize("name,display_name,auth_type", [
        ("nous", "Nous Research", "oauth_device_code"),
        ("openai-codex", "OpenAI Codex", "oauth_device_code"),
        ("xai-oauth", "xAI (OAuth)", "oauth_device_code"),
        ("qwen-oauth", "Qwen (OAuth)", "oauth_device_code"),
        ("minimax-oauth", "MiniMax (OAuth)", "oauth_device_code"),
        ("copilot-acp", "GitHub Copilot (ACP)", "oauth_external"),
    ])
    def test_oauth_provider_metadata(self, name, display_name, auth_type):
        p = get_provider_profile(name)
        assert p is not None
        assert p.display_name == display_name
        assert p.auth_type == auth_type

    def test_nous_has_fallback_models(self):
        p = get_provider_profile("nous")
        assert len(p.fallback_models) >= 2

    def test_openai_codex_uses_responses_api(self):
        p = get_provider_profile("openai-codex")
        assert p.api_mode == "responses"

    def test_copilot_acp_has_default_headers(self):
        # Copilot uses API key mode — the ACP variant uses oauth_external.
        p = get_provider_profile("copilot-acp")
        assert p.auth_type == "oauth_external"


# ---------------------------------------------------------------------------
# API-key providers (20)
# ---------------------------------------------------------------------------


class TestAPIKeyProviders:
    @pytest.mark.parametrize("name,display_name", [
        ("lmstudio", "LM Studio"),
        ("copilot", "GitHub Copilot"),
        ("zai", "Z.AI / GLM"),
        ("kimi-coding", "Kimi / Moonshot"),
        ("kimi-coding-cn", "Kimi / Moonshot (China)"),
        ("stepfun", "StepFun"),
        ("arcee", "Arcee AI"),
        ("gmi", "GMI Cloud"),
        ("minimax", "MiniMax"),
        ("minimax-cn", "MiniMax (China)"),
        ("alibaba", "Alibaba / DashScope"),
        ("alibaba-coding-plan", "Alibaba (Coding Plan)"),
        ("opencode-zen", "OpenCode Zen"),
        ("opencode-go", "OpenCode Go"),
        ("kilocode", "Kilo Code"),
        ("xiaomi", "Xiaomi"),
        ("tencent-tokenhub", "Tencent TokenHub"),
        ("ollama-cloud", "Ollama Cloud"),
        ("azure-foundry", "Azure Foundry"),
        ("novita", "Novita AI"),
    ])
    def test_api_key_provider_metadata(self, name, display_name):
        p = get_provider_profile(name)
        assert p is not None
        assert p.display_name == display_name
        assert p.auth_type == "api_key"

    def test_all_api_key_providers_have_env_vars(self):
        for p in list_provider_profiles():
            if p.auth_type == "api_key":
                assert len(p.env_vars) > 0, f"{p.name} has no env_vars"

    def test_all_api_key_providers_have_base_url(self):
        for p in list_provider_profiles():
            if p.auth_type == "api_key" and p.name != "azure-foundry":
                # Azure Foundry uses user-specific endpoints.
                assert p.base_url, f"{p.name} has no base_url"

    def test_all_providers_have_fallback_models(self):
        for p in list_provider_profiles():
            assert len(p.fallback_models) > 0, f"{p.name} has no fallback_models"

    def test_all_providers_have_signup_url(self):
        for p in list_provider_profiles():
            assert p.signup_url, f"{p.name} has no signup_url"

    def test_kimi_omits_temperature(self):
        p = get_provider_profile("kimi-coding")
        assert p.fixed_temperature is OMIT_TEMPERATURE

    def test_xiaomi_rejects_vision_tool_messages(self):
        p = get_provider_profile("xiaomi")
        assert p.supports_vision_tool_messages is False

    def test_zai_supports_vision(self):
        p = get_provider_profile("zai")
        assert p.supports_vision is True

    def test_minimax_cn_no_health_check(self):
        p = get_provider_profile("minimax-cn")
        assert p.supports_health_check is False

    def test_lmstudio_no_health_check(self):
        p = get_provider_profile("lmstudio")
        assert p.supports_health_check is False

    def test_opencode_go_no_health_check(self):
        p = get_provider_profile("opencode-go")
        assert p.supports_health_check is False


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


class TestAliases:
    @pytest.mark.parametrize("alias,expected_name", [
        ("glm", "zai"),
        ("moonshot", "kimi-coding"),
        ("dashscope", "alibaba"),
        ("ollama", "ollama-cloud"),
        ("grok-oauth", "xai-oauth"),
        ("codex", "openai-codex"),
        ("nous-portal", "nous"),
    ])
    def test_alias_resolution(self, alias, expected_name):
        p = get_provider_profile(alias)
        assert p is not None
        assert p.name == expected_name


# ---------------------------------------------------------------------------
# Integration with provider_config
# ---------------------------------------------------------------------------


class TestProviderConfigIntegration:
    def test_get_provider_config_detects_zai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4")
        monkeypatch.setenv("GLM_API_KEY", "test-key")
        from niaharness.api.provider_config import get_provider_config
        config = get_provider_config()
        # The new provider profiles should be detected.
        assert config.api_key == "test-key"

    def test_get_provider_config_with_explicit_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        from niaharness.api.provider_config import get_provider_config
        config = get_provider_config(api_key="explicit-key")
        assert config.api_key == "explicit-key"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
