"""P1 Provider profiles — 22 missing providers with declarative profiles.

Ported from Hermes Agent's ``providers/base.py`` ProviderProfile pattern +
``plugins/model-providers/`` plugin registry. Each profile declares everything
about an inference provider in one place: auth, endpoints, client quirks,
fallback models, vision support.

22 new providers (6 OAuth + 16 API-key):

OAuth providers:
  - nous (Nous Research — Hermes model family, device-code OAuth)
  - openai-codex (OpenAI Codex — Responses API, device-code OAuth)
  - xai-oauth (xAI Grok — device-code OAuth)
  - qwen-oauth (Alibaba Qwen — device-code OAuth)
  - minimax-oauth (MiniMax — device-code OAuth)
  - copilot-acp (GitHub Copilot ACP — OAuth)

API-key providers:
  - lmstudio, copilot, zai, kimi-coding, kimi-coding-cn, stepfun, arcee,
    gmi, minimax, minimax-cn, alibaba, alibaba-coding-plan, opencode-zen,
    opencode-go, kilocode, xiaomi, tencent-tokenhub, ollama-cloud,
    azure-foundry, novita

Each profile carries: name, aliases, env_vars, base_url, models_url,
auth_type, display_name, description, signup_url, fallback_models,
supports_vision, default_headers, default_max_tokens, default_aux_model.

Usage::

    from niaharness.api.provider_profiles import (
        get_provider_profile, list_provider_profiles, register_provider_profile
    )

    profile = get_provider_profile("deepseek")
    if profile:
        print(f"{profile.display_name}: {profile.base_url}")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Sentinel for "omit temperature entirely" (Kimi: server manages it).
OMIT_TEMPERATURE = object()


def _profile_user_agent() -> str:
    """Return a nia-cli/<version> UA string."""
    try:
        from niaharness import __version__ as _ver
        return f"nia-cli/{_ver}"
    except Exception:
        return "nia-cli"


# ---------------------------------------------------------------------------
# ProviderProfile base class
# ---------------------------------------------------------------------------


@dataclass
class ProviderProfile:
    """Declarative provider profile — describes a provider's behavior.

    Provider profiles are DECLARATIVE — they describe the provider's
    auth, endpoints, and quirks. They do NOT own client construction,
    credential rotation, or streaming. Those stay on the API client.

    Attributes:
        name: Canonical provider name (e.g. "deepseek").
        api_mode: API mode — "chat_completions", "responses", "anthropic_messages".
        aliases: Alternative names for lookup (e.g. ("moonshot",) for kimi).
        display_name: Human-readable name (e.g. "DeepSeek").
        description: Short description for the picker.
        signup_url: URL for the signup page.
        env_vars: Tuple of env var names for the API key.
        base_url: Default API base URL.
        models_url: Explicit models endpoint (falls back to base_url + "/models").
        auth_type: "api_key", "oauth_device_code", "oauth_external", "copilot", "aws_sdk".
        supports_health_check: True if doctor should probe /models.
        supports_vision: True if the provider accepts image content.
        supports_vision_tool_messages: True if tool messages can contain images.
        fallback_models: Curated model list for the picker when live fetch fails.
        hostname: Base hostname for URL→provider reverse-mapping.
        default_headers: Extra HTTP headers for all requests.
        fixed_temperature: None = use caller's default, OMIT_TEMPERATURE = don't send.
        default_max_tokens: Default max_tokens cap.
        default_aux_model: Cheap model for auxiliary tasks.
    """

    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    supports_health_check: bool = True
    supports_vision: bool = False
    supports_vision_tool_messages: bool = True
    fallback_models: tuple = ()
    hostname: str = ""
    default_headers: dict[str, str] = field(default_factory=dict)
    fixed_temperature: Any = None
    default_max_tokens: Optional[int] = None
    default_aux_model: str = ""

    def get_hostname(self) -> str:
        """Return the provider's base hostname for URL-based detection."""
        if self.hostname:
            return self.hostname
        if self.base_url:
            return urlparse(self.base_url).hostname or ""
        return ""

    def get_max_tokens(self, model: Optional[str]) -> Optional[int]:
        """Return the default max_tokens cap for *model*."""
        return self.default_max_tokens

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[list[str]]:
        """Fetch the live model list from the provider's models endpoint.

        Returns a list of model ID strings, or None if the fetch failed.
        """
        effective_base = base_url or self.base_url
        url = (self.models_url or "").strip()
        if not url:
            if not effective_base:
                return None
            url = effective_base.rstrip("/") + "/models"

        try:
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", _profile_user_agent())
            for k, v in self.default_headers.items():
                req.add_header(k, v)

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}


def register_provider_profile(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases."""
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias.lower()] = profile.name


def get_provider_profile(name: str) -> Optional[ProviderProfile]:
    """Look up a provider profile by name or alias."""
    canonical = _ALIASES.get(name.lower(), name.lower())
    return _REGISTRY.get(canonical)


def list_provider_profiles() -> List[ProviderProfile]:
    """Return all registered provider profiles."""
    return list(_REGISTRY.values())


def list_provider_names() -> List[str]:
    """Return all registered provider names (sorted)."""
    return sorted(_REGISTRY.keys())


def detect_provider_from_env() -> Optional[ProviderProfile]:
    """Auto-detect the provider from environment variables.

    Checks each registered provider's env_vars against os.environ.
    Returns the first match, or None.
    """
    for profile in _REGISTRY.values():
        for env_var in profile.env_vars:
            if os.environ.get(env_var, "").strip():
                return profile
    return None


def detect_provider_from_url(url: str) -> Optional[ProviderProfile]:
    """Auto-detect the provider from a base URL.

    Matches the URL's hostname against each provider's hostname.
    """
    if not url:
        return None
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        return None
    for profile in _REGISTRY.values():
        if profile.get_hostname() == hostname:
            return profile
    # Substring match fallback.
    for profile in _REGISTRY.values():
        if profile.get_hostname() and profile.get_hostname() in hostname:
            return profile
    return None


# ---------------------------------------------------------------------------
# OAuth providers (6)
# ---------------------------------------------------------------------------


# 1. Nous Research — Hermes model family, device-code OAuth
register_provider_profile(ProviderProfile(
    name="nous",
    aliases=("nous-portal", "nousresearch"),
    display_name="Nous Research",
    description="Nous Research — Hermes model family",
    signup_url="https://nousresearch.com/",
    env_vars=("NOUS_API_KEY",),
    base_url="https://inference.nousresearch.com/v1",
    models_url="https://inference.nousresearch.com/v1/models",
    auth_type="oauth_device_code",
    fallback_models=("hermes-3-405b", "hermes-3-70b"),
    hostname="inference.nousresearch.com",
    default_aux_model="hermes-3-70b",
))

# 2. OpenAI Codex — Responses API, device-code OAuth
register_provider_profile(ProviderProfile(
    name="openai-codex",
    aliases=("codex",),
    display_name="OpenAI Codex",
    description="OpenAI Codex — Responses API for agentic coding",
    signup_url="https://platform.openai.com/",
    env_vars=("OPENAI_CODEX_API_KEY",),
    base_url="https://api.openai.com/v1",
    models_url="https://api.openai.com/v1/models",
    auth_type="oauth_device_code",
    api_mode="responses",
    fallback_models=("codex-mini-latest", "o3-mini", "o4-mini"),
    hostname="api.openai.com",
    supports_vision=True,
))

# 3. xAI OAuth — Grok models, device-code OAuth
register_provider_profile(ProviderProfile(
    name="xai-oauth",
    aliases=("grok-oauth", "x-ai-oauth"),
    display_name="xAI (OAuth)",
    description="xAI Grok — OAuth flow for Grok-3/4 models",
    signup_url="https://x.ai/",
    env_vars=("XAI_OAUTH_TOKEN",),
    base_url="https://api.x.ai/v1",
    models_url="https://api.x.ai/v1/models",
    auth_type="oauth_device_code",
    fallback_models=("grok-3", "grok-3-mini", "grok-4", "grok-4-fast"),
    hostname="api.x.ai",
    supports_vision=True,
))

# 4. Qwen OAuth — Alibaba Qwen, device-code OAuth
register_provider_profile(ProviderProfile(
    name="qwen-oauth",
    aliases=("qwen", "tongyi"),
    display_name="Qwen (OAuth)",
    description="Alibaba Qwen — Tongyi Qianwen models via OAuth",
    signup_url="https://dashscope.aliyun.com/",
    env_vars=("QWEN_OAUTH_TOKEN",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    models_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
    auth_type="oauth_device_code",
    fallback_models=("qwen-max", "qwen-plus", "qwen-turbo", "qwen-long"),
    hostname="dashscope-intl.aliyuncs.com",
    supports_vision=True,
))

# 5. MiniMax OAuth — device-code OAuth
register_provider_profile(ProviderProfile(
    name="minimax-oauth",
    aliases=("minimax-oauth-flow",),
    display_name="MiniMax (OAuth)",
    description="MiniMax — OAuth flow for abab models",
    signup_url="https://www.minimax.io/",
    env_vars=("MINIMAX_OAUTH_TOKEN",),
    base_url="https://api.minimax.io/v1",
    models_url="https://api.minimax.io/v1/models",
    auth_type="oauth_device_code",
    fallback_models=("abab6.5s-chat", "abab6.5g-chat", "abab6.5t-chat"),
    hostname="api.minimax.io",
))

# 6. Copilot ACP — GitHub Copilot via ACP, OAuth
register_provider_profile(ProviderProfile(
    name="copilot-acp",
    aliases=("github-copilot-acp",),
    display_name="GitHub Copilot (ACP)",
    description="GitHub Copilot — ACP protocol for agentic coding",
    signup_url="https://github.com/features/copilot",
    env_vars=("COPILOT_ACP_TOKEN",),
    base_url="https://api.githubcopilot.com",
    models_url="https://api.githubcopilot.com/models",
    auth_type="oauth_external",
    fallback_models=("gpt-4o", "claude-3.5-sonnet", "o3-mini"),
    hostname="api.githubcopilot.com",
    supports_vision=True,
))


# ---------------------------------------------------------------------------
# API-key providers (16+)
# ---------------------------------------------------------------------------

# 7. LM Studio — local OpenAI-compatible server
register_provider_profile(ProviderProfile(
    name="lmstudio",
    aliases=("lm-studio",),
    display_name="LM Studio",
    description="LM Studio — local model server (OpenAI-compatible)",
    signup_url="https://lmstudio.ai/",
    env_vars=("LMSTUDIO_API_KEY",),
    base_url="http://localhost:1234/v1",
    models_url="http://localhost:1234/v1/models",
    auth_type="api_key",
    supports_health_check=False,  # Local — no health check
    fallback_models=("local-model",),
    hostname="localhost",
))

# 8. Copilot — GitHub Copilot (API key mode)
register_provider_profile(ProviderProfile(
    name="copilot",
    aliases=("github-copilot",),
    display_name="GitHub Copilot",
    description="GitHub Copilot — API key mode",
    signup_url="https://github.com/features/copilot",
    env_vars=("COPILOT_API_KEY", "GITHUB_COPILOT_TOKEN"),
    base_url="https://api.githubcopilot.com",
    models_url="https://api.githubcopilot.com/models",
    auth_type="api_key",
    fallback_models=("gpt-4o", "claude-3.5-sonnet", "o3-mini"),
    hostname="api.githubcopilot.com",
    supports_vision=True,
    default_headers={"Editor-Version": "vscode/1.85.0"},
))

# 9. Z.AI / GLM
register_provider_profile(ProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai"),
    display_name="Z.AI / GLM",
    description="Z.AI — GLM model family (Claude-compatible API)",
    signup_url="https://z.ai/",
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4",
    models_url="https://api.z.ai/api/paas/v4/models",
    auth_type="api_key",
    fallback_models=("glm-4-plus", "glm-4-air", "glm-4-flash", "glm-4-long"),
    hostname="api.z.ai",
    supports_vision=True,
))

# 10. Kimi / Moonshot
register_provider_profile(ProviderProfile(
    name="kimi-coding",
    aliases=("kimi", "moonshot"),
    display_name="Kimi / Moonshot",
    description="Kimi (Moonshot AI) — long-context models",
    signup_url="https://platform.moonshot.ai/",
    env_vars=("KIMI_API_KEY",),
    base_url="https://api.moonshot.ai/v1",
    models_url="https://api.moonshot.ai/v1/models",
    auth_type="api_key",
    fallback_models=("moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k", "kimi-latest"),
    hostname="api.moonshot.ai",
    fixed_temperature=OMIT_TEMPERATURE,  # Kimi: server manages temperature
))

# 11. Kimi / Moonshot (China)
register_provider_profile(ProviderProfile(
    name="kimi-coding-cn",
    aliases=("kimi-cn", "moonshot-cn"),
    display_name="Kimi / Moonshot (China)",
    description="Kimi (Moonshot AI) — China endpoint",
    signup_url="https://platform.moonshot.cn/",
    env_vars=("KIMI_CN_API_KEY",),
    base_url="https://api.moonshot.cn/v1",
    models_url="https://api.moonshot.cn/v1/models",
    auth_type="api_key",
    supports_health_check=False,  # CN endpoint doesn't support /models
    fallback_models=("moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"),
    hostname="api.moonshot.cn",
    fixed_temperature=OMIT_TEMPERATURE,
))

# 12. StepFun
register_provider_profile(ProviderProfile(
    name="stepfun",
    aliases=("step-plan", "stepfun-ai"),
    display_name="StepFun",
    description="StepFun — Step Plan model family",
    signup_url="https://www.stepfun.ai/",
    env_vars=("STEPFUN_API_KEY",),
    base_url="https://api.stepfun.ai/step_plan/v1",
    models_url="https://api.stepfun.ai/step_plan/v1/models",
    auth_type="api_key",
    fallback_models=("step-1-8k", "step-1-32k", "step-1-128k", "step-1v-32k"),
    hostname="api.stepfun.ai",
    supports_vision=True,
))

# 13. Arcee AI
register_provider_profile(ProviderProfile(
    name="arcee",
    aliases=("arcee-ai",),
    display_name="Arcee AI",
    description="Arcee AI — model merging and fine-tuning platform",
    signup_url="https://arcee.ai/",
    env_vars=("ARCEEAI_API_KEY",),
    base_url="https://api.arcee.ai/api/v1",
    models_url="https://api.arcee.ai/api/v1/models",
    auth_type="api_key",
    fallback_models=("arcee-blend", "maestro-researcher"),
    hostname="api.arcee.ai",
))

# 14. GMI Cloud
register_provider_profile(ProviderProfile(
    name="gmi",
    aliases=("gmi-cloud", "gmicloud"),
    display_name="GMI Cloud",
    description="GMI Cloud — multi-model direct API",
    signup_url="https://www.gmicloud.ai/",
    env_vars=("GMI_API_KEY",),
    base_url="https://api.gmi-serving.com/v1",
    models_url="https://api.gmi-serving.com/v1/models",
    auth_type="api_key",
    fallback_models=("gmi-llama-3.1-405b", "gmi-llama-3.1-70b"),
    hostname="api.gmi-serving.com",
))

# 15. MiniMax (API key)
register_provider_profile(ProviderProfile(
    name="minimax",
    display_name="MiniMax",
    description="MiniMax — abab model family (API key)",
    signup_url="https://www.minimax.io/",
    env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/v1",
    models_url="https://api.minimax.io/v1/models",
    auth_type="api_key",
    fallback_models=("abab6.5s-chat", "abab6.5g-chat", "abab6.5t-chat"),
    hostname="api.minimax.io",
))

# 16. MiniMax (China, API key)
register_provider_profile(ProviderProfile(
    name="minimax-cn",
    aliases=("minimax-china",),
    display_name="MiniMax (China)",
    description="MiniMax — China endpoint (API key)",
    signup_url="https://www.minimaxi.com/",
    env_vars=("MINIMAX_CN_API_KEY",),
    base_url="https://api.minimaxi.com/v1",
    models_url="https://api.minimaxi.com/v1/models",
    auth_type="api_key",
    supports_health_check=False,  # CN endpoint doesn't support /models
    fallback_models=("abab6.5s-chat", "abab6.5g-chat", "abab6.5t-chat"),
    hostname="api.minimaxi.com",
))

# 17. Alibaba / DashScope
register_provider_profile(ProviderProfile(
    name="alibaba",
    aliases=("dashscope",),
    display_name="Alibaba / DashScope",
    description="Alibaba DashScope — Qwen model family",
    signup_url="https://dashscope.aliyun.com/",
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    models_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
    auth_type="api_key",
    fallback_models=("qwen-max", "qwen-plus", "qwen-turbo", "qwen-long"),
    hostname="dashscope-intl.aliyuncs.com",
    supports_vision=True,
))

# 18. Alibaba Coding Plan
register_provider_profile(ProviderProfile(
    name="alibaba-coding-plan",
    aliases=("dashscope-coding",),
    display_name="Alibaba (Coding Plan)",
    description="Alibaba DashScope — Coding Plan endpoint",
    signup_url="https://dashscope.aliyun.com/",
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    models_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
    auth_type="api_key",
    fallback_models=("qwen-coder-plus", "qwen-coder-turbo"),
    hostname="dashscope-intl.aliyuncs.com",
))

# 19. OpenCode Zen
register_provider_profile(ProviderProfile(
    name="opencode-zen",
    aliases=("opencode", "zen"),
    display_name="OpenCode Zen",
    description="OpenCode Zen — agentic coding models",
    signup_url="https://opencode.ai/",
    env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
    models_url="https://opencode.ai/zen/v1/models",
    auth_type="api_key",
    fallback_models=("zen-1", "zen-1-mini"),
    hostname="opencode.ai",
))

# 20. OpenCode Go
register_provider_profile(ProviderProfile(
    name="opencode-go",
    aliases=("opencode-golang",),
    display_name="OpenCode Go",
    description="OpenCode Go — Go-based agentic models",
    signup_url="https://opencode.ai/",
    env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/go/v1",
    auth_type="api_key",
    supports_health_check=False,  # No shared /models endpoint
    fallback_models=("go-1", "go-1-mini"),
    hostname="opencode.ai",
))

# 21. Kilo Code
register_provider_profile(ProviderProfile(
    name="kilocode",
    aliases=("kilo", "kilo-code"),
    display_name="Kilo Code",
    description="Kilo Code — gateway for multiple model providers",
    signup_url="https://kilo.ai/",
    env_vars=("KILOCODE_API_KEY",),
    base_url="https://api.kilo.ai/api/gateway",
    models_url="https://api.kilo.ai/api/gateway/models",
    auth_type="api_key",
    fallback_models=("anthropic/claude-3.5-sonnet", "openai/gpt-4o"),
    hostname="api.kilo.ai",
))

# 22. Xiaomi
register_provider_profile(ProviderProfile(
    name="xiaomi",
    aliases=("mimo",),
    display_name="Xiaomi",
    description="Xiaomi MiMo — local language models",
    signup_url="https://www.mi.com/",
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomi.com/v1",
    models_url="https://api.xiaomi.com/v1/models",
    auth_type="api_key",
    fallback_models=("mimo-7b", "mimo-13b"),
    hostname="api.xiaomi.com",
    supports_vision_tool_messages=False,  # Xiaomi rejects list-type tool content
))

# 23. Tencent TokenHub
register_provider_profile(ProviderProfile(
    name="tencent-tokenhub",
    aliases=("tencent", "tokenhub"),
    display_name="Tencent TokenHub",
    description="Tencent TokenHub — multi-model API gateway",
    signup_url="https://cloud.tencent.com/",
    env_vars=("TENCENT_TOKENHUB_API_KEY",),
    base_url="https://api.tencent.com/tokenhub/v1",
    models_url="https://api.tencent.com/tokenhub/v1/models",
    auth_type="api_key",
    fallback_models=("hunyuan-pro", "hunyuan-standard", "hunyuan-lite"),
    hostname="api.tencent.com",
))

# 24. Ollama Cloud
register_provider_profile(ProviderProfile(
    name="ollama-cloud",
    aliases=("ollama",),
    display_name="Ollama Cloud",
    description="Ollama Cloud — managed Ollama instances",
    signup_url="https://ollama.ai/",
    env_vars=("OLLAMA_CLOUD_API_KEY",),
    base_url="https://api.ollama.ai/v1",
    models_url="https://api.ollama.ai/v1/models",
    auth_type="api_key",
    fallback_models=("llama3.3:70b", "qwen2.5:72b", "deepseek-r1:70b"),
    hostname="api.ollama.ai",
))

# 25. Azure Foundry
register_provider_profile(ProviderProfile(
    name="azure-foundry",
    aliases=("azure", "azure-openai"),
    display_name="Azure Foundry",
    description="Azure AI Foundry — managed OpenAI models on Azure",
    signup_url="https://azure.microsoft.com/en-us/products/ai-foundry",
    env_vars=("AZURE_OPENAI_API_KEY",),
    base_url="",  # User-specific endpoint
    auth_type="api_key",
    supports_health_check=False,  # Azure requires endpoint-specific URLs
    fallback_models=("gpt-4o", "gpt-4o-mini", "o3-mini"),
    hostname="",
    default_headers={"api-key": ""},  # Azure uses api-key header, not Bearer
))

# 26. Novita
register_provider_profile(ProviderProfile(
    name="novita",
    aliases=("novita-ai",),
    display_name="Novita AI",
    description="Novita AI — affordable GPU-backed model API",
    signup_url="https://novita.ai/",
    env_vars=("NOVITA_API_KEY",),
    base_url="https://api.novita.ai/v3/openai",
    models_url="https://api.novita.ai/v3/openai/models",
    auth_type="api_key",
    fallback_models=("meta-llama/llama-3.1-405b-instruct", "qwen/qwen-2.5-72b-instruct"),
    hostname="api.novita.ai",
))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_provider_summary() -> dict[str, Any]:
    """Return a summary dict of all registered providers."""
    return {
        "total": len(_REGISTRY),
        "oauth": [p.name for p in _REGISTRY.values() if p.auth_type.startswith("oauth")],
        "api_key": [p.name for p in _REGISTRY.values() if p.auth_type == "api_key"],
        "names": list_provider_names(),
    }


__all__ = [
    "OMIT_TEMPERATURE",
    "ProviderProfile",
    "detect_provider_from_env",
    "detect_provider_from_url",
    "get_provider_profile",
    "get_provider_summary",
    "list_provider_names",
    "list_provider_profiles",
    "register_provider_profile",
]
