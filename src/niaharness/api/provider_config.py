"""Provider configuration for OpenAI-compatible APIs.

Ported from OpenClaude's providerConfig.ts with support for all major providers:
- OpenAI, Azure OpenAI, Ollama, OpenRouter, Groq, Together AI, DeepSeek,
- Fireworks, NVIDIA NIM, Cerebras, AWS Bedrock, Google Vertex, Mistral
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Provider-specific defaults
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

OLLAMA_DEFAULT_PORT = 11434
LOCALHOST_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


# ---------------------------------------------------------------------------
# Provider types
# ---------------------------------------------------------------------------

class ProviderTransport(Enum):
    """Transport type for provider requests."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI = "gemini"


class ReasoningEffort(Enum):
    """Reasoning effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass(frozen=True)
class ResolvedProviderRequest:
    """Resolved provider request configuration."""

    transport: ProviderTransport
    requested_model: str
    resolved_model: str
    base_url: str
    reasoning_effort: Optional[ReasoningEffort] = None


@dataclass(frozen=True)
class LocalFastPathConfig:
    """Configuration for local provider fast-path optimizations."""

    enabled: bool
    skip_stable_stringify: bool
    skip_strict_tools: bool
    skip_tool_history_compression: bool


LOCAL_FAST_PATH_OFF = LocalFastPathConfig(
    enabled=False,
    skip_stable_stringify=False,
    skip_strict_tools=False,
    skip_tool_history_compression=False,
)

LOCAL_FAST_PATH_ON = LocalFastPathConfig(
    enabled=True,
    skip_stable_stringify=True,
    skip_strict_tools=True,
    skip_tool_history_compression=True,
)


@dataclass
class ProviderConfig:
    """Complete provider configuration."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    provider_name: str = "openai"
    is_azure: bool = False
    is_local: bool = False
    is_gemini: bool = False
    is_mistral: bool = False
    is_groq: bool = False
    is_together: bool = False
    is_deepseek: bool = False
    is_fireworks: bool = False
    is_nvidia_nim: bool = False
    is_cerebras: bool = False
    is_openrouter: bool = False
    is_bedrock: bool = False
    is_vertex: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# URL / hostname helpers
# ---------------------------------------------------------------------------

def _is_private_ipv4(hostname: str) -> bool:
    """Check if an IPv4 address is private (RFC1918)."""
    try:
        octets = [int(part) for part in hostname.split(".")]
        if len(octets) != 4 or any(o < 0 or o > 255 for o in octets):
            return False
        return (
            octets[0] == 10
            or (octets[0] == 172 and 16 <= octets[1] <= 31)
            or (octets[0] == 192 and octets[1] == 168)
        )
    except (ValueError, IndexError):
        return False


def _is_private_ipv6(hostname: str) -> bool:
    """Check if an IPv6 address is private (ULA/LL)."""
    try:
        addr = ipaddress.ip_address(hostname)
        return isinstance(addr, ipaddress.IPv6Address) and (addr.is_link_local or addr.is_unique_local)
    except ValueError:
        return False


def is_local_provider_url(base_url: Optional[str]) -> bool:
    """Check if a base URL points to a local provider."""
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()

        # Strip IPv6 brackets
        if hostname.startswith("[") and hostname.endswith("]"):
            hostname = hostname[1:-1]

        # Strip RFC6874 zone identifiers
        zone_idx = hostname.find("%25")
        if zone_idx != -1:
            hostname = hostname[:zone_idx]

        if hostname in LOCALHOST_HOSTNAMES or hostname == "0.0.0.0":
            return True
        if hostname.endswith(".local"):
            return True

        try:
            addr = ipaddress.ip_address(hostname)
            if isinstance(addr, ipaddress.IPv4Address):
                return addr.is_loopback or _is_private_ipv4(hostname)
            if isinstance(addr, ipaddress.IPv6Address):
                return addr.is_loopback or _is_private_ipv6(hostname)
        except ValueError:
            pass

        return False
    except Exception:
        return False


def is_likely_ollama_endpoint(base_url: Optional[str]) -> bool:
    """Check if a URL is likely an Ollama endpoint."""
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
        port = parsed.port
        if port == OLLAMA_DEFAULT_PORT:
            return True
        hostname = (parsed.hostname or "").lower()
        pathname = (parsed.path or "").lower()
        return "ollama" in hostname or "ollama" in pathname
    except Exception:
        return False


def is_azure_endpoint(base_url: str) -> bool:
    """Check if a URL is an Azure OpenAI endpoint."""
    try:
        hostname = urlparse(base_url).hostname or ""
        hostname = hostname.lower()
        return (
            hostname.endswith(".azure.com")
            and ("cognitiveservices" in hostname or "openai" in hostname or "services.ai" in hostname)
        )
    except Exception:
        return False


def is_gemini_endpoint(base_url: str) -> bool:
    """Check if a URL is a Google Gemini/Vertex endpoint."""
    try:
        hostname = urlparse(base_url).hostname or ""
        return "generativelanguage.googleapis.com" in hostname.lower()
    except Exception:
        return False


def get_local_provider_retry_base_urls(base_url: str) -> list[str]:
    """Get alternative base URLs for local provider retry."""
    if not is_local_provider_url(base_url):
        return []

    try:
        parsed = urlparse(base_url)
        original = base_url.rstrip("/")
        seen = {original}
        candidates: list[str] = []

        # Try adding /v1 if missing
        path = parsed.path.rstrip("/")
        if not path or path == "/":
            new_url = f"{parsed.scheme}://{parsed.netloc}/v1"
            if new_url not in seen:
                seen.add(new_url)
                candidates.append(new_url)
        elif not path.endswith("/v1"):
            new_url = f"{parsed.scheme}://{parsed.netloc}{path}/v1"
            if new_url not in seen:
                seen.add(new_url)
                candidates.append(new_url)

        # Try localhost -> 127.0.0.1
        hostname = (parsed.hostname or "").lower()
        if hostname in ("localhost", "::1"):
            new_url = f"{parsed.scheme}://127.0.0.1:{parsed.port or ''}{parsed.path}"
            if new_url not in seen:
                seen.add(new_url)
                candidates.append(new_url)

        return candidates
    except Exception:
        return []


def should_attempt_local_toolless_retry(base_url: str, has_tools: bool) -> bool:
    """Check if we should retry without tools for local providers (Ollama)."""
    if not has_tools:
        return False
    if not is_local_provider_url(base_url):
        return False
    return is_likely_ollama_endpoint(base_url)


# ---------------------------------------------------------------------------
# Fast-path config
# ---------------------------------------------------------------------------

def get_local_fast_path_config(base_url: Optional[str]) -> LocalFastPathConfig:
    """Get fast-path configuration for local providers."""
    env_override = os.environ.get("OPENCLAUDE_LOCAL_FAST_PATH", "").strip().lower()
    if env_override in ("0", "false", "off", "no"):
        return LOCAL_FAST_PATH_OFF
    if env_override in ("1", "true", "on", "yes"):
        return LOCAL_FAST_PATH_ON
    if env_override in ("", "auto"):
        return LOCAL_FAST_PATH_ON if is_local_provider_url(base_url) else LOCAL_FAST_PATH_OFF
    return LOCAL_FAST_PATH_OFF if is_local_provider_url(base_url) else LOCAL_FAST_PATH_OFF


# ---------------------------------------------------------------------------
# Model descriptor parsing
# ---------------------------------------------------------------------------

def _parse_reasoning_effort(value: Optional[str]) -> Optional[ReasoningEffort]:
    """Parse a reasoning effort string."""
    if not value:
        return None
    normalized = value.strip().lower()
    try:
        return ReasoningEffort(normalized)
    except ValueError:
        return None


def parse_model_descriptor(model: str) -> tuple[str, Optional[ReasoningEffort]]:
    """Parse model string, extracting base model and optional reasoning effort.

    Supports formats:
        - "gpt-4o"
        - "gpt-4o?reasoning=high"
        - "deepseek-reasoner?reasoning=xhigh"
    """
    trimmed = model.strip()
    query_idx = trimmed.find("?")
    if query_idx == -1:
        return trimmed, None

    base_model = trimmed[:query_idx].strip()
    params = trimmed[query_idx + 1:]
    effort = None
    for param in params.split("&"):
        if param.startswith("reasoning="):
            effort = _parse_reasoning_effort(param[len("reasoning="):])
    return base_model, effort


# ---------------------------------------------------------------------------
# Provider request resolution
# ---------------------------------------------------------------------------

def resolve_provider_request(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> ResolvedProviderRequest:
    """Resolve a provider request from configuration.

    This function resolves the base URL, model, and transport type
    based on the provided parameters and environment variables.
    """
    # Determine base URL
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_OPENAI_BASE_URL
    )

    # Determine model
    resolved_model = (
        model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MISTRAL_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gpt-4o"
    )

    base_model, model_effort = parse_model_descriptor(resolved_model)
    effort = model_effort or _parse_reasoning_effort(reasoning_effort)

    # Determine transport type
    transport = ProviderTransport.CHAT_COMPLETIONS

    # Check for provider-specific transports
    if is_gemini_endpoint(resolved_base_url):
        transport = ProviderTransport.GEMINI

    return ResolvedProviderRequest(
        transport=transport,
        requested_model=resolved_model,
        resolved_model=base_model,
        base_url=resolved_base_url,
        reasoning_effort=effort,
    )


# ---------------------------------------------------------------------------
# Provider detection and configuration
# ---------------------------------------------------------------------------

def detect_provider_from_url(base_url: str) -> ProviderConfig:
    """Detect provider type from base URL and return appropriate configuration."""
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()

    api_key = os.environ.get("OPENAI_API_KEY")

    # Azure OpenAI
    if is_azure_endpoint(base_url):
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            provider_name="azure-openai",
            is_azure=True,
        )

    # Google Gemini
    if is_gemini_endpoint(base_url):
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            provider_name="gemini",
            is_gemini=True,
        )

    # Ollama (local)
    if is_likely_ollama_endpoint(base_url):
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            provider_name="ollama",
            is_local=True,
        )

    # AWS Bedrock
    if "bedrock" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            provider_name="bedrock",
            is_bedrock=True,
        )

    # Google Vertex AI
    if "vertex" in hostname or "aiplatform" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("VERTEX_MODEL", "gemini-2.0-flash"),
            provider_name="vertex",
            is_vertex=True,
        )

    # Mistral
    if "mistral" in hostname or os.environ.get("CLAUDE_CODE_USE_MISTRAL"):
        return ProviderConfig(
            base_url=base_url or DEFAULT_MISTRAL_BASE_URL,
            model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
            api_key=api_key or os.environ.get("MISTRAL_API_KEY"),
            provider_name="mistral",
            is_mistral=True,
        )

    # Groq
    if "groq" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            provider_name="groq",
            is_groq=True,
        )

    # Together AI
    if "together" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            provider_name="together",
            is_together=True,
        )

    # DeepSeek
    if "deepseek" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            provider_name="deepseek",
            is_deepseek=True,
        )

    # Fireworks
    if "fireworks" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY"),
            provider_name="fireworks",
            is_fireworks=True,
        )

    # NVIDIA NIM
    if "nvidia" in hostname or "nim" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-405b-instruct"),
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
            provider_name="nvidia-nim",
            is_nvidia_nim=True,
        )

    # Cerebras
    if "cerebras" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b"),
            api_key=api_key or os.environ.get("CEREBRAS_API_KEY"),
            provider_name="cerebras",
            is_cerebras=True,
        )

    # OpenRouter
    if "openrouter" in hostname:
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            provider_name="openrouter",
            is_openrouter=True,
        )

    # Local provider
    if is_local_provider_url(base_url):
        return ProviderConfig(
            base_url=base_url,
            model=os.environ.get("LOCAL_MODEL", "default"),
            provider_name="local",
            is_local=True,
        )

    # Default: generic OpenAI-compatible
    return ProviderConfig(
        base_url=base_url,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_key=api_key,
        provider_name="openai",
    )


def get_provider_config(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ProviderConfig:
    """Get complete provider configuration from environment and parameters."""
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_OPENAI_BASE_URL
    )

    config = detect_provider_from_url(resolved_base_url)

    # Override with explicit parameters
    if model:
        config.model = model
    if api_key:
        config.api_key = api_key

    return config
