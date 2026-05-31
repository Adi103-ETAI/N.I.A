"""Provider and model type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderStatus(Enum):
    """Provider connection status."""
    CONFIGURED = "configured"
    CONNECTED = "connected"
    ERROR = "error"
    UNKNOWN = "unknown"


class ModelCapability(Enum):
    """What a model can do."""
    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    STREAMING = "streaming"
    REASONING = "reasoning"


@dataclass
class ModelInfo:
    """Metadata about a specific model."""
    id: str
    name: str
    provider_id: str
    context_window: int = 8192
    max_output: int = 4096
    capabilities: list[ModelCapability] = field(default_factory=lambda: [ModelCapability.CHAT])
    cost_input: float | None = None   # per 1M tokens
    cost_output: float | None = None  # per 1M tokens

    @property
    def display_name(self) -> str:
        return f"{self.provider_id}/{self.id}"


@dataclass
class ProviderInfo:
    """Metadata about a provider."""
    id: str
    name: str
    description: str = ""
    status: ProviderStatus = ProviderStatus.UNKNOWN
    api_key_configured: bool = False
    base_url: str | None = None
    models: list[ModelInfo] = field(default_factory=list)
    auth_url: str | None = None  # OAuth URL if supported
    supports_oauth: bool = False
    supports_api_key: bool = True

    @property
    def model_count(self) -> int:
        return len(self.models)


@dataclass
class LLMRequest:
    """A request to an LLM provider."""
    model: str
    messages: list[dict[str, str]]
    system: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: list[dict[str, Any]] | None = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None  # For thinking models
