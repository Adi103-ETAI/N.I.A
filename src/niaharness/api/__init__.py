"""API exports for NiaHarness.

Provides:
- AnthropicApiClient: Direct Anthropic SDK client with retry logic
- OpenAICompatibleClient: OpenAI-compatible provider client
- OpenAI shim utilities for message/tool conversion
- Provider configuration and detection
- Comprehensive error types
"""

from niaharness.api.client import AnthropicApiClient
from niaharness.api.errors import (
    AuthenticationFailure,
    ConnectionFailure,
    ContextOverflowFailure,
    ModelNotFoundFailure,
    NiaHarnessApiError,
    ProviderUnavailableFailure,
    RateLimitFailure,
    RequestFailure,
    ToolCallIncompatibleFailure,
)
from niaharness.api.openai_client import OpenAICompatibleClient
from niaharness.api.openai_shim import (
    OpenAIMessage,
    OpenAITool,
    AnthropicStreamEvent,
    convert_messages,
    convert_tools,
    openai_stream_to_anthropic,
    gemini_sse_to_anthropic,
    convert_non_streaming_response,
)
from niaharness.api.provider import ProviderInfo, auth_status, detect_provider
from niaharness.api.provider_config import (
    ProviderConfig,
    ProviderTransport,
    ReasoningEffort,
    ResolvedProviderRequest,
    detect_provider_from_url,
    get_provider_config,
    get_local_fast_path_config,
    is_local_provider_url,
    is_likely_ollama_endpoint,
    resolve_provider_request,
)
from niaharness.api.usage import UsageSnapshot

__all__ = [
    # Clients
    "AnthropicApiClient",
    "OpenAICompatibleClient",
    # OpenAI shim
    "OpenAIMessage",
    "OpenAITool",
    "AnthropicStreamEvent",
    "convert_messages",
    "convert_tools",
    "openai_stream_to_anthropic",
    "gemini_sse_to_anthropic",
    "convert_non_streaming_response",
    # Provider config
    "ProviderConfig",
    "ProviderInfo",
    "ProviderTransport",
    "ReasoningEffort",
    "ResolvedProviderRequest",
    "auth_status",
    "detect_provider",
    "detect_provider_from_url",
    "get_provider_config",
    "get_local_fast_path_config",
    "is_local_provider_url",
    "is_likely_ollama_endpoint",
    "resolve_provider_request",
    # Errors
    "AuthenticationFailure",
    "ConnectionFailure",
    "ContextOverflowFailure",
    "ModelNotFoundFailure",
    "NiaHarnessApiError",
    "ProviderUnavailableFailure",
    "RateLimitFailure",
    "RequestFailure",
    "ToolCallIncompatibleFailure",
    # Usage
    "UsageSnapshot",
]
