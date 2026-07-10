"""Tests for the P1 auxiliary extensions.

Covers:
  - call_llm universal entry (no client, with client, per-task defaults)
  - complete_with_tools (Anthropic + OpenAI response parsing)
  - complete_stream (Anthropic + OpenAI streaming)
  - get_available_vision_backends / resolve_vision_provider_client
  - shutdown_cached_clients / cleanup_stale_async_clients
  - set_runtime_main / get_runtime_main / reset_runtime_main
  - refresh_credentials_for_client
  - detect_additional_providers
  - get_per_task_defaults
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.auxiliary import (
    AuxConfig,
    AuxiliaryClient,
)
from niaharness.auxiliary.extensions import (
    ADDITIONAL_PROVIDER_ENV_VARS,
    VisionBackend,
    call_llm,
    cleanup_stale_async_clients,
    complete_stream,
    complete_with_tools,
    detect_additional_providers,
    get_available_vision_backends,
    get_cached_client_count,
    get_per_task_defaults,
    get_runtime_main,
    refresh_credentials_for_client,
    reset_runtime_main,
    resolve_vision_provider_client,
    set_runtime_main,
    shutdown_cached_clients,
)
import niaharness.auxiliary.extensions as ext_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset all module-level state between tests."""
    # Clear the client cache.
    ext_mod._client_cache.clear()
    # Reset the runtime main.
    reset_runtime_main()
    yield
    # Cleanup after.
    ext_mod._client_cache.clear()
    reset_runtime_main()


@pytest.fixture(autouse=True)
def _clear_aux_env(monkeypatch):
    """Clear aux-related env vars so tests don't pick up the host config."""
    for key in list(os.environ.keys()):
        if key.startswith("NIA_AUX") or key in {
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
            "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
            "XAI_API_KEY", "NOUS_API_KEY", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
            "TOGETHER_API_KEY", "FIREWORKS_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# Fake SDK clients
# ---------------------------------------------------------------------------


class FakeAnthropicBlock:
    def __init__(self, btype: str, **kwargs):
        self.type = btype
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeAnthropicResponse:
    def __init__(self, content: List[Any], stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeAnthropicTextStream:
    """Fake async iterator over text chunks (matches real SDK's .text_stream attribute)."""

    def __init__(self, chunks: List[str]):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class FakeAnthropicStream:
    """Fake async context manager that yields text chunks.

    Matches the real Anthropic SDK's MessageStream interface: .stream(...)
    returns an async context manager whose .text_stream is an async
    iterator (attribute, not method).
    """

    def __init__(self, chunks: List[str]):
        self._text_stream = FakeAnthropicTextStream(chunks)

    @property
    def text_stream(self):
        return self._text_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeAnthropicMessages:
    def __init__(self):
        self.create = AsyncMock()
        self.stream = MagicMock()

    def configure(self, response: FakeAnthropicResponse):
        self.create = AsyncMock(return_value=response)

    def configure_stream(self, chunks: List[str]):
        self.stream = MagicMock(return_value=FakeAnthropicStream(chunks))


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()
        self.close = AsyncMock()


class FakeOpenAIFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeOpenAIToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = FakeOpenAIFunction(name, arguments)


class FakeOpenAIMessage:
    def __init__(self, content: str = "", tool_calls: List[FakeOpenAIToolCall] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeOpenAIChoice:
    def __init__(self, message: FakeOpenAIMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeOpenAIResponse:
    def __init__(self, choice: FakeOpenAIChoice):
        self.choices = [choice]


class FakeOpenAIStreamChunk:
    def __init__(self, content: str = ""):
        delta = MagicMock()
        delta.content = content
        self.choices = [MagicMock(delta=delta)]


class FakeOpenAIChatCompletions:
    def __init__(self):
        self.create = AsyncMock()

    def configure(self, response: FakeOpenAIResponse):
        self.create = AsyncMock(return_value=response)

    def configure_stream(self, chunks: List[str]):
        async def _stream():
            for c in chunks:
                yield FakeOpenAIStreamChunk(c)
        self.create = AsyncMock(return_value=_stream())


class FakeOpenAIChat:
    def __init__(self):
        self.completions = FakeOpenAIChatCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeOpenAIChat()
        self.close = AsyncMock()


def _make_anthropic_aux_client(response_text: str = "hello") -> AuxiliaryClient:
    """Build an AuxiliaryClient with a fake Anthropic SDK client."""
    config = AuxConfig(
        model="claude-3-haiku-20240307",
        api_key="test-key",
        provider="anthropic",
    )
    client = AuxiliaryClient(config)
    fake_sdk = FakeAnthropicClient()
    fake_sdk.messages.configure(FakeAnthropicResponse(
        content=[FakeAnthropicBlock("text", text=response_text)],
        stop_reason="end_turn",
    ))
    client._client = fake_sdk  # type: ignore[attr-defined]
    return client


def _make_openai_aux_client(response_text: str = "hello") -> AuxiliaryClient:
    """Build an AuxiliaryClient with a fake OpenAI SDK client."""
    config = AuxConfig(
        model="gpt-4o-mini",
        api_key="test-key",
        provider="openai",
    )
    client = AuxiliaryClient(config)
    fake_sdk = FakeOpenAIClient()
    fake_sdk.chat.completions.configure(FakeOpenAIResponse(
        choice=FakeOpenAIChoice(message=FakeOpenAIMessage(content=response_text)),
    ))
    client._client = fake_sdk  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# get_per_task_defaults
# ---------------------------------------------------------------------------


class TestGetPerTaskDefaults:
    def test_known_task_returns_defaults(self):
        defaults = get_per_task_defaults("compression")
        assert defaults["max_tokens"] == 1024
        assert defaults["temperature"] == 0.0

    def test_title_generation_has_low_max_tokens(self):
        defaults = get_per_task_defaults("title_generation")
        assert defaults["max_tokens"] == 64
        assert defaults["temperature"] == 0.3

    def test_unknown_task_returns_sensible_default(self):
        defaults = get_per_task_defaults("nonexistent_task")
        assert defaults["max_tokens"] == 1024
        assert defaults["temperature"] == 0.0

    def test_returns_copy(self):
        d1 = get_per_task_defaults("compression")
        d1["max_tokens"] = 9999
        d2 = get_per_task_defaults("compression")
        assert d2["max_tokens"] == 1024  # original unchanged


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


class TestCallLlm:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_aux_configured(self):
        """With no env vars and no config, call_llm returns None."""
        result = await call_llm("test prompt", use_fallback=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_cached_client(self, monkeypatch):
        """call_llm should cache the client for reuse."""
        # Inject a fake client via _get_or_create_client.
        fake_client = _make_anthropic_aux_client("summary text")
        call_count = 0

        async def fake_get_or_create(task=None):
            nonlocal call_count
            call_count += 1
            return fake_client

        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result1 = await call_llm("prompt 1", task="compression", use_fallback=False)
        result2 = await call_llm("prompt 2", task="compression", use_fallback=False)

        assert result1 == "summary text"
        assert result2 == "summary text"
        # _get_or_create_client was called twice (once per call_llm), but
        # internally it should have returned the cached client.
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_per_task_defaults_applied(self, monkeypatch):
        """Per-task max_tokens / temperature should be applied."""
        fake_client = _make_anthropic_aux_client("ok")
        # Track the kwargs passed to the SDK.
        original_create = fake_client._client.messages.create  # type: ignore[attr-defined]
        captured_kwargs: Dict[str, Any] = {}

        async def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return await original_create(**kwargs)

        fake_client._client.messages.create = capture_create  # type: ignore[attr-defined]

        async def fake_get_or_create(task=None):
            return fake_client

        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        await call_llm("test", task="title_generation", use_fallback=False)
        # title_generation defaults: max_tokens=64, temperature=0.3
        assert captured_kwargs["max_tokens"] == 64
        assert captured_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_explicit_overrides_per_task_defaults(self, monkeypatch):
        """Explicit max_tokens/temperature should override per-task defaults."""
        fake_client = _make_anthropic_aux_client("ok")
        original_create = fake_client._client.messages.create  # type: ignore[attr-defined]
        captured_kwargs: Dict[str, Any] = {}

        async def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return await original_create(**kwargs)

        fake_client._client.messages.create = capture_create  # type: ignore[attr-defined]

        async def fake_get_or_create(task=None):
            return fake_client

        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        await call_llm(
            "test",
            task="title_generation",
            max_tokens=200,
            temperature=0.7,
            use_fallback=False,
        )
        assert captured_kwargs["max_tokens"] == 200
        assert captured_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_use_fallback_false_propagates_failure(self, monkeypatch):
        """With use_fallback=False, a failing client returns None."""
        fake_client = _make_anthropic_aux_client("ok")
        # Make complete raise.
        async def fail_complete(*args, **kwargs):
            raise RuntimeError("API down")
        fake_client.complete = fail_complete  # type: ignore[assignment]

        async def fake_get_or_create(task=None):
            return fake_client

        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await call_llm("test", use_fallback=False)
        assert result is None


# ---------------------------------------------------------------------------
# complete_with_tools
# ---------------------------------------------------------------------------


class TestCompleteWithTools:
    @pytest.mark.asyncio
    async def test_anthropic_text_response(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ok")
        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await complete_with_tools(
            "check this",
            tools=[{"name": "check", "description": "checks", "input_schema": {"type": "object"}}],
        )
        assert result["text"] == "ok"
        assert result["tool_calls"] == []
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_anthropic_tool_call_response(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ignored")
        # Override the response to include a tool_use block.
        fake_client._client.messages.configure(  # type: ignore[attr-defined]
            FakeAnthropicResponse(
                content=[
                    FakeAnthropicBlock(
                        "tool_use",
                        id="call_1",
                        name="check_safety",
                        input={"verdict": "safe"},
                    ),
                ],
                stop_reason="tool_use",
            )
        )

        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await complete_with_tools(
            "is this safe?",
            tools=[{"name": "check_safety", "description": "...", "input_schema": {"type": "object"}}],
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "check_safety"
        assert result["tool_calls"][0]["input"] == {"verdict": "safe"}
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_openai_text_response(self, monkeypatch):
        fake_client = _make_openai_aux_client("ok")
        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await complete_with_tools(
            "check this",
            tools=[{"name": "check", "description": "...", "input_schema": {"type": "object"}}],
        )
        assert result["text"] == "ok"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_openai_tool_call_response(self, monkeypatch):
        fake_client = _make_openai_aux_client("ignored")
        # Override with a tool-call response.
        import json
        fake_client._client.chat.completions.configure(  # type: ignore[attr-defined]
            FakeOpenAIResponse(
                choice=FakeOpenAIChoice(
                    message=FakeOpenAIMessage(
                        content="",
                        tool_calls=[
                            FakeOpenAIToolCall(
                                id="call_1",
                                name="check_safety",
                                arguments=json.dumps({"verdict": "safe"}),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
            )
        )

        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await complete_with_tools(
            "is this safe?",
            tools=[{"name": "check_safety", "description": "...", "input_schema": {"type": "object"}}],
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "check_safety"
        assert result["tool_calls"][0]["input"] == {"verdict": "safe"}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_aux(self, monkeypatch):
        async def fake_get_or_create(task=None):
            return None
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        result = await complete_with_tools("test", tools=[])
        assert result == {}


# ---------------------------------------------------------------------------
# complete_stream
# ---------------------------------------------------------------------------


class TestCompleteStream:
    @pytest.mark.asyncio
    async def test_anthropic_stream(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ignored")
        fake_client._client.messages.configure_stream(["Hello ", "world", "!"])  # type: ignore[attr-defined]

        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        chunks = []
        async for chunk in complete_stream("test"):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_openai_stream(self, monkeypatch):
        fake_client = _make_openai_aux_client("ignored")
        fake_client._client.chat.completions.configure_stream(["Hello ", "world"])  # type: ignore[attr-defined]

        async def fake_get_or_create(task=None):
            return fake_client
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        chunks = []
        async for chunk in complete_stream("test"):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_no_aux_yields_nothing(self, monkeypatch):
        async def fake_get_or_create(task=None):
            return None
        monkeypatch.setattr(ext_mod, "_get_or_create_client", fake_get_or_create)

        chunks = []
        async for chunk in complete_stream("test"):
            chunks.append(chunk)
        assert chunks == []


# ---------------------------------------------------------------------------
# Vision backend
# ---------------------------------------------------------------------------


class TestVisionBackends:
    def test_no_backends_when_no_env(self):
        """With no env vars set, no vision backends are available."""
        backends = get_available_vision_backends()
        assert backends == []

    def test_anthropic_backend_detected(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        backends = get_available_vision_backends()
        assert len(backends) >= 1
        anthropic = [b for b in backends if b.provider == "anthropic"]
        assert len(anthropic) == 1
        assert anthropic[0].model in {
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
        }

    def test_openai_backend_detected(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        backends = get_available_vision_backends()
        openai_backends = [b for b in backends if b.provider == "openai"]
        assert len(openai_backends) == 1

    def test_multiple_backends_detected(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        backends = get_available_vision_backends()
        providers = {b.provider for b in backends}
        assert "anthropic" in providers
        assert "openai" in providers
        assert "groq" in providers

    def test_resolve_returns_first_available(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        backend = resolve_vision_provider_client()
        assert backend is not None
        assert backend.provider == "anthropic"

    def test_resolve_prefers_requested_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        backend = resolve_vision_provider_client(preferred_provider="openai")
        assert backend is not None
        assert backend.provider == "openai"

    def test_resolve_returns_none_when_no_backends(self):
        backend = resolve_vision_provider_client()
        assert backend is None

    def test_resolve_falls_back_when_preferred_unavailable(self, monkeypatch):
        """If the preferred provider isn't configured, return the first available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        backend = resolve_vision_provider_client(preferred_provider="openai")
        assert backend is not None
        assert backend.provider == "anthropic"


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_clears_cache(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ok")
        async def fake_get_aux_client(task=None):
            return fake_client
        monkeypatch.setattr("niaharness.auxiliary.get_aux_client", fake_get_aux_client)

        await call_llm("test", use_fallback=False)
        assert get_cached_client_count() >= 1

        await shutdown_cached_clients()
        assert get_cached_client_count() == 0

    @pytest.mark.asyncio
    async def test_shutdown_closes_sdk_clients(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ok")
        close_mock = fake_client._client.close  # type: ignore[attr-defined]

        async def fake_get_aux_client(task=None):
            return fake_client
        monkeypatch.setattr("niaharness.auxiliary.get_aux_client", fake_get_aux_client)

        await call_llm("test", use_fallback=False)
        await shutdown_cached_clients()

        close_mock.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Calling shutdown twice should not crash."""
        await shutdown_cached_clients()
        await shutdown_cached_clients()
        assert get_cached_client_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_evicts_stale_clients(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ok")
        async def fake_get_aux_client(task=None):
            return fake_client
        monkeypatch.setattr("niaharness.auxiliary.get_aux_client", fake_get_aux_client)

        await call_llm("test", use_fallback=False)
        assert get_cached_client_count() >= 1

        # Manually age the cached entry.
        for cached in ext_mod._client_cache.values():
            cached.last_used = time.monotonic() - 9999

        evicted = await cleanup_stale_async_clients(max_idle_seconds=60)
        assert evicted >= 1
        assert get_cached_client_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_clients(self, monkeypatch):
        fake_client = _make_anthropic_aux_client("ok")
        async def fake_get_aux_client(task=None):
            return fake_client
        monkeypatch.setattr("niaharness.auxiliary.get_aux_client", fake_get_aux_client)

        await call_llm("test", use_fallback=False)
        evicted = await cleanup_stale_async_clients(max_idle_seconds=3600)
        assert evicted == 0
        assert get_cached_client_count() >= 1


# ---------------------------------------------------------------------------
# Runtime main injection
# ---------------------------------------------------------------------------


class TestRuntimeMain:
    def test_set_and_get_runtime(self):
        runtime = MagicMock()
        set_runtime_main(runtime)
        assert get_runtime_main() is runtime

    def test_reset_runtime(self):
        runtime = MagicMock()
        set_runtime_main(runtime)
        reset_runtime_main()
        assert get_runtime_main() is None

    def test_get_returns_none_when_not_set(self):
        assert get_runtime_main() is None


# ---------------------------------------------------------------------------
# Credential refresh
# ---------------------------------------------------------------------------


class TestCredentialRefresh:
    @pytest.mark.asyncio
    async def test_no_runtime_returns_false(self):
        client = _make_anthropic_aux_client("ok")
        result = await refresh_credentials_for_client(client)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_credential_pool_returns_false(self):
        runtime = MagicMock()
        runtime.credential_pool = None
        set_runtime_main(runtime)

        client = _make_anthropic_aux_client("ok")
        result = await refresh_credentials_for_client(client)
        assert result is False

    @pytest.mark.asyncio
    async def test_pool_returns_none_returns_false(self):
        runtime = MagicMock()
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=None)
        runtime.credential_pool = pool
        set_runtime_main(runtime)

        client = _make_anthropic_aux_client("ok")
        result = await refresh_credentials_for_client(client)
        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_updates_config_and_clears_sdk_client(self):
        runtime = MagicMock()
        pool = MagicMock()
        cred = MagicMock()
        cred.api_key = "new-key"
        pool.acquire = MagicMock(return_value=cred)
        runtime.credential_pool = pool
        set_runtime_main(runtime)

        client = _make_anthropic_aux_client("ok")
        original_sdk = client._client  # type: ignore[attr-defined]
        result = await refresh_credentials_for_client(client)

        assert result is True
        # SDK client should be cleared (will be re-created on next call).
        assert client._client is None  # type: ignore[attr-defined]
        # Config's api_key should be updated.
        assert client.config.api_key == "new-key"


# ---------------------------------------------------------------------------
# Additional providers
# ---------------------------------------------------------------------------


class TestAdditionalProviders:
    def test_no_providers_when_no_env(self):
        assert detect_additional_providers() == []

    def test_xai_detected(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        providers = detect_additional_providers()
        xai = [p for p in providers if p["provider"] == "xai"]
        assert len(xai) == 1
        assert xai[0]["api_key"] == "test-key"
        assert xai[0]["base_url"] == "https://api.x.ai/v1"
        assert xai[0]["model"] == "grok-2-vision-1212"

    def test_nous_detected(self, monkeypatch):
        monkeypatch.setenv("NOUS_API_KEY", "test-key")
        providers = detect_additional_providers()
        nous = [p for p in providers if p["provider"] == "nous"]
        assert len(nous) == 1

    def test_azure_requires_three_env_vars(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        # Missing endpoint + deployment → not detected.
        providers = detect_additional_providers()
        azure = [p for p in providers if p["provider"] == "azure"]
        assert len(azure) == 0

        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        providers = detect_additional_providers()
        azure = [p for p in providers if p["provider"] == "azure"]
        assert len(azure) == 1
        assert azure[0]["base_url"] == "https://example.openai.azure.com"
        assert azure[0]["model"] == "gpt-4o"

    def test_multiple_providers_detected(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "x")
        monkeypatch.setenv("NOUS_API_KEY", "n")
        monkeypatch.setenv("TOGETHER_API_KEY", "t")
        monkeypatch.setenv("FIREWORKS_API_KEY", "f")
        providers = detect_additional_providers()
        provider_names = {p["provider"] for p in providers}
        assert {"xai", "nous", "together", "fireworks"} <= provider_names

    def test_additional_provider_env_vars_dict_complete(self):
        """All entries should have the required keys."""
        for name, info in ADDITIONAL_PROVIDER_ENV_VARS.items():
            assert "env_var" in info, f"{name} missing env_var"
            assert "default_model" in info or "default_model_env" in info, \
                f"{name} missing model"
            assert "openai_compatible" in info, f"{name} missing openai_compatible"


# ---------------------------------------------------------------------------
# Client cache behavior
# ---------------------------------------------------------------------------


class TestClientCache:
    @pytest.mark.asyncio
    async def test_same_task_reuses_client(self, monkeypatch):
        """Two call_llm with the same task should reuse the cached client."""
        fake_client = _make_anthropic_aux_client("ok")
        create_count = 0

        async def fake_get_aux_client(task=None):
            nonlocal create_count
            create_count += 1
            return fake_client

        monkeypatch.setattr("niaharness.auxiliary.get_aux_client", fake_get_aux_client)

        await call_llm("test1", use_fallback=False)
        await call_llm("test2", use_fallback=False)

        # get_aux_client should only have been called once — the second
        # call_llm should have hit the cache.
        assert create_count == 1
        assert get_cached_client_count() == 1
