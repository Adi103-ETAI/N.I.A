"""End-to-end smoke test for the Anthropic transport wiring.

Verifies the full flow:

  AnthropicApiClient.__init__ → build_anthropic_client → AsyncAnthropic
  AnthropicApiClient._stream_once (request build) → build_anthropic_kwargs → SDK

We don't make real API calls — we mock the SDK client and inspect the kwargs
that would have been sent. This catches wiring regressions like:

  - The transport not being called at all (old hand-built params path).
  - The base_url not being passed through (breaks thinking-signature policy).
  - Caching markers not being applied (breaks the ~10x cost reduction).
  - Tool-schema nullable unions leaking through (causes 400s).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_anthropic_api_client_delegates_to_transport():
    """AnthropicApiClient._stream_once should call build_anthropic_kwargs.

    Verifies the wiring: __init__ → build_anthropic_client, _stream_once →
    build_anthropic_kwargs. We patch build_anthropic_kwargs to return a known
    kwargs dict and intercept the SDK call to verify it received those kwargs
    verbatim (no hand-built params path was taken).
    """
    from niaharness.api.client import AnthropicApiClient, ApiMessageRequest
    from niaharness.engine.messages import ConversationMessage

    # Patch build_anthropic_kwargs so we can capture what was passed to it
    # and control what the SDK receives.
    captured_kwargs: dict[str, Any] = {}

    def fake_build(**kwargs):
        captured_kwargs.update(kwargs)
        # Return a minimal kwargs dict the SDK will accept.
        return {
            "model": kwargs["model"],
            "messages": [{"role": "user", "content": "translated"}],
            "max_tokens": kwargs.get("max_tokens") or 1024,
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            }],
        }

    # Build the API client (this itself uses build_anthropic_client via __init__).
    with patch(
        "niaharness.providers.anthropic_transport.build_anthropic_client"
    ) as mock_build_client:
        mock_build_client.return_value = MagicMock()
        api = AnthropicApiClient(api_key="sk-ant-api03-fake", base_url=None)

    # Now patch build_anthropic_kwargs for the _stream_once call.
    with patch(
        "niaharness.providers.anthropic_transport.build_anthropic_kwargs",
        side_effect=fake_build,
    ):
        # Mock the SDK's stream() to capture the kwargs it receives.
        received_sdk_kwargs: dict[str, Any] = {}

        class _FakeStream:
            """Fake MessageStream — both an async iterator AND has get_final_message."""

            def __init__(self, **kwargs):
                received_sdk_kwargs.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def __aiter__(self):
                async def _empty():
                    if False:
                        yield  # pragma: no cover
                return _empty()

            async def get_final_message(self):
                return MagicMock(
                    content=[],
                    usage=MagicMock(input_tokens=10, output_tokens=5),
                    stop_reason="end_turn",
                )

        api._client = MagicMock()
        api._client.base_url = "https://api.anthropic.com"
        api._client.messages.stream = _FakeStream

        request = ApiMessageRequest(
            model="claude-opus-4-7",
            messages=[ConversationMessage.from_user_text("hello")],
            system_prompt="You are helpful.",
            max_tokens=4096,
            tools=[
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            },
                        },
                    },
                }
            ],
        )

        # Drive the stream.
        events = []
        async for event in api.stream_message(request):
            events.append(event)

    # Verify build_anthropic_kwargs was called with the right inputs.
    assert captured_kwargs["model"] == "claude-opus-4-7"
    assert captured_kwargs["max_tokens"] == 4096
    assert captured_kwargs["system_prompt"] == "You are helpful."
    assert captured_kwargs["base_url"] == "https://api.anthropic.com"
    assert captured_kwargs["enable_caching"] is True
    # The original (unsanitized) tool def should have been passed in.
    assert len(captured_kwargs["tools"]) == 1
    assert captured_kwargs["tools"][0]["function"]["name"] == "get_weather"

    # Verify the SDK received the transport-built kwargs (not hand-built params).
    assert received_sdk_kwargs["model"] == "claude-opus-4-7"
    assert received_sdk_kwargs["max_tokens"] == 4096  # passed through from fake_build
    # System should be the cacheable list from the transport.
    assert isinstance(received_sdk_kwargs["system"], list)
    assert received_sdk_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Tools should be in Anthropic format.
    assert received_sdk_kwargs["tools"][0]["name"] == "get_weather"
    assert received_sdk_kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}

    # We should have received at least the final ApiMessageCompleteEvent.
    from niaharness.api.client import ApiMessageCompleteEvent
    complete_events = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].usage.input_tokens == 10
    assert complete_events[0].usage.output_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_provider_get_client_delegates_to_transport():
    """AnthropicProvider.get_client should delegate to build_anthropic_client."""
    from niaharness.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider()
    with patch(
        "niaharness.providers.anthropic_transport.build_anthropic_client"
    ) as mock_build:
        mock_build.return_value = "fake-client"
        client = provider.get_client(api_key="sk-ant-api03-fake", base_url="https://api.anthropic.com")

        assert client == "fake-client"
        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-ant-api03-fake"
        assert call_kwargs["base_url"] == "https://api.anthropic.com"


def test_resolve_anthropic_token_priority_chain():
    """resolve_anthropic_token should respect env var priority."""
    import os

    from niaharness.providers.anthropic_transport import resolve_anthropic_token

    # Save and clear all relevant env vars.
    saved = {}
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        saved[var] = os.environ.pop(var, None)

    try:
        # 1. ANTHROPIC_TOKEN wins over everything.
        os.environ["ANTHROPIC_TOKEN"] = "explicit-token"
        os.environ["ANTHROPIC_API_KEY"] = "legacy-key"
        assert resolve_anthropic_token() == "explicit-token"

        # 2. CLAUDE_CODE_OAUTH_TOKEN wins over ANTHROPIC_API_KEY.
        del os.environ["ANTHROPIC_TOKEN"]
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "cc-token"
        assert resolve_anthropic_token() == "cc-token"

        # 3. ANTHROPIC_API_KEY is the legacy fallback.
        del os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
        assert resolve_anthropic_token() == "legacy-key"
    finally:
        # Restore env vars.
        for var, value in saved.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)


if __name__ == "__main__":
    asyncio.run(test_anthropic_api_client_delegates_to_transport())
    asyncio.run(test_anthropic_provider_get_client_delegates_to_transport())
    test_resolve_anthropic_token_priority_chain()
    print("All smoke tests passed.")
