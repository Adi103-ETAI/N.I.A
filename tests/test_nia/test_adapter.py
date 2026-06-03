"""Tests for the NIA provider adapter."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from niaharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest, ApiStreamEvent, ApiTextDeltaEvent
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock

from agents.nia.providers.adapter import NIAProviderAdapter
from agents.nia.providers.types import LLMRequest, LLMResponse


class FakeNIAProvider:
    """Fake NIA LLMProvider for testing the adapter."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._calls.append(request)
        return self._responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        resp = await self.complete(request)
        yield resp.content

    @property
    def id(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return "Fake Provider"

    def list_models(self):
        return []


@pytest.mark.asyncio
async def test_adapter_converts_text_response(tmp_path: Path):
    """Adapter converts NIA LLMResponse to ApiMessageCompleteEvent."""
    provider = FakeNIAProvider([
        LLMResponse(
            content="Hello from NIA",
            model="test-model",
            provider="fake",
            input_tokens=10,
            output_tokens=5,
        )
    ])

    adapter = NIAProviderAdapter(provider, model="test-model")

    request = ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
        system_prompt="system",
        max_tokens=4096,
    )

    events = [event async for event in adapter.stream_message(request)]

    # Should have text delta + complete event
    assert len(events) == 2
    assert isinstance(events[0], ApiTextDeltaEvent)
    assert events[0].text == "Hello from NIA"
    assert isinstance(events[1], ApiMessageCompleteEvent)
    assert events[1].usage.input_tokens == 10
    assert events[1].usage.output_tokens == 5

    # Verify the provider was called with correct format
    assert len(provider._calls) == 1
    call = provider._calls[0]
    assert call.model == "test-model"
    assert call.messages == [{"role": "user", "content": "hi"}]
    assert call.system == "system"


@pytest.mark.asyncio
async def test_adapter_converts_tool_calls(tmp_path: Path):
    """Adapter converts tool calls from NIA provider to ToolUseBlock."""
    provider = FakeNIAProvider([
        LLMResponse(
            content="I'll read the file",
            model="test-model",
            provider="fake",
            input_tokens=5,
            output_tokens=3,
            tool_calls=[
                {"id": "toolu_123", "name": "file_read", "input": {"file_path": "test.py"}}
            ],
        )
    ])

    adapter = NIAProviderAdapter(provider)

    request = ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="read test.py")])],
        system_prompt="system",
        max_tokens=4096,
    )

    events = [event async for event in adapter.stream_message(request)]

    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(complete) == 1

    msg = complete[0].message
    # Should have ToolUseBlock + TextBlock
    tool_blocks = [b for b in msg.content if isinstance(b, ToolUseBlock)]
    text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]

    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "file_read"
    assert tool_blocks[0].input == {"file_path": "test.py"}
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "I'll read the file"


@pytest.mark.asyncio
async def test_adapter_handles_provider_error(tmp_path: Path):
    """Adapter handles provider errors gracefully."""
    class FailingProvider(FakeNIAProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self._calls.append(request)
            raise RuntimeError("Provider failed")

    provider = FailingProvider([])
    adapter = NIAProviderAdapter(provider)

    request = ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
        system_prompt="system",
        max_tokens=4096,
    )

    events = [event async for event in adapter.stream_message(request)]

    # Should yield one complete event with error text, not crash
    assert len(events) == 1
    assert isinstance(events[0], ApiMessageCompleteEvent)
    assert "Error" in events[0].message.content[0].text


@pytest.mark.asyncio
async def test_adapter_empty_response(tmp_path: Path):
    """Adapter handles empty content from provider."""
    provider = FakeNIAProvider([
        LLMResponse(
            content="",
            model="test-model",
            provider="fake",
        )
    ])

    adapter = NIAProviderAdapter(provider)

    request = ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
        system_prompt="system",
        max_tokens=4096,
    )

    events = [event async for event in adapter.stream_message(request)]

    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(complete) == 1
    # Should still have a text block (even if empty)
    assert len(complete[0].message.content) >= 1
