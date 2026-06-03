"""Adapter: NIA LLMProvider → niaharness SupportsStreamingMessages.

Bridges NIA's provider interface (complete/stream) to the
SupportsStreamingMessages protocol expected by QueryEngine.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from niaharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
)
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import LLMRequest

logger = logging.getLogger(__name__)


class NIAProviderAdapter:
    """Wraps an NIA LLMProvider to satisfy the SupportsStreamingMessages protocol.

    QueryEngine expects:
        async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]

    NIA providers offer:
        async def complete(self, request: LLMRequest) -> LLMResponse
        async def stream(self, request: LLMRequest) -> AsyncIterator[str]

    This adapter bridges the two by converting request/response formats.
    """

    def __init__(self, provider: LLMProvider, model: str | None = None) -> None:
        self._provider = provider
        self._model = model

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Convert ApiMessageRequest → LLMRequest, call provider, yield ApiStreamEvents."""
        # Convert ConversationMessage list to plain dict list
        messages = []
        for msg in request.messages:
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    messages.append({"role": msg.role, "content": block.text})

        # Build NIA LLMRequest
        nia_request = LLMRequest(
            model=request.model,
            messages=messages,
            system=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature or 0.3,
            tools=request.tools if request.tools else None,
            stream=False,  # Use complete() for simplicity
        )

        try:
            response = await self._provider.complete(nia_request)
        except Exception as e:
            logger.error(f"Provider call failed: {e}")
            # Yield an empty text response so QueryEngine doesn't crash
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=f"Error: {e}")],
                ),
                usage=UsageSnapshot(input_tokens=0, output_tokens=0),
                stop_reason="error",
            )
            return

        # Build response blocks
        content_blocks = []

        # If there are tool calls from the provider
        if response.tool_calls:
            for tc in response.tool_calls:
                content_blocks.append(
                    ToolUseBlock(
                        id=tc.get("id", f"toolu_{id(tc)}"),
                        name=tc.get("name", "unknown"),
                        input=tc.get("input", {}),
                    )
                )

        # Always include text content
        if response.content:
            content_blocks.append(TextBlock(text=response.content))

        # If no blocks at all, add empty text
        if not content_blocks:
            content_blocks.append(TextBlock(text=""))

        message = ConversationMessage(
            role="assistant",
            content=content_blocks,
        )

        usage = UsageSnapshot(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        # Yield text deltas for streaming display, then the complete event
        if response.content:
            yield ApiTextDeltaEvent(text=response.content)

        yield ApiMessageCompleteEvent(
            message=message,
            usage=usage,
            stop_reason=response.finish_reason,
        )
