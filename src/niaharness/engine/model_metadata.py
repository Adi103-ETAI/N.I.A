"""Model metadata — context length, vision support, pricing per model.

Ported from Hermes Agent's agent/model_metadata.py (2,434 LOC), scoped
to the models NIA actually uses. Used by cost_tracker.py for accurate
cost estimation and by the query loop for context-window detection.

Usage::

    from niaharness.engine.model_metadata import get_context_window, estimate_cost

    ctx = get_context_window("claude-sonnet-4-20250514")  # → 200000
    cost = estimate_cost("claude-sonnet-4-20250514", 1000, 500)  # → 0.012
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single model.

    Attributes:
        context_window: Max context window in tokens.
        max_output_tokens: Max output tokens per response.
        supports_vision: Whether the model supports image input.
        supports_thinking: Whether the model supports extended thinking.
        input_price_per_1m: Price per 1M input tokens in USD.
        output_price_per_1m: Price per 1M output tokens in USD.
        cache_read_price_per_1m: Price per 1M cached input tokens (if applicable).
    """

    context_window: int = 200_000
    max_output_tokens: int = 8192
    supports_vision: bool = False
    supports_thinking: bool = False
    input_price_per_1m: float = 3.0
    output_price_per_1m: float = 15.0
    cache_read_price_per_1m: Optional[float] = None


# Model registry — prefix-matched (longest prefix wins).
# Prices as of 2026-07. Update quarterly.
_MODELS: dict[str, ModelInfo] = {
    # Anthropic Claude
    "claude-sonnet-4": ModelInfo(
        context_window=200_000, max_output_tokens=16384,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
        cache_read_price_per_1m=0.30,
    ),
    "claude-opus-4": ModelInfo(
        context_window=200_000, max_output_tokens=16384,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=15.0, output_price_per_1m=75.0,
        cache_read_price_per_1m=1.50,
    ),
    "claude-3-5-sonnet": ModelInfo(
        context_window=200_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
        cache_read_price_per_1m=0.30,
    ),
    "claude-3-5-haiku": ModelInfo(
        context_window=200_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=0.80, output_price_per_1m=4.0,
        cache_read_price_per_1m=0.08,
    ),
    "claude-3-opus": ModelInfo(
        context_window=200_000, max_output_tokens=4096,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=15.0, output_price_per_1m=75.0,
    ),
    "claude-3-sonnet": ModelInfo(
        context_window=200_000, max_output_tokens=4096,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
    ),
    "claude-3-haiku": ModelInfo(
        context_window=200_000, max_output_tokens=4096,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=0.25, output_price_per_1m=1.25,
    ),
    # OpenAI
    "gpt-4o": ModelInfo(
        context_window=128_000, max_output_tokens=16384,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=5.0, output_price_per_1m=15.0,
        cache_read_price_per_1m=2.50,
    ),
    "gpt-4o-mini": ModelInfo(
        context_window=128_000, max_output_tokens=16384,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=0.15, output_price_per_1m=0.60,
    ),
    "gpt-4-turbo": ModelInfo(
        context_window=128_000, max_output_tokens=4096,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=10.0, output_price_per_1m=30.0,
    ),
    "gpt-4": ModelInfo(
        context_window=8192, max_output_tokens=4096,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=30.0, output_price_per_1m=60.0,
    ),
    "o1": ModelInfo(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=False, supports_thinking=True,
        input_price_per_1m=15.0, output_price_per_1m=60.0,
    ),
    "o3-mini": ModelInfo(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=False, supports_thinking=True,
        input_price_per_1m=3.0, output_price_per_1m=12.0,
    ),
    "o3": ModelInfo(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=False, supports_thinking=True,
        input_price_per_1m=15.0, output_price_per_1m=60.0,
    ),
    "o4-mini": ModelInfo(
        context_window=200_000, max_output_tokens=100_000,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=1.10, output_price_per_1m=4.40,
    ),
    # Google Gemini
    "gemini-2.5-pro": ModelInfo(
        context_window=1_000_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=True,
        input_price_per_1m=1.25, output_price_per_1m=10.0,
    ),
    "gemini-2.0-flash": ModelInfo(
        context_window=1_000_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=0.10, output_price_per_1m=0.40,
    ),
    "gemini-1.5-pro": ModelInfo(
        context_window=2_000_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=1.25, output_price_per_1m=5.0,
    ),
    "gemini-1.5-flash": ModelInfo(
        context_window=1_000_000, max_output_tokens=8192,
        supports_vision=True, supports_thinking=False,
        input_price_per_1m=0.075, output_price_per_1m=0.30,
    ),
    # DeepSeek
    "deepseek-chat": ModelInfo(
        context_window=64_000, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=0.27, output_price_per_1m=1.10,
    ),
    "deepseek-reasoner": ModelInfo(
        context_window=64_000, max_output_tokens=8192,
        supports_vision=False, supports_thinking=True,
        input_price_per_1m=0.55, output_price_per_1m=2.19,
    ),
    "deepseek-v4": ModelInfo(
        context_window=128_000, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=0.27, output_price_per_1m=1.10,
    ),
    # Groq
    "llama-3.3-70b": ModelInfo(
        context_window=128_000, max_output_tokens=32768,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=0.59, output_price_per_1m=0.79,
    ),
    # xAI
    "grok-2": ModelInfo(
        context_window=131_072, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=2.0, output_price_per_1m=10.0,
    ),
    "grok-3": ModelInfo(
        context_window=131_072, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
    ),
    # Mistral
    "mistral-large": ModelInfo(
        context_window=128_000, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=2.0, output_price_per_1m=6.0,
    ),
    # Default fallback
    "_default": ModelInfo(
        context_window=200_000, max_output_tokens=8192,
        supports_vision=False, supports_thinking=False,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
    ),
}


def _get_model_info(model: str) -> ModelInfo:
    """Look up model info by name (prefix match, longest prefix wins)."""
    if not model:
        return _MODELS["_default"]
    model_lower = model.lower()
    # Try exact match first.
    if model_lower in _MODELS:
        return _MODELS[model_lower]
    # Try prefix match (longest prefix first).
    for prefix in sorted(_MODELS.keys(), key=len, reverse=True):
        if prefix == "_default":
            continue
        if model_lower.startswith(prefix):
            return _MODELS[prefix]
    return _MODELS["_default"]


def get_context_window(model: str) -> int:
    """Return the context window size in tokens for a model."""
    return _get_model_info(model).context_window


def get_max_output_tokens(model: str) -> int:
    """Return the max output tokens for a model."""
    return _get_model_info(model).max_output_tokens


def supports_vision(model: str) -> bool:
    """Return True if the model supports image input."""
    return _get_model_info(model).supports_vision


def supports_thinking(model: str) -> bool:
    """Return True if the model supports extended thinking."""
    return _get_model_info(model).supports_thinking


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost in USD for a model call.

    Args:
        model: The model name (e.g. 'claude-sonnet-4-20250514').
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.

    Returns:
        Estimated cost in USD.
    """
    info = _get_model_info(model)
    return (
        (input_tokens / 1_000_000) * info.input_price_per_1m
        + (output_tokens / 1_000_000) * info.output_price_per_1m
    )


__all__ = [
    "ModelInfo",
    "estimate_cost",
    "get_context_window",
    "get_max_output_tokens",
    "supports_thinking",
    "supports_vision",
]
