"""Context window management utilities.

Provides functions for managing model context windows, output token limits,
and context usage calculations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Model context window size (200k tokens for all models right now)
MODEL_CONTEXT_WINDOW_DEFAULT = 200_000

# Fallback context window for unknown models
OPENAI_FALLBACK_CONTEXT_WINDOW = 128_000

# Maximum output tokens for compact operations
COMPACT_MAX_OUTPUT_TOKENS = 20_000

# Default max output tokens
MAX_OUTPUT_TOKENS_DEFAULT = 32_000
MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64_000

# Capped default for slot-reservation optimization
CAPPED_DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000


@dataclass(frozen=True)
class ContextPercentages:
    """Context window usage percentages."""

    used: Optional[int]
    remaining: Optional[int]


@dataclass(frozen=True)
class ModelOutputTokens:
    """Model output token limits."""

    default: int
    upper_limit: int


def is_1m_context_disabled() -> bool:
    """Check if 1M context is disabled via environment variable."""
    return os.environ.get("NIAHARNESS_DISABLE_1M_CONTEXT", "").lower() in (
        "1",
        "true",
        "yes",
    )


def has_1m_context(model: str) -> bool:
    """Check if a model string has the [1m] suffix."""
    if is_1m_context_disabled():
        return False
    return bool(re.search(r"\[1m\]", model, re.IGNORECASE))


def model_supports_1m(model: str) -> bool:
    """Check if a model supports 1M context window."""
    if is_1m_context_disabled():
        return False
    # Check for known 1M-capable models
    model_lower = model.lower()
    return "claude-sonnet-4" in model_lower or "opus-4-6" in model_lower


import re  # noqa: E402 (needed for model_supports_1m)


def get_context_window_for_model(
    model: str,
    betas: Optional[list[str]] = None,
) -> int:
    """Get the context window size for a model.

    Returns the context window size in tokens.
    """
    # Allow override via environment variable
    override_str = os.environ.get("NIAHARNESS_MAX_CONTEXT_TOKENS", "")
    if override_str:
        try:
            override = int(override_str)
            if override > 0:
                return override
        except ValueError:
            pass

    # [1m] suffix — explicit client-side opt-in
    if has_1m_context(model):
        return 1_000_000

    # Check model-specific capabilities
    model_lower = model.lower()

    if "opus-4-6" in model_lower or "sonnet-4-6" in model_lower:
        return 1_000_000
    if "claude-3" in model_lower or "sonnet-4" in model_lower or "haiku-4" in model_lower:
        return 200_000

    return MODEL_CONTEXT_WINDOW_DEFAULT


def calculate_context_percentages(
    current_usage: Optional[dict[str, int]],
    context_window_size: int,
) -> ContextPercentages:
    """Calculate context window usage percentage from token usage data.

    Returns used and remaining percentages, or None values if no usage data.
    """
    if not current_usage:
        return ContextPercentages(used=None, remaining=None)

    total_input_tokens = (
        current_usage.get("input_tokens", 0)
        + current_usage.get("cache_creation_input_tokens", 0)
        + current_usage.get("cache_read_input_tokens", 0)
    )

    used_percentage = round((total_input_tokens / context_window_size) * 100)
    clamped_used = max(0, min(100, used_percentage))

    return ContextPercentages(used=clamped_used, remaining=100 - clamped_used)


def get_model_max_output_tokens(model: str) -> ModelOutputTokens:
    """Get the default and upper limit for max output tokens of a model."""
    model_lower = model.lower()

    if "opus-4-6" in model_lower:
        return ModelOutputTokens(default=64_000, upper_limit=128_000)
    if "sonnet-4-6" in model_lower:
        return ModelOutputTokens(default=32_000, upper_limit=128_000)
    if any(m in model_lower for m in ("opus-4-5", "sonnet-4", "haiku-4")):
        return ModelOutputTokens(default=32_000, upper_limit=64_000)
    if any(m in model_lower for m in ("opus-4-1", "opus-4")):
        return ModelOutputTokens(default=32_000, upper_limit=32_000)
    if "claude-3-opus" in model_lower:
        return ModelOutputTokens(default=4_096, upper_limit=4_096)
    if "claude-3-sonnet" in model_lower:
        return ModelOutputTokens(default=8_192, upper_limit=8_192)
    if "claude-3-haiku" in model_lower:
        return ModelOutputTokens(default=4_096, upper_limit=4_096)
    if any(m in model_lower for m in ("3-5-sonnet", "3-5-haiku")):
        return ModelOutputTokens(default=8_192, upper_limit=8_192)
    if "3-7-sonnet" in model_lower:
        return ModelOutputTokens(default=32_000, upper_limit=64_000)

    return ModelOutputTokens(default=MAX_OUTPUT_TOKENS_DEFAULT, upper_limit=MAX_OUTPUT_TOKENS_UPPER_LIMIT)


def get_max_thinking_tokens_for_model(model: str) -> int:
    """Get the max thinking budget tokens for a model.

    The max thinking tokens should be strictly less than the max output tokens.
    """
    return get_model_max_output_tokens(model).upper_limit - 1
