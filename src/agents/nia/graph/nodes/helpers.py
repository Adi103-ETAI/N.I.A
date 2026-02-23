"""NIA Graph Nodes — Config & Summarization Helpers.

Provides:
    - Config file loaders (vision triggers, prompts) with lazy caching
    - summarize_oldest / asummarize_oldest for context-window management

Config files are now at src/core/config/defaults/ (moved from root config/ in Tier 2).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from src.core.logger import setup_logger
from src.core.config import settings

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    _HAS_LANGCHAIN_MESSAGES = True
except ImportError:
    _HAS_LANGCHAIN_MESSAGES = False
    SystemMessage = None   # type: ignore
    HumanMessage  = None   # type: ignore
    AIMessage     = None   # type: ignore

from src.agents.nia.state import safe_get_content

logger = setup_logger("NIA.Nodes.Helpers")

# Config data directory:
#   helpers.py → nodes/ → graph/ → nia/ → agents/ → src/
#   parents[4] = src/,  so data lives at src/core/config/defaults/
_CONFIG_DATA = Path(__file__).resolve().parents[4] / "core" / "config" / "data"


# =============================================================================
# Vision Config (lazy-cached)
# =============================================================================

_VISION_CONFIG: Optional[dict] = None


def _load_vision_config() -> dict:
    """Load vision trigger keywords from data/iris/triggers.json."""
    config_path = _CONFIG_DATA / "iris" / "triggers.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load triggers.json: {e}. Using defaults.")
        return {
            "triggers": {
                "screen":  ["screen", "screenshot", "window"],
                "camera":  ["camera", "webcam", "photo"],
                "actions": ["look at", "what do you see", "vision"],
            }
        }


def get_vision_keywords() -> list:
    """Return all vision trigger keywords (cached after first load)."""
    global _VISION_CONFIG
    if _VISION_CONFIG is None:
        _VISION_CONFIG = _load_vision_config()
    triggers = _VISION_CONFIG.get("triggers", {})
    return (
        triggers.get("screen",  []) +
        triggers.get("camera",  []) +
        triggers.get("actions", [])
    )


# =============================================================================
# Prompts Config (lazy-cached)
# =============================================================================

_PROMPTS_CONFIG: Optional[dict] = None


def _load_prompts_config() -> dict:
    """Load system prompts from data/nia/prompts.json."""
    config_path = _CONFIG_DATA / "nia" / "prompts.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load prompts.json: {e}. Using defaults.")
        return {
            "identity":   "You are N.I.A., a helpful AI assistant.",
            "supervisor": "You are the Supervisor. Route questions to workers.",
        }


def get_prompts() -> dict:
    """Return all system prompts (cached after first load)."""
    global _PROMPTS_CONFIG
    if _PROMPTS_CONFIG is None:
        _PROMPTS_CONFIG = _load_prompts_config()
    return _PROMPTS_CONFIG


# =============================================================================
# Context-Window Summarizer
# =============================================================================

def summarize_oldest(messages: List, llm=None) -> List:
    """Compress the oldest messages into a summary when history exceeds the limit.

    Prevents context-window overflow by replacing a batch of old messages
    with a single concise summary paragraph. Always preserves:
        - messages[0]  — the System Prompt
        - most recent messages (beyond ``prune_count``)

    Args:
        messages: Full message list.
        llm:      LLM to use for summarization. Lazy-loaded if None.

    Returns:
        Compressed list if over limit, original list otherwise.
    """
    max_history = settings.MAX_HISTORY
    prune_count = settings.PRUNE_COUNT

    if len(messages) <= max_history:
        return messages

    if not _HAS_LANGCHAIN_MESSAGES:
        logger.warning("LangChain messages unavailable — truncating history")
        return messages[:max_history]

    logger.info(f"🧹 Pruning: {len(messages)} messages > {max_history}. Compressing...")

    # Lazy-load LLM if not provided
    if not llm:
        try:
            from src.models.manager import ModelManager
            manager = ModelManager()
            llm = manager.get_fast_model() or manager.get_smart_model()
            if llm:
                logger.debug("Loaded LLM for summarization via ModelManager")
        except Exception as e:
            logger.error(f"Failed to load LLM for summarization: {e}")
            return messages[-max_history:]

    if not llm:
        logger.warning("No LLM available — truncating to recent messages")
        return messages[-max_history:]

    system_prompt  = messages[0]
    to_summarize   = messages[1:1 + prune_count]
    remaining      = messages[1 + prune_count:]

    summary_request = [
        SystemMessage(content=(
            "You are a helpful assistant. Summarize the following conversation "
            "into a concise context paragraph. Preserve key constraints, facts, "
            "and user goals. Keep it under 200 words."
        )),
        HumanMessage(content=f"Conversation to summarize:\n{to_summarize}"),
    ]

    try:
        response     = llm.invoke(summary_request)
        summary_text = safe_get_content(response)
        summary_msg  = SystemMessage(content=f"📝 [PREVIOUS CONTEXT SUMMARY]: {summary_text}")
        new_messages = [system_prompt, summary_msg] + remaining
        logger.info(f"🧹 Pruned {prune_count} messages into summary. New length: {len(new_messages)}")
        return new_messages
    except Exception as e:
        logger.error(f"❌ Summarization failed: {e}")
        return messages  # safe fallback


async def asummarize_oldest(messages: List, llm=None) -> List:
    """Async wrapper around summarize_oldest (runs in a thread)."""
    return await asyncio.to_thread(summarize_oldest, messages, llm)


__all__ = [
    "get_vision_keywords",
    "get_prompts",
    "summarize_oldest",
    "asummarize_oldest",
]
