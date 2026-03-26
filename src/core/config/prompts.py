"""Prompt Management System.

Loads prompts from markdown files in src/core/config/prompts/ directory.
Allows external, versioned prompt management without code changes.

Usage:
    from src.core.config.prompts import load_prompt

    # Load a specific prompt
    prompt = load_prompt("planner")

    # Load with fallback
    prompt = load_prompt("planner", fallback="default mission planner prompt")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PromptsLoader")

# Directory where prompts are stored
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(prompt_name: str, fallback: Optional[str] = None) -> str:
    """Load a prompt from markdown file.

    Args:
        prompt_name: Name of the prompt (e.g., "planner" → planner.md)
        fallback: Fallback text if file not found

    Returns:
        The prompt text from the markdown file, or fallback if not found.

    Raises:
        FileNotFoundError: If prompt file not found and no fallback provided.
    """
    prompt_file = PROMPTS_DIR / f"{prompt_name}.md"

    if prompt_file.exists():
        try:
            content = prompt_file.read_text(encoding="utf-8").strip()
            logger.debug(f"✅ Loaded prompt: {prompt_name}")
            return content
        except Exception as e:
            logger.error(f"Failed to read prompt {prompt_name}: {e}")
            if fallback is not None:
                logger.warning(f"Using fallback for {prompt_name}")
                return fallback
            raise

    # File doesn't exist
    if fallback is not None:
        logger.warning(f"Prompt file not found: {prompt_file} — using fallback")
        return fallback

    raise FileNotFoundError(f"Prompt not found: {prompt_file}")


def list_prompts() -> list[str]:
    """List all available prompts.

    Returns:
        List of prompt names (without .md extension).
    """
    if not PROMPTS_DIR.exists():
        return []

    return sorted([
        f.stem
        for f in PROMPTS_DIR.glob("*.md")
        if f.is_file()
    ])


def prompt_exists(prompt_name: str) -> bool:
    """Check if a prompt exists.

    Args:
        prompt_name: Name of the prompt (without .md)

    Returns:
        True if prompt file exists, False otherwise.
    """
    return (PROMPTS_DIR / f"{prompt_name}.md").exists()
