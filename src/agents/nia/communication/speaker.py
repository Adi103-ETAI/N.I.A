"""N.I.A Speaker - Generates and formats responses.

Handles response generation, TTS preparation, and output formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ResponseStyle(Enum):
    """Style of response output."""
    BRIEF = "brief"  # Short, direct
    DETAILED = "detailed"  # Full explanation
    CONVERSATIONAL = "conversational"  # Natural dialogue
    TECHNICAL = "technical"  # Technical precision
    CASUAL = "casual"  # Relaxed tone


@dataclass
class Response:
    """A formatted response ready for output."""
    text: str
    style: ResponseStyle
    metadata: dict[str, Any] | None = None


class Speaker:
    """Generates and formats N.I.A's responses.

    Responsibilities:
    - Format responses for different output styles
    - Handle multi-line responses
    - Prepare text for TTS if needed
    - Add personality touches
    """

    def __init__(self) -> None:
        self._default_style = ResponseStyle.CONVERSATIONAL
        self._response_history: list[Response] = []

    def speak(self, text: str, style: ResponseStyle | None = None) -> Response:
        """Generate a response with the specified style."""
        effective_style = style or self._default_style

        formatted = self._format_text(text, effective_style)
        response = Response(text=formatted, style=effective_style)

        self._response_history.append(response)
        return response

    def speak_thinking(self) -> str:
        """Generate a thinking/processing message."""
        messages = [
            "Processing your request...",
            "Let me analyze that...",
            "Working on it...",
            "Calculating optimal approach...",
        ]
        import random
        return random.choice(messages)

    def speak_acknowledgment(self) -> str:
        """Generate an acknowledgment message."""
        messages = [
            "Understood.",
            "Acknowledged.",
            "Right away.",
            "On it.",
        ]
        import random
        return random.choice(messages)

    def speak_completion(self, task: str) -> str:
        """Generate a completion message."""
        return f"Completed: {task}"

    def speak_error(self, error: str) -> str:
        """Generate an error message."""
        return f"I encountered an issue: {error}. Let me investigate."

    def speak_clarification(self, question: str) -> str:
        """Generate a clarification question."""
        return f"I need some clarification: {question}"

    def format_for_tts(self, text: str) -> str:
        """Format text for text-to-speech output."""
        # Remove markdown formatting
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('`', '')

        # Remove code blocks
        import re
        text = re.sub(r'```[\s\S]*?```', 'code block omitted', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Simplify URLs
        text = re.sub(r'https?://\S+', 'link', text)

        return text

    def _format_text(self, text: str, style: ResponseStyle) -> str:
        """Format text according to style."""
        if style == ResponseStyle.BRIEF:
            return self._make_brief(text)
        elif style == ResponseStyle.DETAILED:
            return self._make_detailed(text)
        elif style == ResponseStyle.TECHNICAL:
            return self._make_technical(text)
        elif style == ResponseStyle.CASUAL:
            return self._make_casual(text)
        else:
            return text

    def _make_brief(self, text: str) -> str:
        """Make response brief and direct."""
        # Take first sentence if multiple
        sentences = text.split('. ')
        if len(sentences) > 1:
            return sentences[0] + '.'
        return text

    def _make_detailed(self, text: str) -> str:
        """Add more detail to response."""
        # For now, just return as-is
        # Could add explanations, examples, etc.
        return text

    def _make_technical(self, text: str) -> str:
        """Format for technical precision."""
        # Ensure technical terms are clear
        return text

    def _make_casual(self, text: str) -> str:
        """Make response more casual."""
        replacements = {
            "I will": "I'll",
            "I am": "I'm",
            "cannot": "can't",
            "do not": "don't",
            "will not": "won't",
        }
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        return text

    def get_history(self, limit: int = 10) -> list[Response]:
        """Get recent response history."""
        return self._response_history[-limit:]
