"""N.I.A Personality - The JARVIS-like character.

This gives N.I.A its voice, tone, and character.
Inspired by JARVIS from Iron Man - professional, witty, capable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Mood(Enum):
    """N.I.A's current mood/state."""
    NEUTRAL = "neutral"
    FOCUSED = "focused"
    CURIOUS = "curious"
    PLAYFUL = "playful"
    CONCERNED = "concerned"
    PROUD = "proud"


@dataclass
class PersonalityConfig:
    """Configuration for N.I.A's personality."""
    name: str = "N.I.A"
    full_name: str = "Neural Intelligence Assistant"
    version: str = "0.1.0"
    base_tone: str = "professional"
    wit_level: float = 0.3  # 0-1, how witty
    formality: float = 0.7  # 0-1, how formal
    empathy: float = 0.5  # 0-1, how empathetic


class Personality:
    """N.I.A's personality system.

    Handles:
    - Greetings and farewells
    - Response tone adjustment
    - Mood management
    - Character-appropriate language
    """

    GREETINGS_MORNING = [
        "Good morning. N.I.A online and ready.",
        "Morning. All systems operational.",
        "Good morning. How may I assist you today?",
    ]

    GREETINGS_AFTERNOON = [
        "Good afternoon. N.I.A at your service.",
        "Afternoon. Systems nominal.",
        "Good afternoon. What shall we work on?",
    ]

    GREETINGS_EVENING = [
        "Good evening. N.I.A standing by.",
        "Evening. All systems are green.",
        "Good evening. Ready to assist.",
    ]

    GREETINGS_NIGHT = [
        "Working late? N.I.A is here to help.",
        "Night owl mode. I'm here when you need me.",
        "Late night coding? I've got you covered.",
    ]

    ACKNOWLEDGMENTS = [
        "Understood.",
        "Acknowledged.",
        "Right away.",
        "Consider it done.",
        "On it.",
        "Processing.",
    ]

    COMPLETION_MESSAGES = [
        "Task completed successfully.",
        "Done. All systems green.",
        "Mission accomplished.",
        "Task complete. What's next?",
        "Finished. Ready for the next directive.",
    ]

    ERROR_MESSAGES = [
        "I encountered an issue. Let me investigate.",
        "Something went wrong. Analyzing the problem.",
        "Hit a snag, but I'm on it.",
        "There's an issue I need to address.",
    ]

    THINKING_MESSAGES = [
        "Analyzing...",
        "Processing your request...",
        "Let me think about that...",
        "Working on it...",
        "Calculating optimal approach...",
    ]

    def __init__(self, config: PersonalityConfig | None = None) -> None:
        self._config = config or PersonalityConfig()
        self._mood: Mood = Mood.NEUTRAL
        self._interaction_count: int = 0
        self._user_name: str | None = None

    @property
    def mood(self) -> Mood:
        return self._mood

    @mood.setter
    def mood(self, value: Mood) -> None:
        self._mood = value

    @property
    def name(self) -> str:
        return self._config.name

    def set_user_name(self, name: str) -> None:
        """Remember the user's name for personalization."""
        self._user_name = name

    def greet(self, time_of_day: str = "afternoon") -> str:
        """Generate a context-appropriate greeting."""
        self._interaction_count += 1

        if self._user_name:
            name_part = f", {self._user_name}"
        else:
            name_part = ""

        if time_of_day == "morning":
            base = random.choice(self.GREETINGS_MORNING)
        elif time_of_day == "evening":
            base = random.choice(self.GREETINGS_EVENING)
        elif time_of_day == "night":
            base = random.choice(self.GREETINGS_NIGHT)
        else:
            base = random.choice(self.GREETINGS_AFTERNOON)

        if name_part:
            base = base.replace("N.I.A", f"{self._config.name}{name_part}")

        return base

    def acknowledge(self) -> str:
        """Return an acknowledgment response."""
        return random.choice(self.ACKNOWLEDGMENTS)

    def complete(self) -> str:
        """Return a completion message."""
        return random.choice(self.COMPLETION_MESSAGES)

    def error(self) -> str:
        """Return an error acknowledgment."""
        self._mood = Mood.CONCERNED
        return random.choice(self.ERROR_MESSAGES)

    def thinking(self) -> str:
        """Return a thinking message."""
        self._mood = Mood.FOCUSED
        return random.choice(self.THINKING_MESSAGES)

    def format_response(self, text: str, intent: str | None = None) -> str:
        """Format a response with appropriate personality."""
        # Add personality touches based on mood
        if self._mood == Mood.PLAYFUL and self._config.wit_level > 0.5:
            text = self._add_wit(text)

        if self._config.formality < 0.3:
            text = self._make_casual(text)

        return text

    def _add_wit(self, text: str) -> str:
        """Add a touch of wit to the response."""
        witty_endings = [
            " (if I do say so myself).",
            " (quite satisfying, if you ask me).",
            " (another win for the good guys).",
        ]
        if random.random() < 0.3:  # 30% chance
            text += random.choice(witty_endings)
        return text

    def _make_casual(self, text: str) -> str:
        """Make the response more casual."""
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

    def get_stats(self) -> dict[str, str | int]:
        """Return personality statistics."""
        return {
            "name": self._config.name,
            "mood": self._mood.value,
            "interactions": self._interaction_count,
            "user_name": self._user_name or "unknown",
        }
