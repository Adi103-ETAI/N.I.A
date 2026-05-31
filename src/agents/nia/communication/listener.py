"""N.I.A Listener - Processes incoming user input.

Handles voice/text input processing and initial parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InputType(Enum):
    """Type of user input."""
    TEXT = "text"
    VOICE = "voice"
    COMMAND = "command"  # Slash commands like /help
    KEYWORD = "keyword"  # Quick keywords like "fix", "run"


@dataclass
class ParsedInput:
    """Parsed user input with metadata."""
    raw: str
    cleaned: str
    input_type: InputType
    command: str | None = None  # For command-type inputs
    args: dict[str, str] | None = None  # Parsed arguments


class Listener:
    """Processes and parses user input.

    Responsibilities:
    - Clean and normalize input
    - Detect input type
    - Parse commands and arguments
    - Handle voice transcription formatting
    """

    COMMAND_PREFIX = "/"
    KEYWORDS = {
        "fix": "debug",
        "run": "execute",
        "test": "verify",
        "help": "assist",
        "stop": "cancel",
        "undo": "revert",
        "redo": "retry",
    }

    def listen(self, raw_input: str) -> ParsedInput:
        """Process raw user input into structured form."""
        cleaned = self._clean_input(raw_input)
        input_type = self._detect_type(cleaned)
        command = None
        args = None

        if input_type == InputType.COMMAND:
            command, args = self._parse_command(cleaned)
        elif input_type == InputType.KEYWORD:
            command = self._map_keyword(cleaned)

        return ParsedInput(
            raw=raw_input,
            cleaned=cleaned,
            input_type=input_type,
            command=command,
            args=args,
        )

    def _clean_input(self, text: str) -> str:
        """Clean and normalize input text."""
        # Strip whitespace
        text = text.strip()

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove trailing punctuation that's not needed
        text = text.rstrip('.')

        return text

    def _detect_type(self, text: str) -> InputType:
        """Detect the type of input."""
        if text.startswith(self.COMMAND_PREFIX):
            return InputType.COMMAND

        # Check for keywords (single word that matches)
        words = text.lower().split()
        if len(words) <= 2 and words[0] in self.KEYWORDS:
            return InputType.KEYWORD

        return InputType.TEXT

    def _parse_command(self, text: str) -> tuple[str, dict[str, str]]:
        """Parse a command and its arguments."""
        parts = text.split()
        command = parts[0].lstrip(self.COMMAND_PREFIX)

        args = {}
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                args[key] = value
            else:
                args[f"arg{len(args)}"] = part

        return command, args

    def _map_keyword(self, text: str) -> str:
        """Map a keyword to its full intent."""
        word = text.lower().split()[0]
        return self.KEYWORDS.get(word, word)

    def format_for_processing(self, parsed: ParsedInput) -> str:
        """Format parsed input for brain processing."""
        if parsed.input_type == InputType.COMMAND:
            return f"Execute command: {parsed.command} with args {parsed.args}"
        elif parsed.input_type == InputType.KEYWORD:
            return f"Perform action: {parsed.command}"
        else:
            return parsed.cleaned
