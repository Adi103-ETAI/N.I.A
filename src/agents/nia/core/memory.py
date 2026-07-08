"""N.I.A Memory - Short-term and long-term memory systems.

Handles conversation history, user preferences, and learned patterns.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    category: str  # "conversation", "preference", "pattern", "fact"
    timestamp: float = field(default_factory=time.time)
    relevance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Memory:
    """N.I.A's memory system.

    Short-term: Recent conversation turns
    Long-term: User preferences, learned patterns, important facts
    """

    def __init__(
        self,
        max_short_term: int = 20,
        max_long_term: int = 100,
        storage_path: Path | None = None,
    ) -> None:
        self._short_term: list[MemoryEntry] = []
        self._long_term: list[MemoryEntry] = []
        self._max_short_term = max_short_term
        self._max_long_term = max_long_term
        self._storage_path = storage_path
        self._loaded = False

    def add_conversation(self, role: str, content: str) -> None:
        """Add a conversation turn to short-term memory."""
        entry = MemoryEntry(
            content=content,
            category="conversation",
            metadata={"role": role},
        )
        self._short_term.append(entry)

        # Trim if over limit
        if len(self._short_term) > self._max_short_term:
            # Promote important conversations to long-term
            old = self._short_term.pop(0)
            if self._is_important(old):
                self._long_term.append(old)
                self._trim_long_term()

    def add_preference(self, key: str, value: str) -> None:
        """Store a user preference."""
        entry = MemoryEntry(
            content=f"{key}: {value}",
            category="preference",
            metadata={"key": key, "value": value},
        )
        # Remove existing preference with same key
        self._long_term = [
            m for m in self._long_term
            if not (m.category == "preference" and m.metadata.get("key") == key)
        ]
        self._long_term.append(entry)

    def add_pattern(self, description: str, frequency: int = 1) -> None:
        """Store a learned pattern."""
        entry = MemoryEntry(
            content=description,
            category="pattern",
            metadata={"frequency": frequency},
        )
        self._long_term.append(entry)
        self._trim_long_term()

    def add_fact(self, fact: str) -> None:
        """Store an important fact."""
        entry = MemoryEntry(
            content=fact,
            category="fact",
        )
        self._long_term.append(entry)
        self._trim_long_term()

    def search_relevant(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search for relevant memories."""
        query_lower = query.lower()
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._short_term + self._long_term:
            score = self._relevance_score(entry, query_lower)
            if score > 0.1:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def get_recent_conversation(self, limit: int = 10) -> list[MemoryEntry]:
        """Get recent conversation turns."""
        return self._short_term[-limit:]

    def get_preferences(self) -> dict[str, str]:
        """Get all stored preferences."""
        prefs = {}
        for entry in self._long_term:
            if entry.category == "preference":
                key = entry.metadata.get("key", "")
                value = entry.metadata.get("value", "")
                if key:
                    prefs[key] = value
        return prefs

    def get_context_summary(self) -> str:
        """Get a summary of current memory context."""
        conv_count = len(self._short_term)
        pref_count = sum(1 for m in self._long_term if m.category == "preference")
        pattern_count = sum(1 for m in self._long_term if m.category == "pattern")

        parts = [f"Remembering {conv_count} recent exchanges"]
        if pref_count:
            parts.append(f"{pref_count} user preferences")
        if pattern_count:
            parts.append(f"{pattern_count} learned patterns")

        return ", ".join(parts)

    def _relevance_score(self, entry: MemoryEntry, query_lower: str) -> float:
        """Calculate relevance score for an entry."""
        content_lower = entry.content.lower()
        words = set(query_lower.split())
        content_words = set(content_lower.split())

        overlap = len(words & content_words)
        if not words:
            return 0.0

        base_score = overlap / len(words)

        # Boost recent entries
        age = time.time() - entry.timestamp
        recency_boost = max(0, 1 - (age / 3600))  # Decay over 1 hour

        return base_score * 0.7 + recency_boost * 0.3

    def _is_important(self, entry: MemoryEntry) -> bool:
        """Determine if an entry is important enough for long-term storage."""
        important_keywords = ["prefer", "always", "never", "important", "remember"]
        return any(kw in entry.content.lower() for kw in important_keywords)

    def _trim_long_term(self) -> None:
        """Trim long-term memory to max size."""
        if len(self._long_term) > self._max_long_term:
            # Keep preferences and facts, trim patterns
            prefs = [m for m in self._long_term if m.category in ("preference", "fact")]
            patterns = [m for m in self._long_term if m.category == "pattern"]

            # Keep most recent patterns
            patterns.sort(key=lambda x: x.timestamp, reverse=True)
            max_patterns = self._max_long_term - len(prefs)

            self._long_term = prefs + patterns[:max_patterns]

    def save(self, path: Path | None = None) -> None:
        """Save long-term memory to disk."""
        save_path = path or self._storage_path
        if not save_path:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "long_term": [
                {
                    "content": m.content,
                    "category": m.category,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self._long_term
            ]
        }
        save_path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path | None = None) -> None:
        """Load long-term memory from disk."""
        load_path = path or self._storage_path
        if not load_path or not load_path.exists():
            return

        data = json.loads(load_path.read_text())
        self._long_term = [
            MemoryEntry(
                content=item["content"],
                category=item["category"],
                timestamp=item.get("timestamp", 0),
                metadata=item.get("metadata", {}),
            )
            for item in data.get("long_term", [])
        ]
        self._loaded = True

    def clear_short_term(self) -> None:
        """Clear short-term memory (new session)."""
        self._short_term.clear()

    def get_stats(self) -> dict[str, int]:
        """Return memory statistics."""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "total_memories": len(self._short_term) + len(self._long_term),
        }

    def get_summary_for_prompt(self, max_entries: int = 10) -> str:
        """Return a markdown summary of long-term memory for the system prompt.

        P2 fix: unifies the two memory systems. The system prompt builder
        can call this to inject NIA's memory (preferences, facts, patterns)
        into the prompt, alongside the project-scoped markdown memory files
        from ``niaharness/memory/``. This gives the model a single coherent
        view of both global (user-level) and local (project-level) memory.

        Args:
            max_entries: Maximum number of entries to include (most recent first).

        Returns:
            A markdown-formatted string suitable for the system prompt.
        """
        if not self._long_term:
            return ""

        # Group by category.
        by_category: dict[str, list[MemoryEntry]] = {}
        for entry in self._long_term:
            by_category.setdefault(entry.category, []).append(entry)

        lines: list[str] = ["## NIA Memory (global)"]

        if "preference" in by_category:
            lines.append("\n### Preferences")
            for entry in by_category["preference"][-5:]:  # Last 5 prefs
                lines.append(f"- {entry.content}")

        if "fact" in by_category:
            lines.append("\n### Facts")
            for entry in by_category["fact"][-5:]:
                lines.append(f"- {entry.content}")

        if "pattern" in by_category:
            lines.append("\n### Patterns")
            for entry in by_category["pattern"][-3:]:
                lines.append(f"- {entry.content}")

        return "\n".join(lines)

    def export_to_markdown(self, output_path: Path) -> Path:
        """Export long-term memory to a markdown file.

        P2 fix: bridges the JSON-based Memory system to the markdown-based
        niaharness/memory/ system. Writes a ``nia-memory.md`` file in the
        project memory directory so both systems can be read from a single
        location by the system prompt.

        Args:
            output_path: The path to write the markdown file to.

        Returns:
            The path that was written to.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.get_summary_for_prompt()
        output_path.write_text(content + "\n", encoding="utf-8")
        return output_path
