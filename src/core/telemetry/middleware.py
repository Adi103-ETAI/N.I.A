"""Token counting middleware — tracks LLM token usage per subagent."""
from __future__ import annotations
import logging
import threading

logger = logging.getLogger("NIA.Telemetry.Tokens")

class TokenCounter:
    """Thread-safe token counter keyed by agent_id."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def record(self, agent_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for an agent."""
        with self._lock:
            if agent_id not in self._counts:
                self._counts[agent_id] = {"prompt": 0, "completion": 0, "total": 0}
            entry = self._counts[agent_id]
            entry["prompt"] += prompt_tokens
            entry["completion"] += completion_tokens
            entry["total"] += prompt_tokens + completion_tokens

    def get_usage(self, agent_id: str) -> dict[str, int]:
        """Get token usage for a specific agent."""
        with self._lock:
            return dict(self._counts.get(agent_id, {"prompt": 0, "completion": 0, "total": 0}))

    def get_all_usage(self) -> dict[str, dict[str, int]]:
        """Get usage for all agents."""
        with self._lock:
            return {k: dict(v) for k, v in self._counts.items()}

    def get_mission_total(self) -> dict[str, int]:
        """Get total token usage across all agents."""
        with self._lock:
            total = {"prompt": 0, "completion": 0, "total": 0}
            for entry in self._counts.values():
                total["prompt"] += entry["prompt"]
                total["completion"] += entry["completion"]
                total["total"] += entry["total"]
            return total

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._counts.clear()

_counter: TokenCounter | None = None

def get_token_counter() -> TokenCounter:
    """Get or create the global token counter."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter

__all__ = ["TokenCounter", "get_token_counter"]
