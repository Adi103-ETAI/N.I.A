"""Context Wormhole — Event-bus wiring between subagents and the Coordinator.

Connects subagent observations to the Coordinator's shared context log via
the AsyncEventBus, ensuring decoupled communication without direct imports
between subagent wrappers and the Coordinator graph.

Architecture::

    invoke_tara / invoke_iris
        │
        ▼  emit_observation(agent_id, observation, tags)
    AsyncEventBus  ──"context_observation"──►  ContextWormhole.on_observation()
        │                                              │
        │                                              ▼
        │                                     _observations (list[dict])
        │
        ▼  get_subagent_context(wormhole, intent)
    Condensed summary string injected into subagent prompts

Public API:
    emit_observation    — async helper called by subagent wrappers after a result.
    ContextWormhole     — Coordinator-side listener accumulating cross-agent obs.
    get_subagent_context — Builds a focused context block for subagent prompts.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List

logger = logging.getLogger("NIA.ContextWormhole")


# ============================================================================
# emit_observation — Subagent-side helper
# ============================================================================

async def emit_observation(
    agent_id: str,
    observation: str,
    relevance_tags: List[str] | None = None,
) -> None:
    """Create a ContextObservation and emit it on the bus.

    This is the function that subagent wrappers (invoke_tara, invoke_iris)
    call after receiving a SubagentResult.  It publishes a
    ``"context_observation"`` event carrying the observation as a dict.

    Args:
        agent_id: Identifier of the subagent that produced the observation.
        observation: Free-text summary of what the subagent observed or learned.
        relevance_tags: Optional tags indicating related plan steps or topics.
    """
    # Lazy imports to avoid circular dependencies at module load time.
    from src.core.schema.coordinator import ContextObservation
    from src.core.bus.events import get_event_bus

    obs = ContextObservation(
        agent_id=agent_id,
        observation=observation,
        relevance_tags=relevance_tags or [],
    )

    bus = get_event_bus()
    try:
        await bus.emit("context_observation", obs.model_dump())
        logger.debug(
            "Emitted context_observation from '%s': %.80s",
            agent_id,
            observation,
        )
    except Exception:
        # Event bus errors are non-critical — never crash the subagent.
        logger.warning(
            "Failed to emit context_observation from '%s'",
            agent_id,
            exc_info=True,
        )


# ============================================================================
# ContextWormhole — Coordinator-side listener
# ============================================================================

class ContextWormhole:
    """Coordinator-side listener that accumulates cross-agent observations.

    The Coordinator creates one instance per mission and subscribes it
    to the event bus.  On mission completion, it unsubscribes.

    Observations are stored as plain dicts (via ``model_dump()``) to avoid
    carrying Pydantic objects through the LangGraph state boundary.

    Thread-safety:
        Uses ``asyncio.Lock`` for safe accumulation when multiple subagents
        emit concurrently within the same event loop.
    """

    def __init__(self, mission_id: str, max_observations: int = 50) -> None:
        self._mission_id = mission_id
        self._observations: list[dict] = []
        self._lock = asyncio.Lock()
        self._max_observations = max_observations
        self._active = True

    # ------------------------------------------------------------------
    # Event bus callback
    # ------------------------------------------------------------------

    async def on_observation(self, data: Any) -> None:
        """Event bus callback for ``"context_observation"`` events.

        Validates the incoming data, converts Pydantic models to dicts
        if needed, and appends to the internal observation list.  When
        the list exceeds ``max_observations``, the oldest entries are
        dropped (FIFO).

        Args:
            data: A dict or ContextObservation instance emitted by
                :func:`emit_observation`.
        """
        if not self._active:
            return

        # Normalise to dict
        if data is None:
            logger.debug("on_observation received None — ignoring.")
            return

        if isinstance(data, dict):
            obs_dict = data
        elif hasattr(data, "model_dump"):
            # Pydantic model — convert to dict
            obs_dict = data.model_dump()
        else:
            logger.warning(
                "on_observation received unexpected type %s — ignoring.",
                type(data).__name__,
            )
            return

        async with self._lock:
            self._observations.append(obs_dict)

            # FIFO cap — drop oldest when over the limit
            if len(self._observations) > self._max_observations:
                overflow = len(self._observations) - self._max_observations
                self._observations = self._observations[overflow:]
                logger.debug(
                    "Observation buffer capped: dropped %d oldest entries.",
                    overflow,
                )

        logger.debug(
            "Wormhole[%s] received observation from '%s' (total=%d).",
            self._mission_id,
            obs_dict.get("agent_id", "?"),
            len(self._observations),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_observations(self) -> list[dict]:
        """Return a shallow copy of all accumulated observations."""
        return list(self._observations)

    def get_condensed_summary(self, max_items: int = 10) -> str:
        """Build a condensed text summary of the most recent observations.

        This is what gets injected into subagent context -- not the full
        N.I.A. conversation history.  Keeps subagent prompts focused and
        within token budgets.

        Format::

            [agent_id] observation_text (tag1, tag2)
            [agent_id] observation_text (tag1)
            ...

        Args:
            max_items: Maximum number of recent observations to include.

        Returns:
            A multi-line string summary, or an empty string if no
            observations have been collected yet.
        """
        if not self._observations:
            return ""

        # Take the most recent entries
        recent = self._observations[-max_items:]
        lines: list[str] = []

        for obs in recent:
            agent = obs.get("agent_id", "unknown")
            text = obs.get("observation", "")
            tags = obs.get("relevance_tags", [])

            # Truncate long observations to keep the summary tight
            if len(text) > 200:
                text = text[:197] + "..."

            tag_suffix = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"[{agent}] {text}{tag_suffix}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Subscribe to the event bus for ``context_observation`` events.

        Should be called once when the Coordinator starts a mission.
        """
        from src.core.bus.events import get_event_bus

        bus = get_event_bus()
        bus.subscribe("context_observation", self.on_observation)
        self._active = True
        logger.info(
            "Wormhole[%s] subscribed to 'context_observation'.",
            self._mission_id,
        )

    def unsubscribe(self) -> None:
        """Unsubscribe from the event bus.  Called on mission end.

        Since ``AsyncEventBus`` does not currently support listener
        removal, we deactivate via the ``_active`` flag so that
        ``on_observation`` becomes a no-op for this instance.
        """
        self._active = False
        logger.info(
            "Wormhole[%s] deactivated (observations=%d).",
            self._mission_id,
            len(self._observations),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mission_id(self) -> str:
        """The mission ID this wormhole is bound to."""
        return self._mission_id

    @property
    def observation_count(self) -> int:
        """Number of observations currently held."""
        return len(self._observations)

    @property
    def active(self) -> bool:
        """Whether this wormhole is still listening for events."""
        return self._active


# ============================================================================
# get_subagent_context — Context builder for subagent prompts
# ============================================================================

def get_subagent_context(
    wormhole: ContextWormhole | None,
    mission_intent: str,
) -> str:
    """Build a focused context block for subagent prompts.

    Combines the mission intent with a condensed summary of cross-agent
    observations so that subagents have situational awareness without
    being overwhelmed by the full N.I.A. conversation history.

    Args:
        wormhole: The active ContextWormhole for the current mission,
            or ``None`` if context sharing is not available.
        mission_intent: The high-level mission objective string.

    Returns:
        A formatted string suitable for prepending to subagent objectives.
        If *wormhole* is ``None`` or has no observations, returns just the
        mission intent section.
    """
    sections: list[str] = []

    # Always include mission intent
    sections.append(f"=== Mission Objective ===\n{mission_intent}")

    # Append cross-agent observations if available
    if wormhole is not None:
        summary = wormhole.get_condensed_summary()
        if summary:
            sections.append(
                f"=== Cross-Agent Observations ({wormhole.observation_count} total) ===\n"
                f"{summary}"
            )

    return "\n\n".join(sections)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "emit_observation",
    "ContextWormhole",
    "get_subagent_context",
]
