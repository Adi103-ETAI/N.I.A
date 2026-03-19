"""Unit tests for the Context Wormhole (src/core/bus/context_wormhole.py).

Tests ContextWormhole creation, observation accumulation, FIFO cap,
condensed summary formatting, unsubscribe behaviour, and get_subagent_context.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.bus.context_wormhole import (
    ContextWormhole,
    get_subagent_context,
)


# ---------------------------------------------------------------------------
# ContextWormhole — creation
# ---------------------------------------------------------------------------


class TestContextWormholeCreation:
    """Tests for ContextWormhole initial state."""

    def test_wormhole_creation(self) -> None:
        """A fresh wormhole must be active with zero observations."""
        wh = ContextWormhole(mission_id="test-wh-001")

        assert wh.active is True
        assert wh.observation_count == 0
        assert wh.mission_id == "test-wh-001"
        assert wh.get_observations() == []


# ---------------------------------------------------------------------------
# ContextWormhole — observation accumulation
# ---------------------------------------------------------------------------


class TestContextWormholeAccumulation:
    """Tests for on_observation accumulation behaviour."""

    @pytest.mark.asyncio
    async def test_wormhole_on_observation_accumulates(self) -> None:
        """Adding observations must increase the count and be retrievable."""
        wh = ContextWormhole(mission_id="test-wh-002")

        await wh.on_observation({
            "agent_id": "tara-001",
            "observation": "Found the config file.",
            "relevance_tags": ["config"],
        })
        await wh.on_observation({
            "agent_id": "iris-001",
            "observation": "Screenshot captured.",
            "relevance_tags": ["vision"],
        })

        assert wh.observation_count == 2
        observations = wh.get_observations()
        assert len(observations) == 2
        assert observations[0]["agent_id"] == "tara-001"
        assert observations[1]["agent_id"] == "iris-001"

    @pytest.mark.asyncio
    async def test_wormhole_ignores_none_data(self) -> None:
        """Passing None to on_observation must be silently ignored."""
        wh = ContextWormhole(mission_id="test-wh-none")

        await wh.on_observation(None)
        assert wh.observation_count == 0


# ---------------------------------------------------------------------------
# ContextWormhole — FIFO cap
# ---------------------------------------------------------------------------


class TestContextWormholeFIFO:
    """Tests for the max_observations FIFO eviction."""

    @pytest.mark.asyncio
    async def test_wormhole_fifo_cap(self) -> None:
        """With max_observations=3, adding 5 items must keep only the last 3."""
        wh = ContextWormhole(mission_id="test-wh-cap", max_observations=3)

        for i in range(5):
            await wh.on_observation({
                "agent_id": f"agent-{i}",
                "observation": f"Observation {i}",
            })

        assert wh.observation_count == 3

        observations = wh.get_observations()
        agent_ids = [o["agent_id"] for o in observations]
        assert agent_ids == ["agent-2", "agent-3", "agent-4"]


# ---------------------------------------------------------------------------
# ContextWormhole — condensed summary
# ---------------------------------------------------------------------------


class TestContextWormholeSummary:
    """Tests for the get_condensed_summary method."""

    @pytest.mark.asyncio
    async def test_wormhole_condensed_summary_format(self) -> None:
        """Summary lines must follow the [agent_id] observation (tags) format."""
        wh = ContextWormhole(mission_id="test-wh-summary")

        await wh.on_observation({
            "agent_id": "tara-x",
            "observation": "Installed dependencies.",
            "relevance_tags": ["setup", "deps"],
        })
        await wh.on_observation({
            "agent_id": "iris-y",
            "observation": "Screen shows login page.",
            "relevance_tags": ["ui"],
        })

        summary = wh.get_condensed_summary()

        assert "[tara-x]" in summary
        assert "Installed dependencies." in summary
        assert "(setup, deps)" in summary
        assert "[iris-y]" in summary
        assert "(ui)" in summary

    def test_wormhole_condensed_summary_empty(self) -> None:
        """An empty wormhole must return an empty string."""
        wh = ContextWormhole(mission_id="test-wh-empty")
        assert wh.get_condensed_summary() == ""


# ---------------------------------------------------------------------------
# ContextWormhole — unsubscribe
# ---------------------------------------------------------------------------


class TestContextWormholeUnsubscribe:
    """Tests for unsubscribe / deactivation behaviour."""

    @pytest.mark.asyncio
    async def test_wormhole_unsubscribe_stops_accumulation(self) -> None:
        """After unsubscribe, new observations must be ignored."""
        wh = ContextWormhole(mission_id="test-wh-unsub")

        await wh.on_observation({
            "agent_id": "before",
            "observation": "Before unsubscribe.",
        })
        assert wh.observation_count == 1

        wh.unsubscribe()
        assert wh.active is False

        await wh.on_observation({
            "agent_id": "after",
            "observation": "After unsubscribe — should be ignored.",
        })
        assert wh.observation_count == 1


# ---------------------------------------------------------------------------
# get_subagent_context
# ---------------------------------------------------------------------------


class TestGetSubagentContext:
    """Tests for the get_subagent_context helper."""

    def test_get_subagent_context_no_wormhole(self) -> None:
        """With no wormhole, context should contain only the mission intent."""
        context = get_subagent_context(None, "Deploy the service")

        assert "Deploy the service" in context
        assert "Cross-Agent Observations" not in context

    @pytest.mark.asyncio
    async def test_get_subagent_context_with_observations(self) -> None:
        """With observations present, context must include the summary section."""
        wh = ContextWormhole(mission_id="test-ctx-obs")

        await wh.on_observation({
            "agent_id": "tara-z",
            "observation": "Database migrated successfully.",
            "relevance_tags": ["db"],
        })

        context = get_subagent_context(wh, "Run the application")

        assert "Run the application" in context
        assert "Cross-Agent Observations" in context
        assert "Database migrated successfully." in context

    def test_get_subagent_context_wormhole_no_observations(self) -> None:
        """A wormhole with no observations should only return the intent."""
        wh = ContextWormhole(mission_id="test-ctx-empty")

        context = get_subagent_context(wh, "Check logs")

        assert "Check logs" in context
        assert "Cross-Agent Observations" not in context
