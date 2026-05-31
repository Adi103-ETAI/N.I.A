"""Simple usage and cost aggregation."""

from __future__ import annotations

from typing import Callable

from niaharness.api.usage import UsageSnapshot


class CostTracker:
    """Accumulate usage and estimated cost over the lifetime of a session.

    Supports an optional cost_per_token callback for USD budget enforcement.
    """

    def __init__(
        self,
        cost_per_token_fn: Callable[[UsageSnapshot], float] | None = None,
    ) -> None:
        self._usage = UsageSnapshot()
        self._cost_per_token_fn = cost_per_token_fn
        self._total_cost_usd: float = 0.0

    def add(self, usage: UsageSnapshot) -> None:
        """Add a usage snapshot to the running total."""
        self._usage = UsageSnapshot(
            input_tokens=self._usage.input_tokens + usage.input_tokens,
            output_tokens=self._usage.output_tokens + usage.output_tokens,
        )
        if self._cost_per_token_fn is not None:
            self._total_cost_usd += self._cost_per_token_fn(usage)

    @property
    def total(self) -> UsageSnapshot:
        """Return the aggregated usage."""
        return self._usage

    @property
    def total_cost_usd(self) -> float:
        """Return the total estimated cost in USD.

        Returns 0.0 if no cost_per_token_fn was provided.
        """
        return self._total_cost_usd
