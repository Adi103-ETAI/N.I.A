"""Insights — usage analytics, cost estimation, and terminal formatter.

Ported from the reference project's agent/insights.py (946 lines),
providing a system for tracking and reporting on NIA's usage:

  - **Token aggregation** — total input/output tokens per session/day/provider
  - **Cost estimation** — per-model pricing applied to token counts
  - **Breakdowns** — by model, by provider, by day, by session
  - **Terminal formatter** — pretty-printed tables for the ``/insights`` command
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-3-opus": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3.5-sonnet": (3.0, 15.0),
    "claude-3.5-haiku": (0.8, 4.0),
    "claude-4-opus": (15.0, 75.0),
    "claude-4-sonnet": (3.0, 15.0),
    "gpt-4": (30.0, 60.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-5": (25.0, 50.0),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3": (15.0, 60.0),
    "o3-mini": (3.0, 12.0),
    "o4-mini": (1.1, 4.4),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
    "gemini-2.0-flash": (0.1, 0.4),
    "gemini-2.5-pro": (1.25, 10.0),
    "llama-3.3-70b": (0.59, 0.79),
    "mixtral-8x7b": (0.24, 0.24),
    "grok-2": (2.0, 10.0),
    "grok-3": (3.0, 15.0),
    "deepseek-chat": (0.27, 1.1),
    "deepseek-reasoner": (0.55, 2.19),
    "_default": (3.0, 15.0),
}


def _get_model_pricing(model: str) -> Tuple[float, float]:
    """Return (input_per_1m, output_per_1m) for a model name."""
    if not model:
        return _MODEL_PRICING["_default"]
    model_lower = model.lower()
    if model_lower in _MODEL_PRICING:
        return _MODEL_PRICING[model_lower]
    for pattern in sorted(_MODEL_PRICING.keys(), key=len, reverse=True):
        if pattern == "_default":
            continue
        if model_lower.startswith(pattern):
            return _MODEL_PRICING[pattern]
    return _MODEL_PRICING["_default"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost in USD for a model call."""
    input_per_1m, output_per_1m = _get_model_pricing(model)
    return (input_tokens / 1_000_000) * input_per_1m + (output_tokens / 1_000_000) * output_per_1m


@dataclass
class UsageBreakdown:
    """A single row in a usage breakdown."""
    key: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0


@dataclass
class InsightsReport:
    """Aggregated insights report."""
    period_start: datetime
    period_end: datetime
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_requests: int = 0
    total_sessions: int = 0
    by_model: List[UsageBreakdown] = field(default_factory=list)
    by_provider: List[UsageBreakdown] = field(default_factory=list)
    by_day: List[UsageBreakdown] = field(default_factory=list)
    by_session: List[UsageBreakdown] = field(default_factory=list)


class InsightsEngine:
    """Aggregate usage insights from the session DB."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path

    def _get_db_path(self) -> str:
        if self._db_path:
            return self._db_path
        from niaharness.profiles import get_profile_sessions_db_path
        return str(get_profile_sessions_db_path())

    def get_report(self, *, days: int = 7) -> InsightsReport:
        """Get an insights report for the last N days."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)

        try:
            import sqlite3
            conn = sqlite3.connect(self._get_db_path())
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            logger.warning("Insights: cannot open session DB: %s", exc)
            return InsightsReport(period_start=start, period_end=now)

        try:
            sessions = conn.execute(
                """
                SELECT id, model, provider, message_count, token_count, created_at, updated_at
                FROM sessions
                WHERE updated_at >= ?
                ORDER BY updated_at DESC
                """,
                (start.isoformat(),),
            ).fetchall()

            by_model: Dict[str, UsageBreakdown] = defaultdict(UsageBreakdown)
            by_provider: Dict[str, UsageBreakdown] = defaultdict(UsageBreakdown)
            by_day: Dict[str, UsageBreakdown] = defaultdict(UsageBreakdown)
            by_session: Dict[str, UsageBreakdown] = defaultdict(UsageBreakdown)

            total_input = 0
            total_output = 0
            total_cost = 0.0
            total_requests = 0

            for row in sessions:
                model = row["model"] or "unknown"
                provider = row["provider"] or "unknown"
                total_tokens = row["token_count"] or 0
                est_input = int(total_tokens * 0.7)
                est_output = total_tokens - est_input
                cost = estimate_cost(model, est_input, est_output)
                msg_count = row["message_count"] or 0

                for bucket, key in [
                    (by_model, model), (by_provider, provider),
                    (by_session, row["id"]),
                ]:
                    b = bucket[key]
                    b.key = key
                    b.input_tokens += est_input
                    b.output_tokens += est_output
                    b.total_tokens += total_tokens
                    b.cost_usd += cost
                    b.request_count += msg_count

                day = (row["updated_at"] or "")[:10]
                if day:
                    b = by_day[day]
                    b.key = day
                    b.input_tokens += est_input
                    b.output_tokens += est_output
                    b.total_tokens += total_tokens
                    b.cost_usd += cost
                    b.request_count += msg_count

                total_input += est_input
                total_output += est_output
                total_cost += cost
                total_requests += msg_count

            return InsightsReport(
                period_start=start,
                period_end=now,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_tokens=total_input + total_output,
                total_cost_usd=total_cost,
                total_requests=total_requests,
                total_sessions=len(sessions),
                by_model=sorted(by_model.values(), key=lambda b: b.cost_usd, reverse=True),
                by_provider=sorted(by_provider.values(), key=lambda b: b.cost_usd, reverse=True),
                by_day=sorted(by_day.values(), key=lambda b: b.key),
                by_session=sorted(by_session.values(), key=lambda b: b.cost_usd, reverse=True),
            )
        finally:
            conn.close()


def format_insights(report: InsightsReport) -> str:
    """Format an insights report for terminal display."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"NIA Insights - {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Totals:")
    lines.append(f"  Sessions:      {report.total_sessions}")
    lines.append(f"  Requests:      {report.total_requests}")
    lines.append(f"  Input tokens:  {report.total_input_tokens:,}")
    lines.append(f"  Output tokens: {report.total_output_tokens:,}")
    lines.append(f"  Total tokens:  {report.total_tokens:,}")
    lines.append(f"  Est. cost:     ${report.total_cost_usd:.4f} USD")
    lines.append("")

    if report.by_model:
        lines.append("By Model:")
        lines.append(f"  {'Model':<30} {'Tokens':>12} {'Cost':>10}")
        lines.append(f"  {'-' * 30} {'-' * 12} {'-' * 10}")
        for b in report.by_model[:10]:
            lines.append(f"  {b.key:<30} {b.total_tokens:>12,} ${b.cost_usd:>8.4f}")
        lines.append("")

    if report.by_provider:
        lines.append("By Provider:")
        lines.append(f"  {'Provider':<20} {'Tokens':>12} {'Cost':>10}")
        lines.append(f"  {'-' * 20} {'-' * 12} {'-' * 10}")
        for b in report.by_provider[:10]:
            lines.append(f"  {b.key:<20} {b.total_tokens:>12,} ${b.cost_usd:>8.4f}")
        lines.append("")

    if report.by_day:
        lines.append("By Day:")
        lines.append(f"  {'Date':<12} {'Tokens':>12} {'Cost':>10}")
        lines.append(f"  {'-' * 12} {'-' * 12} {'-' * 10}")
        for b in report.by_day:
            lines.append(f"  {b.key:<12} {b.total_tokens:>12,} ${b.cost_usd:>8.4f}")
        lines.append("")

    if report.by_session:
        lines.append("Top Sessions:")
        lines.append(f"  {'Session ID':<20} {'Tokens':>12} {'Cost':>10}")
        lines.append(f"  {'-' * 20} {'-' * 12} {'-' * 10}")
        for b in report.by_session[:5]:
            lines.append(f"  {b.key[:20]:<20} {b.total_tokens:>12,} ${b.cost_usd:>8.4f}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "InsightsEngine",
    "InsightsReport",
    "UsageBreakdown",
    "estimate_cost",
    "format_insights",
]
