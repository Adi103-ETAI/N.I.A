"""Insights — usage analytics with real token tracking, cost estimation, and formatters.

Ported from Hermes Agent's ``agent/insights.py`` (921 LOC), adapted to
NIA's session DB schema. Reads REAL token columns (input_tokens,
output_tokens, cache_read_tokens, cache_write_tokens) from Task 1's
upgraded sessions table — no more 70/30 token-split hack.

Provides:

  - :class:`InsightsEngine` — main entry point. Reads from
    :class:`niaharness.services.session_db.SessionDB` and computes:
      - **Overview** — totals (sessions, messages, tool_calls, all token
        buckets, estimated/actual cost, hours, averages).
      - **Model breakdown** — per-model aggregation (sessions, tokens, cost).
      - **Platform breakdown** — per-source aggregation (CLI / Telegram /
        Discord / Slack / cron).
      - **Tool breakdown** — per-tool call counts + percentages (via SQL
        JOIN on messages.tool_name + JSON parse of messages.tool_calls).
      - **Skill breakdown** — per-skill view/manage counts + last_used_at
        (scans assistant-message tool_calls for skill_view / skill_manage).
      - **Activity patterns** — day-of-week / hour-of-day bins, busiest
        day/hour, active_days, max_streak (consecutive-day run).
      - **Top sessions** — longest / most-messages / most-tokens / most-tools.
  - :meth:`InsightsEngine.format_terminal` — Unicode-box CLI report.
  - :meth:`InsightsEngine.format_gateway` — Markdown report for chat delivery.
  - :func:`to_json` — machine-readable JSON export.

The cost estimation uses :mod:`niaharness.insights.usage_pricing` which
has the full Hermes pricing table (39 entries across Anthropic / OpenAI /
DeepSeek / Google / Bedrock / xAI) with cache-read / cache-write support.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from niaharness.insights.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    format_duration_compact,
    format_token_count_compact,
    has_known_pricing,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_cost(
    session_or_model: Any,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[float, str]:
    """Thin wrapper over :func:`estimate_usage_cost`.

    Accepts either a session row dict (reads ``model``, token buckets,
    ``billing_provider``, ``billing_base_url``) or a bare model string.

    Returns ``(float_amount, status_str)`` where ``float_amount`` is ``0.0``
    for unknown-cost sessions (NOT ``None`` — keeps the aggregation simple)
    and ``status_str`` is one of ``"estimated"`` / ``"included"`` /
    ``"unknown"``.
    """
    if isinstance(session_or_model, dict):
        model = session_or_model.get("model") or ""
        input_tokens = int(session_or_model.get("input_tokens") or input_tokens)
        output_tokens = int(session_or_model.get("output_tokens") or output_tokens)
        cache_read_tokens = int(session_or_model.get("cache_read_tokens") or cache_read_tokens)
        cache_write_tokens = int(session_or_model.get("cache_write_tokens") or cache_write_tokens)
        provider = session_or_model.get("billing_provider") or provider
        base_url = session_or_model.get("billing_base_url") or base_url
    else:
        model = str(session_or_model or "")

    usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    result = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)
    if result.amount_usd is None:
        return (0.0, result.status)
    return (float(result.amount_usd), result.status)


def _bar_chart(values: List[int], max_width: int = 20) -> List[str]:
    """Build horizontal ``█``-bar strings scaled to the max value.

    Empty string for zero values.  All bars share the same scale so visual
    comparison is meaningful.
    """
    if not values:
        return []
    max_val = max(values) if values else 0
    if max_val <= 0:
        return ["" for _ in values]
    bars: List[str] = []
    for v in values:
        if v <= 0:
            bars.append("")
            continue
        scaled = int((v / max_val) * max_width)
        bars.append("█" * max(1, scaled))
    return bars


# ---------------------------------------------------------------------------
# InsightsEngine
# ---------------------------------------------------------------------------


# Columns we read from the sessions table.  Must match the schema in
# services/session_db.py (Task 1's 45-column upgrade).
_SESSION_COLS = (
    "id, source, model, started_at, ended_at, message_count, tool_call_count, "
    "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
    "billing_provider, billing_base_url, billing_mode, estimated_cost_usd, "
    "actual_cost_usd, cost_status, cost_source"
)

_GET_SESSIONS_WITH_SOURCE = (
    f"SELECT {_SESSION_COLS} FROM sessions WHERE started_at >= ? AND source = ? "
    "ORDER BY started_at DESC"
)
_GET_SESSIONS_ALL = (
    f"SELECT {_SESSION_COLS} FROM sessions WHERE started_at >= ? "
    "ORDER BY started_at DESC"
)


class InsightsEngine:
    """Aggregate usage insights from the session DB.

    Wraps a :class:`SessionDB` instance and exposes :meth:`generate`,
    :meth:`format_terminal`, :meth:`format_gateway`. The DB connection's
    ``row_factory`` must be ``sqlite3.Row`` so ``dict(row)`` works.

    Usage::

        from niaharness.insights import InsightsEngine
        from niaharness.services.session_db import SessionDB

        engine = InsightsEngine(SessionDB())
        report = engine.generate(days=30)
        print(engine.format_terminal(report))
    """

    def __init__(self, db: Any = None) -> None:
        if db is None:
            try:
                from niaharness.services.session_db import SessionDB
                db = SessionDB()
            except Exception as exc:
                logger.warning("Insights: could not open SessionDB: %s", exc)
                db = None
        self.db = db
        self._conn = db._conn if db is not None else None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, days: int = 30, source: Optional[str] = None) -> Dict[str, Any]:
        """Generate a complete insights report.

        Args:
            days: Number of days to look back (default 30).
            source: Optional filter by source platform (e.g. ``"cli"``,
                ``"telegram"``). ``None`` = all sources.

        Returns:
            Dict with all computed insights. See module docstring for shape.
        """
        cutoff = time.time() - (days * 86400)

        sessions = self._get_sessions(cutoff, source)
        tool_usage = self._get_tool_usage(cutoff, source)
        skill_usage = self._get_skill_usage(cutoff, source)
        message_stats = self._get_message_stats(cutoff, source)

        if not sessions:
            return {
                "days": days,
                "source_filter": source,
                "empty": True,
                "overview": {},
                "models": [],
                "platforms": [],
                "tools": [],
                "skills": {
                    "summary": {
                        "total_skill_loads": 0,
                        "total_skill_edits": 0,
                        "total_skill_actions": 0,
                        "distinct_skills_used": 0,
                    },
                    "top_skills": [],
                },
                "activity": {},
                "top_sessions": [],
            }

        overview = self._compute_overview(sessions, message_stats)
        models = self._compute_model_breakdown(sessions)
        platforms = self._compute_platform_breakdown(sessions)
        tools = self._compute_tool_breakdown(tool_usage)
        skills = self._compute_skill_breakdown(skill_usage)
        activity = self._compute_activity_patterns(sessions)
        top_sessions = self._compute_top_sessions(sessions)

        return {
            "days": days,
            "source_filter": source,
            "empty": False,
            "generated_at": time.time(),
            "overview": overview,
            "models": models,
            "platforms": platforms,
            "tools": tools,
            "skills": skills,
            "activity": activity,
            "top_sessions": top_sessions,
        }

    # ------------------------------------------------------------------
    # (a) Token / cost aggregation
    # ------------------------------------------------------------------

    def _get_sessions(self, cutoff: float, source: Optional[str] = None) -> List[Dict]:
        if self._conn is None:
            return []
        try:
            if source:
                rows = self._conn.execute(_GET_SESSIONS_WITH_SOURCE, (cutoff, source)).fetchall()
            else:
                rows = self._conn.execute(_GET_SESSIONS_ALL, (cutoff,)).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("Insights: failed to fetch sessions: %s", exc)
            return []

    def _get_message_stats(self, cutoff: float, source: Optional[str] = None) -> Dict:
        if self._conn is None:
            return {}
        try:
            if source:
                query = (
                    "SELECT "
                    "COUNT(*) as total_messages, "
                    "SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages, "
                    "SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages, "
                    "SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages "
                    "FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? AND s.source = ?"
                )
                row = self._conn.execute(query, (cutoff, source)).fetchone()
            else:
                query = (
                    "SELECT "
                    "COUNT(*) as total_messages, "
                    "SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages, "
                    "SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages, "
                    "SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages "
                    "FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ?"
                )
                row = self._conn.execute(query, (cutoff,)).fetchone()
            return dict(row) if row else {}
        except Exception as exc:
            logger.warning("Insights: failed to fetch message stats: %s", exc)
            return {}

    def _compute_overview(self, sessions: List[Dict], message_stats: Dict) -> Dict:
        total_sessions = len(sessions)
        total_messages = int(message_stats.get("total_messages") or 0)
        total_tool_calls = sum(int(s.get("tool_call_count") or 0) for s in sessions)
        total_input = sum(int(s.get("input_tokens") or 0) for s in sessions)
        total_output = sum(int(s.get("output_tokens") or 0) for s in sessions)
        total_cache_read = sum(int(s.get("cache_read_tokens") or 0) for s in sessions)
        total_cache_write = sum(int(s.get("cache_write_tokens") or 0) for s in sessions)
        total_tokens = total_input + total_output + total_cache_read + total_cache_write

        # Estimated cost: sum of per-session estimate_usage_cost.
        estimated_cost = 0.0
        actual_cost = 0.0
        models_with_pricing: set = set()
        models_without_pricing: set = set()
        unknown_cost_sessions = 0
        included_cost_sessions = 0

        for s in sessions:
            model = s.get("model") or ""
            cost, status = _estimate_cost(s)
            if status == "included":
                included_cost_sessions += 1
            elif status == "unknown":
                unknown_cost_sessions += 1
                if model:
                    models_without_pricing.add(model)
            else:
                estimated_cost += cost
                if model:
                    models_with_pricing.add(model)
            # actual_cost: prefer provider-reported actual_cost_usd if present.
            actual = s.get("actual_cost_usd")
            if actual is not None:
                try:
                    actual_cost += float(actual)
                except (TypeError, ValueError):
                    pass

        # Total active time (hours).
        total_seconds = 0.0
        for s in sessions:
            started = s.get("started_at")
            ended = s.get("ended_at")
            if started and ended:
                try:
                    total_seconds += float(ended) - float(started)
                except (TypeError, ValueError):
                    pass
        total_hours = total_seconds / 3600.0
        avg_session_duration = (total_seconds / total_sessions) if total_sessions else 0.0
        avg_messages_per_session = (total_messages / total_sessions) if total_sessions else 0.0
        avg_tokens_per_session = (total_tokens / total_sessions) if total_sessions else 0.0

        # Date range.
        started_times = [s.get("started_at") for s in sessions if s.get("started_at")]
        date_range_start = min(started_times) if started_times else None
        date_range_end = max(started_times) if started_times else None

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "actual_cost": actual_cost,
            "total_hours": total_hours,
            "avg_session_duration": avg_session_duration,
            "avg_messages_per_session": avg_messages_per_session,
            "avg_tokens_per_session": avg_tokens_per_session,
            "user_messages": int(message_stats.get("user_messages") or 0),
            "assistant_messages": int(message_stats.get("assistant_messages") or 0),
            "tool_messages": int(message_stats.get("tool_messages") or 0),
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "models_with_pricing": sorted(models_with_pricing),
            "models_without_pricing": sorted(models_without_pricing),
            "unknown_cost_sessions": unknown_cost_sessions,
            "included_cost_sessions": included_cost_sessions,
        }

    def _compute_model_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        by_model: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "model": "", "sessions": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0,
            "tool_calls": 0, "cost": 0.0, "has_pricing": False, "cost_status": "unknown",
        })

        for s in sessions:
            model = s.get("model") or "unknown"
            # Display name = last /-segment (strips vendor prefix).
            display = model.rsplit("/", 1)[-1] if "/" in model else model
            entry = by_model[display]
            entry["model"] = display
            entry["sessions"] += 1
            entry["input_tokens"] += int(s.get("input_tokens") or 0)
            entry["output_tokens"] += int(s.get("output_tokens") or 0)
            entry["cache_read_tokens"] += int(s.get("cache_read_tokens") or 0)
            entry["cache_write_tokens"] += int(s.get("cache_write_tokens") or 0)
            entry["tool_calls"] += int(s.get("tool_call_count") or 0)
            cost, status = _estimate_cost(s)
            entry["cost"] += cost
            if status != "unknown":
                entry["has_pricing"] = True
                entry["cost_status"] = status

        # Compute total_tokens + sort by (total_tokens, sessions) desc.
        result = []
        for entry in by_model.values():
            entry["total_tokens"] = (
                entry["input_tokens"] + entry["output_tokens"]
                + entry["cache_read_tokens"] + entry["cache_write_tokens"]
            )
            result.append(entry)
        result.sort(key=lambda e: (e["total_tokens"], e["sessions"]), reverse=True)
        return result

    def _compute_platform_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        by_platform: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "platform": "", "sessions": 0, "messages": 0, "input_tokens": 0,
            "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "total_tokens": 0, "tool_calls": 0,
        })

        for s in sessions:
            platform = s.get("source") or "unknown"
            entry = by_platform[platform]
            entry["platform"] = platform
            entry["sessions"] += 1
            entry["messages"] += int(s.get("message_count") or 0)
            entry["input_tokens"] += int(s.get("input_tokens") or 0)
            entry["output_tokens"] += int(s.get("output_tokens") or 0)
            entry["cache_read_tokens"] += int(s.get("cache_read_tokens") or 0)
            entry["cache_write_tokens"] += int(s.get("cache_write_tokens") or 0)
            entry["tool_calls"] += int(s.get("tool_call_count") or 0)

        result = []
        for entry in by_platform.values():
            entry["total_tokens"] = (
                entry["input_tokens"] + entry["output_tokens"]
                + entry["cache_read_tokens"] + entry["cache_write_tokens"]
            )
            result.append(entry)
        result.sort(key=lambda e: e["sessions"], reverse=True)
        return result

    # ------------------------------------------------------------------
    # (b) Tool usage
    # ------------------------------------------------------------------

    def _get_tool_usage(self, cutoff: float, source: Optional[str] = None) -> List[Dict]:
        """Two-source strategy: ``tool_name`` column + ``tool_calls`` JSON.

        Merges by **max-per-tool** to avoid double-counting when both are
        populated (``tool_name`` is set on tool-response messages;
        ``tool_calls`` JSON is on the preceding assistant message).
        """
        if self._conn is None:
            return []

        # Source 1: tool_name column on role='tool' messages.
        tool_counts: Dict[str, int] = Counter()
        try:
            if source:
                query = (
                    "SELECT m.tool_name, COUNT(*) as count "
                    "FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? AND s.source = ? "
                    "AND m.role = 'tool' AND m.tool_name IS NOT NULL "
                    "GROUP BY m.tool_name ORDER BY count DESC"
                )
                rows = self._conn.execute(query, (cutoff, source)).fetchall()
            else:
                query = (
                    "SELECT m.tool_name, COUNT(*) as count "
                    "FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? "
                    "AND m.role = 'tool' AND m.tool_name IS NOT NULL "
                    "GROUP BY m.tool_name ORDER BY count DESC"
                )
                rows = self._conn.execute(query, (cutoff,)).fetchall()
            for row in rows:
                name = row["tool_name"]
                if name:
                    tool_counts[name] = max(tool_counts.get(name, 0), int(row["count"]))
        except Exception as exc:
            logger.debug("Insights: tool_name query failed: %s", exc)

        # Source 2: tool_calls JSON on role='assistant' messages.
        try:
            if source:
                query = (
                    "SELECT m.tool_calls FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? AND s.source = ? "
                    "AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
                )
                rows = self._conn.execute(query, (cutoff, source)).fetchall()
            else:
                query = (
                    "SELECT m.tool_calls FROM messages m JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? "
                    "AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
                )
                rows = self._conn.execute(query, (cutoff,)).fetchall()
            json_counts: Dict[str, int] = Counter()
            for row in rows:
                raw = row["tool_calls"]
                if not raw:
                    continue
                try:
                    calls = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    name = fn.get("name") if isinstance(fn, dict) else None
                    if name:
                        json_counts[name] += 1
            # Merge by max-per-tool.
            for name, count in json_counts.items():
                tool_counts[name] = max(tool_counts.get(name, 0), count)
        except Exception as exc:
            logger.debug("Insights: tool_calls JSON query failed: %s", exc)

        return [{"tool_name": name, "count": count} for name, count in tool_counts.most_common()]

    def _compute_tool_breakdown(self, tool_usage: List[Dict]) -> List[Dict]:
        total_calls = sum(int(t["count"]) for t in tool_usage)
        result = []
        for t in tool_usage:
            count = int(t["count"])
            percentage = (count / total_calls * 100) if total_calls else 0.0
            result.append({
                "tool": t["tool_name"],
                "count": count,
                "percentage": round(percentage, 1),
            })
        return result

    # ------------------------------------------------------------------
    # (c) Skill usage
    # ------------------------------------------------------------------

    def _get_skill_usage(self, cutoff: float, source: Optional[str] = None) -> List[Dict]:
        """Scan assistant-message tool_calls for skill_view / skill_manage.

        Builds per-skill dict: ``{skill, view_count, manage_count, last_used_at}``.
        """
        if self._conn is None:
            return []

        skill_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"skill": "", "view_count": 0, "manage_count": 0, "last_used_at": 0.0}
        )

        try:
            if source:
                query = (
                    "SELECT m.tool_calls, m.timestamp FROM messages m "
                    "JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? AND s.source = ? "
                    "AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
                )
                rows = self._conn.execute(query, (cutoff, source)).fetchall()
            else:
                query = (
                    "SELECT m.tool_calls, m.timestamp FROM messages m "
                    "JOIN sessions s ON s.id = m.session_id "
                    "WHERE s.started_at >= ? "
                    "AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
                )
                rows = self._conn.execute(query, (cutoff,)).fetchall()

            for row in rows:
                raw = row["tool_calls"]
                ts = row["timestamp"] or 0.0
                if not raw:
                    continue
                try:
                    calls = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name") or ""
                    if name not in {"skill_view", "skill_manage"}:
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    skill_name = args.get("name") or args.get("skill") or "unknown"
                    entry = skill_data[skill_name]
                    entry["skill"] = skill_name
                    if name == "skill_view":
                        entry["view_count"] += 1
                    elif name == "skill_manage":
                        entry["manage_count"] += 1
                    if ts > entry["last_used_at"]:
                        entry["last_used_at"] = float(ts)
        except Exception as exc:
            logger.debug("Insights: skill usage query failed: %s", exc)

        return list(skill_data.values())

    def _compute_skill_breakdown(self, skill_usage: List[Dict]) -> Dict[str, Any]:
        total_loads = sum(int(s["view_count"]) for s in skill_usage)
        total_edits = sum(int(s["manage_count"]) for s in skill_usage)
        total_actions = total_loads + total_edits
        distinct = len(skill_usage)

        top_skills = []
        for s in skill_usage:
            total = int(s["view_count"]) + int(s["manage_count"])
            percentage = (total / total_actions * 100) if total_actions else 0.0
            top_skills.append({
                "skill": s["skill"],
                "view_count": int(s["view_count"]),
                "manage_count": int(s["manage_count"]),
                "total_count": total,
                "percentage": round(percentage, 1),
                "last_used_at": float(s["last_used_at"]) if s["last_used_at"] else None,
            })
        top_skills.sort(
            key=lambda x: (x["total_count"], x["view_count"], x["manage_count"], x["last_used_at"] or 0, x["skill"]),
            reverse=True,
        )

        return {
            "summary": {
                "total_skill_loads": total_loads,
                "total_skill_edits": total_edits,
                "total_skill_actions": total_actions,
                "distinct_skills_used": distinct,
            },
            "top_skills": top_skills,
        }

    # ------------------------------------------------------------------
    # (d) Activity patterns
    # ------------------------------------------------------------------

    def _compute_activity_patterns(self, sessions: List[Dict]) -> Dict:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_counts = [0] * 7
        hour_counts = [0] * 24
        active_dates: set = set()

        for s in sessions:
            started = s.get("started_at")
            if not started:
                continue
            try:
                dt = datetime.fromtimestamp(float(started), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                continue
            day_counts[dt.weekday()] += 1
            hour_counts[dt.hour] += 1
            active_dates.add(dt.strftime("%Y-%m-%d"))

        # Busiest day.
        busiest_day = None
        if any(day_counts):
            max_day_idx = day_counts.index(max(day_counts))
            busiest_day = {"day": day_names[max_day_idx], "count": day_counts[max_day_idx]}

        # Busiest hour.
        busiest_hour = None
        if any(hour_counts):
            max_hour_idx = hour_counts.index(max(hour_counts))
            busiest_hour = {"hour": max_hour_idx, "count": hour_counts[max_hour_idx]}

        # Max streak (consecutive calendar days with ≥1 session).
        sorted_dates = sorted(active_dates)
        max_streak = 0
        current_streak = 0
        prev_date: Optional[datetime] = None
        for date_str in sorted_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if prev_date is None:
                current_streak = 1
            else:
                delta = (dt - prev_date).days
                if delta == 1:
                    current_streak += 1
                else:
                    current_streak = 1
            max_streak = max(max_streak, current_streak)
            prev_date = dt

        return {
            "by_day": [{"day": day_names[i], "count": day_counts[i]} for i in range(7)],
            "by_hour": [{"hour": i, "count": hour_counts[i]} for i in range(24)],
            "busiest_day": busiest_day,
            "busiest_hour": busiest_hour,
            "active_days": len(active_dates),
            "max_streak": max_streak,
        }

    # ------------------------------------------------------------------
    # (e) Top sessions
    # ------------------------------------------------------------------

    def _compute_top_sessions(self, sessions: List[Dict]) -> List[Dict]:
        if not sessions:
            return []

        def _fmt_date(ts: Any) -> str:
            if not ts:
                return ""
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%b %d, %Y")
            except (TypeError, ValueError, OSError):
                return ""

        result = []

        # Longest session (by ended_at - started_at).
        def _duration(s: Dict) -> float:
            started = s.get("started_at")
            ended = s.get("ended_at")
            if started and ended:
                try:
                    return float(ended) - float(started)
                except (TypeError, ValueError):
                    pass
            return 0.0

        longest = max(sessions, key=_duration, default=None)
        if longest and _duration(longest) > 0:
            result.append({
                "label": "Longest session",
                "session_id": (longest.get("id") or "")[:16],
                "value": format_duration_compact(_duration(longest)),
                "date": _fmt_date(longest.get("started_at")),
            })

        # Most messages.
        most_msgs = max(sessions, key=lambda s: int(s.get("message_count") or 0), default=None)
        if most_msgs and int(most_msgs.get("message_count") or 0) > 0:
            result.append({
                "label": "Most messages",
                "session_id": (most_msgs.get("id") or "")[:16],
                "value": str(int(most_msgs.get("message_count") or 0)),
                "date": _fmt_date(most_msgs.get("started_at")),
            })

        # Most tokens.
        def _total_tokens(s: Dict) -> int:
            return (
                int(s.get("input_tokens") or 0)
                + int(s.get("output_tokens") or 0)
                + int(s.get("cache_read_tokens") or 0)
                + int(s.get("cache_write_tokens") or 0)
            )

        most_tokens = max(sessions, key=_total_tokens, default=None)
        if most_tokens and _total_tokens(most_tokens) > 0:
            result.append({
                "label": "Most tokens",
                "session_id": (most_tokens.get("id") or "")[:16],
                "value": format_token_count_compact(_total_tokens(most_tokens)),
                "date": _fmt_date(most_tokens.get("started_at")),
            })

        # Most tool calls.
        most_tools = max(sessions, key=lambda s: int(s.get("tool_call_count") or 0), default=None)
        if most_tools and int(most_tools.get("tool_call_count") or 0) > 0:
            result.append({
                "label": "Most tool calls",
                "session_id": (most_tools.get("id") or "")[:16],
                "value": str(int(most_tools.get("tool_call_count") or 0)),
                "date": _fmt_date(most_tools.get("started_at")),
            })

        return result[:4]

    # ------------------------------------------------------------------
    # (f) Formatting
    # ------------------------------------------------------------------

    def format_terminal(self, report: Dict) -> str:
        """Render the report as a Unicode-box-decorated CLI report."""
        if report.get("empty"):
            days = report.get("days", 30)
            source = report.get("source_filter")
            suffix = f" (source: {source})" if source else ""
            return f"  No sessions found in the last {days} days{suffix}."

        lines: List[str] = []
        days = report.get("days", 30)
        source = report.get("source_filter")

        # Header box.
        title = "📊 NIA Insights"
        subtitle = f"Last {days} days"
        if source:
            subtitle += f" ({source})"
        box_width = max(len(title), len(subtitle)) + 4
        lines.append("╔" + "═" * box_width + "╗")
        lines.append(f"║  {title:<{box_width - 2}}  ║")
        lines.append(f"║  {subtitle:<{box_width - 2}}  ║")
        lines.append("╚" + "═" * box_width + "╝")
        lines.append("")

        ov = report.get("overview", {})

        # Period.
        start = ov.get("date_range_start")
        end = ov.get("date_range_end")
        if start and end:
            try:
                start_str = datetime.fromtimestamp(float(start), tz=timezone.utc).strftime("%b %d, %Y")
                end_str = datetime.fromtimestamp(float(end), tz=timezone.utc).strftime("%b %d, %Y")
                lines.append(f"Period: {start_str} — {end_str}")
                lines.append("")
            except (TypeError, ValueError, OSError):
                pass

        # Overview.
        lines.append("Overview:")
        lines.append(f"  Sessions:      {ov.get('total_sessions', 0)}")
        lines.append(f"  Messages:      {ov.get('total_messages', 0)}")
        lines.append(f"  Tool calls:    {ov.get('total_tool_calls', 0)}")
        lines.append(f"  User messages: {ov.get('user_messages', 0)}")
        lines.append(f"  Input tokens:  {format_token_count_compact(ov.get('total_input_tokens', 0))}")
        lines.append(f"  Output tokens: {format_token_count_compact(ov.get('total_output_tokens', 0))}")
        lines.append(f"  Cache read:    {format_token_count_compact(ov.get('total_cache_read_tokens', 0))}")
        lines.append(f"  Cache write:   {format_token_count_compact(ov.get('total_cache_write_tokens', 0))}")
        lines.append(f"  Total tokens:  {format_token_count_compact(ov.get('total_tokens', 0))}")
        est_cost = ov.get("estimated_cost", 0.0)
        lines.append(f"  Est. cost:     ${est_cost:.4f}")
        if ov.get("unknown_cost_sessions"):
            lines.append(f"  Unknown cost:  {ov['unknown_cost_sessions']} sessions (no pricing data)")
        if ov.get("included_cost_sessions"):
            lines.append(f"  Included:      {ov['included_cost_sessions']} sessions (subscription)")
        if ov.get("total_hours"):
            lines.append(f"  Active time:   {format_duration_compact(ov['total_hours'] * 3600)}")
        if ov.get("avg_session_duration"):
            lines.append(f"  Avg session:   {format_duration_compact(ov['avg_session_duration'])}")
        if ov.get("avg_messages_per_session"):
            lines.append(f"  Avg msgs/sess: {ov['avg_messages_per_session']:.1f}")
        lines.append("")

        # Models.
        models = report.get("models", [])
        if models:
            lines.append("Models Used:")
            lines.append(f"  {'Model':<30} {'Sessions':>8} {'Tokens':>12}")
            lines.append(f"  {'-' * 30} {'-' * 8} {'-' * 12}")
            for m in models[:15]:
                name = (m.get("model") or "unknown")[:28]
                lines.append(
                    f"  {name:<30} {m.get('sessions', 0):>8} "
                    f"{format_token_count_compact(m.get('total_tokens', 0)):>12}"
                )
            if len(models) > 15:
                lines.append(f"  ... and {len(models) - 15} more models")
            lines.append("")

        # Platforms.
        platforms = report.get("platforms", [])
        if len(platforms) > 1 or (platforms and platforms[0].get("platform") != "cli"):
            lines.append("Platforms:")
            lines.append(f"  {'Platform':<14} {'Sessions':>8} {'Messages':>9} {'Tokens':>12}")
            lines.append(f"  {'-' * 14} {'-' * 8} {'-' * 9} {'-' * 12}")
            for p in platforms:
                name = (p.get("platform") or "unknown")[:12]
                lines.append(
                    f"  {name:<14} {p.get('sessions', 0):>8} {p.get('messages', 0):>9} "
                    f"{format_token_count_compact(p.get('total_tokens', 0)):>12}"
                )
            lines.append("")

        # Top tools.
        tools = report.get("tools", [])
        if tools:
            lines.append("Top Tools:")
            lines.append(f"  {'Tool':<28} {'Calls':>6} {'%':>6}")
            lines.append(f"  {'-' * 28} {'-' * 6} {'-' * 6}")
            for t in tools[:15]:
                name = (t.get("tool") or "unknown")[:26]
                lines.append(
                    f"  {name:<28} {t.get('count', 0):>6} {t.get('percentage', 0):>5.1f}%"
                )
            if len(tools) > 15:
                lines.append(f"  ... and {len(tools) - 15} more tools")
            lines.append("")

        # Top skills.
        skills = report.get("skills", {})
        top_skills = skills.get("top_skills", [])
        if top_skills:
            lines.append("Top Skills:")
            lines.append(f"  {'Skill':<28} {'Loads':>6} {'Edits':>6} {'Last used':>12}")
            lines.append(f"  {'-' * 28} {'-' * 6} {'-' * 6} {'-' * 12}")
            for s in top_skills[:10]:
                name = (s.get("skill") or "unknown")[:26]
                last = s.get("last_used_at")
                last_str = ""
                if last:
                    try:
                        last_str = datetime.fromtimestamp(float(last), tz=timezone.utc).strftime("%b %d, %Y")
                    except (TypeError, ValueError, OSError):
                        pass
                lines.append(
                    f"  {name:<28} {s.get('view_count', 0):>6} {s.get('manage_count', 0):>6} {last_str:>12}"
                )
            summary = skills.get("summary", {})
            lines.append(
                f"  Summary: {summary.get('distinct_skills_used', 0)} skills, "
                f"{summary.get('total_skill_loads', 0)} loads, "
                f"{summary.get('total_skill_edits', 0)} edits"
            )
            lines.append("")

        # Activity patterns.
        activity = report.get("activity", {})
        if activity:
            lines.append("Activity Patterns:")
            by_day = activity.get("by_day", [])
            if by_day:
                day_values = [d.get("count", 0) for d in by_day]
                bars = _bar_chart(day_values, max_width=15)
                for i, day_entry in enumerate(by_day):
                    bar = bars[i] if i < len(bars) else ""
                    lines.append(f"  {day_entry['day']}: {bar} {day_entry['count']}")
            busiest_day = activity.get("busiest_day")
            busiest_hour = activity.get("busiest_hour")
            if busiest_day or busiest_hour:
                parts = []
                if busiest_day:
                    parts.append(f"{busiest_day['day']}s ({busiest_day['count']} sessions)")
                if busiest_hour:
                    hr = busiest_hour["hour"]
                    hr_str = f"{hr % 12 or 12}{'AM' if hr < 12 else 'PM'}"
                    parts.append(f"{hr_str} ({busiest_hour['count']} sessions)")
                lines.append(f"  Peak: {', '.join(parts)}")
            if activity.get("active_days"):
                lines.append(f"  Active days: {activity['active_days']}")
            if activity.get("max_streak"):
                lines.append(f"  Best streak: {activity['max_streak']} consecutive days")
            lines.append("")

        # Top sessions.
        top_sessions = report.get("top_sessions", [])
        if top_sessions:
            lines.append("Notable Sessions:")
            for ts in top_sessions:
                label = (ts.get("label") or "")[:20]
                value = (ts.get("value") or "")[:18]
                date = ts.get("date") or ""
                sid = ts.get("session_id") or ""
                lines.append(f"  {label:<20} {value:<18} ({date}, {sid})")
            lines.append("")

        return "\n".join(lines)

    def format_gateway(self, report: Dict) -> str:
        """Render a Markdown-flavored shorter report for chat delivery."""
        if report.get("empty"):
            days = report.get("days", 30)
            return f"No sessions found in the last {days} days."

        lines: List[str] = []
        days = report.get("days", 30)
        lines.append(f"📊 **NIA Insights** — Last {days} days")
        lines.append("")

        ov = report.get("overview", {})
        lines.append(
            f"**Sessions:** {ov.get('total_sessions', 0)} | "
            f"**Messages:** {ov.get('total_messages', 0)} | "
            f"**Tool calls:** {ov.get('total_tool_calls', 0)}"
        )
        total_tokens = ov.get("total_tokens", 0)
        lines.append(
            f"**Tokens:** {format_token_count_compact(total_tokens)} "
            f"(in: {format_token_count_compact(ov.get('total_input_tokens', 0))} / "
            f"out: {format_token_count_compact(ov.get('total_output_tokens', 0))})"
        )
        est_cost = ov.get("estimated_cost", 0.0)
        if est_cost:
            lines.append(f"**Est. cost:** ${est_cost:.4f}")
        if ov.get("total_hours"):
            lines.append(
                f"**Active time:** ~{format_duration_compact(ov['total_hours'] * 3600)} | "
                f"**Avg session:** ~{format_duration_compact(ov.get('avg_session_duration', 0))}"
            )
        lines.append("")

        # Models (top 5).
        models = report.get("models", [])[:5]
        if models:
            lines.append("**Models:**")
            for m in models:
                lines.append(
                    f"  {m.get('model', 'unknown')} — "
                    f"{m.get('sessions', 0)} sessions, "
                    f"{format_token_count_compact(m.get('total_tokens', 0))} tokens"
                )
            lines.append("")

        # Platforms (only if multi-platform).
        platforms = report.get("platforms", [])
        if len(platforms) > 1:
            lines.append("**Platforms:**")
            for p in platforms:
                lines.append(
                    f"  {p.get('platform', 'unknown')} — "
                    f"{p.get('sessions', 0)} sessions, "
                    f"{format_token_count_compact(p.get('total_tokens', 0))} tokens"
                )
            lines.append("")

        # Top tools (top 8).
        tools = report.get("tools", [])[:8]
        if tools:
            lines.append("**Top Tools:**")
            for t in tools:
                lines.append(f"  {t.get('tool', 'unknown')} — {t.get('count', 0)} calls ({t.get('percentage', 0):.1f}%)")
            lines.append("")

        # Top skills (top 5).
        skills = report.get("skills", {})
        top_skills = skills.get("top_skills", [])[:5]
        if top_skills:
            lines.append("**Top Skills:**")
            for s in top_skills:
                last = s.get("last_used_at")
                last_str = ""
                if last:
                    try:
                        last_str = datetime.fromtimestamp(float(last), tz=timezone.utc).strftime(", last used %b %d")
                    except (TypeError, ValueError, OSError):
                        pass
                lines.append(
                    f"  {s.get('skill', 'unknown')} — "
                    f"{s.get('view_count', 0)} loads, {s.get('manage_count', 0)} edits{last_str}"
                )
            lines.append("")

        # Activity summary.
        activity = report.get("activity", {})
        if activity:
            busiest_day = activity.get("busiest_day")
            busiest_hour = activity.get("busiest_hour")
            parts = []
            if busiest_day:
                parts.append(f"{busiest_day['day']}s ({busiest_day['count']} sessions)")
            if busiest_hour:
                hr = busiest_hour["hour"]
                hr_str = f"{hr % 12 or 12}{'AM' if hr < 12 else 'PM'}"
                parts.append(f"{hr_str} ({busiest_hour['count']} sessions)")
            if parts:
                lines.append(f"**Busiest:** {', '.join(parts)}")
            if activity.get("active_days"):
                lines.append(f"**Active days:** {activity['active_days']}")
            if activity.get("max_streak"):
                lines.append(f"**Best streak:** {activity['max_streak']} consecutive days")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def to_json(report: Dict, *, indent: int = 2) -> str:
    """Serialize an insights report to JSON.

    Handles ``datetime`` / ``Decimal`` types by converting to str / float.
    """
    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    return json.dumps(report, default=_default, indent=indent)


# ---------------------------------------------------------------------------
# Backwards-compat shims (old API)
# ---------------------------------------------------------------------------


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Backwards-compat: estimate cost from a model + token counts.

    New code should use :func:`estimate_usage_cost` with a
    :class:`CanonicalUsage` directly.
    """
    usage = CanonicalUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost, _ = _estimate_cost(model, input_tokens, output_tokens)
    return cost


__all__ = [
    "InsightsEngine",
    "estimate_cost",
    "to_json",
]
