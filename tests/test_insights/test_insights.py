"""Tests for the insights module — usage pricing + analytics engine.

Covers:
  - usage_pricing: CanonicalUsage, BillingRoute resolution, PricingEntry lookup,
    estimate_usage_cost, normalize_usage (3 API shapes), model-name normalization,
    format_duration_compact, format_token_count_compact, has_known_pricing
  - InsightsEngine: generate (empty + populated), overview, model breakdown,
    platform breakdown, tool usage (2-source merge), skill usage, activity
    patterns (day/hour/streak), top sessions, format_terminal, format_gateway, to_json
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from niaharness.insights.usage_pricing import (
    CanonicalUsage,
    BillingRoute,
    PricingEntry,
    CostResult,
    estimate_usage_cost,
    resolve_billing_route,
    normalize_usage,
    has_known_pricing,
    get_pricing_entry,
    format_duration_compact,
    format_token_count_compact,
    _normalize_anthropic_model_name,
    _normalize_bedrock_model_name,
    _lookup_official_docs_pricing,
    _OFFICIAL_DOCS_PRICING,
)
from niaharness.insights import InsightsEngine, to_json, estimate_cost


# ---------------------------------------------------------------------------
# usage_pricing — CanonicalUsage
# ---------------------------------------------------------------------------


class TestCanonicalUsage:
    def test_default_values(self):
        usage = CanonicalUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.request_count == 1
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

    def test_prompt_tokens_includes_cache(self):
        usage = CanonicalUsage(
            input_tokens=100,
            cache_read_tokens=50,
            cache_write_tokens=25,
        )
        assert usage.prompt_tokens == 175  # 100 + 50 + 25

    def test_total_tokens(self):
        usage = CanonicalUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_tokens=50,
        )
        assert usage.total_tokens == 350  # 100 + 50 + 0 + 200

    def test_add_sums_all_buckets(self):
        a = CanonicalUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10)
        b = CanonicalUsage(input_tokens=200, output_tokens=25, cache_write_tokens=5)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 75
        assert c.cache_read_tokens == 10
        assert c.cache_write_tokens == 5
        assert c.request_count == 2

    def test_add_drops_raw_usage(self):
        a = CanonicalUsage(input_tokens=1, raw_usage={"x": 1})
        b = CanonicalUsage(input_tokens=2, raw_usage={"y": 2})
        c = a + b
        assert c.raw_usage is None

    def test_add_with_non_canonical_returns_notimplemented(self):
        a = CanonicalUsage()
        result = a.__add__("not a CanonicalUsage")
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# usage_pricing — resolve_billing_route
# ---------------------------------------------------------------------------


class TestResolveBillingRoute:
    def test_anthropic_prefix_inferred(self):
        route = resolve_billing_route("anthropic/claude-opus-4-7")
        assert route.provider == "anthropic"
        assert route.model == "claude-opus-4-7"

    def test_openai_prefix_inferred(self):
        route = resolve_billing_route("openai/gpt-4o")
        assert route.provider == "openai"
        assert route.model == "gpt-4o"

    def test_explicit_provider(self):
        route = resolve_billing_route("claude-opus-4-7", provider="anthropic")
        assert route.provider == "anthropic"
        assert route.model == "claude-opus-4-7"

    def test_openrouter_detected_by_base_url(self):
        route = resolve_billing_route("claude-opus-4-7", base_url="https://openrouter.ai/api/v1")
        assert route.provider == "openrouter"

    def test_subscription_included_route(self):
        route = resolve_billing_route("claude-opus-4-7", provider="claude-code-oauth")
        assert route.billing_mode == "subscription_included"
        assert route.provider == "subscription_included"

    def test_model_name_inference_claude(self):
        route = resolve_billing_route("claude-opus-4-7")
        assert route.provider == "anthropic"

    def test_model_name_inference_gpt(self):
        route = resolve_billing_route("gpt-4o")
        assert route.provider == "openai"

    def test_model_name_inference_gemini(self):
        route = resolve_billing_route("gemini-2.5-pro")
        assert route.provider == "google"

    def test_model_name_inference_deepseek(self):
        route = resolve_billing_route("deepseek-chat")
        assert route.provider == "deepseek"

    def test_empty_model(self):
        route = resolve_billing_route("")
        assert route.model == ""


# ---------------------------------------------------------------------------
# usage_pricing — model-name normalization
# ---------------------------------------------------------------------------


class TestModelNameNormalization:
    def test_anthropic_strips_prefix(self):
        assert _normalize_anthropic_model_name("anthropic/claude-opus-4-7") == "claude-opus-4-7"

    def test_anthropic_dot_to_dash(self):
        assert _normalize_anthropic_model_name("claude-opus-4.7") == "claude-opus-4-7"
        assert _normalize_anthropic_model_name("claude-sonnet-4.6") == "claude-sonnet-4-6"

    def test_anthropic_already_normalized(self):
        assert _normalize_anthropic_model_name("claude-opus-4-7") == "claude-opus-4-7"

    def test_bedrock_strips_region_prefix(self):
        assert _normalize_bedrock_model_name("us.anthropic.claude-opus-4-7") == "anthropic.claude-opus-4-7"
        assert _normalize_bedrock_model_name("global.anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"
        assert _normalize_bedrock_model_name("eu.anthropic.claude-haiku-4-5") == "anthropic.claude-haiku-4-5"

    def test_bedrock_dot_to_dash(self):
        assert _normalize_bedrock_model_name("anthropic.claude-opus-4.7") == "anthropic.claude-opus-4-7"

    def test_bedrock_no_prefix(self):
        assert _normalize_bedrock_model_name("anthropic.claude-opus-4-7") == "anthropic.claude-opus-4-7"


# ---------------------------------------------------------------------------
# usage_pricing — pricing table lookup
# ---------------------------------------------------------------------------


class TestPricingTableLookup:
    def test_direct_lookup_anthropic(self):
        route = BillingRoute(provider="anthropic", model="claude-opus-4-7")
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None
        assert entry.input_cost_per_million is not None
        assert entry.output_cost_per_million is not None

    def test_anthropic_dot_normalization(self):
        route = BillingRoute(provider="anthropic", model="claude-opus-4.7")
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None  # Should find "claude-opus-4-7" after normalization

    def test_bedrock_region_normalization(self):
        route = BillingRoute(provider="bedrock", model="us.anthropic.claude-opus-4-6")
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None

    def test_unknown_model_returns_none(self):
        route = BillingRoute(provider="anthropic", model="totally-unknown-model")
        entry = _lookup_official_docs_pricing(route)
        assert entry is None

    def test_table_has_30_plus_entries(self):
        # Sanity check that the pricing table is populated.
        assert len(_OFFICIAL_DOCS_PRICING) >= 30

    def test_table_includes_all_providers(self):
        providers = {key[0] for key in _OFFICIAL_DOCS_PRICING}
        assert "anthropic" in providers
        assert "openai" in providers
        assert "deepseek" in providers
        assert "google" in providers
        assert "bedrock" in providers

    def test_anthropic_entries_have_cache_pricing(self):
        # Modern Anthropic models (4.x) all have cache pricing; legacy 3.x may not.
        for (provider, model), entry in _OFFICIAL_DOCS_PRICING.items():
            if provider == "anthropic" and ("4-" in model or "4." in model):
                assert entry.cache_read_cost_per_million is not None
                assert entry.cache_write_cost_per_million is not None

    def test_openai_entries_have_cache_read(self):
        for (provider, model), entry in _OFFICIAL_DOCS_PRICING.items():
            if provider == "openai":
                # OpenAI supports cache_read (prompt caching) but not cache_write.
                assert entry.cache_read_cost_per_million is not None


# ---------------------------------------------------------------------------
# usage_pricing — estimate_usage_cost
# ---------------------------------------------------------------------------


class TestEstimateUsageCost:
    def test_anthropic_known_model(self):
        usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=500_000)
        result = estimate_usage_cost("claude-opus-4-7", usage, provider="anthropic")
        assert result.status == "estimated"
        assert result.amount_usd is not None
        # $5 (1M input) + $12.50 (500K output) = $17.50
        assert abs(float(result.amount_usd) - 17.50) < 0.01

    def test_anthropic_with_cache_tokens(self):
        usage = CanonicalUsage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_read_tokens=200_000,
            cache_write_tokens=100_000,
        )
        result = estimate_usage_cost("claude-opus-4-7", usage, provider="anthropic")
        # $5 (in) + $12.50 (out) + $0.10 (cache_read) + $0.625 (cache_write)
        # = $18.225
        assert abs(float(result.amount_usd) - 18.225) < 0.01

    def test_unknown_model(self):
        usage = CanonicalUsage(input_tokens=1_000_000)
        result = estimate_usage_cost("totally-unknown-model", usage)
        assert result.status == "unknown"
        assert result.amount_usd is None
        assert result.label == "n/a"

    def test_subscription_included(self):
        usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        result = estimate_usage_cost("claude-opus-4-7", usage, provider="claude-code-oauth")
        assert result.status == "included"
        assert float(result.amount_usd) == 0.0
        assert result.label == "included"

    def test_cache_read_unavailable_returns_unknown(self):
        # DeepSeek has no cache pricing — using cache_read tokens → unknown.
        usage = CanonicalUsage(input_tokens=100, cache_read_tokens=50)
        result = estimate_usage_cost("deepseek-chat", usage, provider="deepseek")
        assert result.status == "unknown"
        assert any("cache-read pricing unavailable" in note for note in result.notes)

    def test_zero_tokens_zero_cost(self):
        usage = CanonicalUsage()
        result = estimate_usage_cost("claude-opus-4-7", usage, provider="anthropic")
        assert result.status == "estimated"
        assert float(result.amount_usd) == 0.0

    def test_label_format(self):
        usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=0)
        result = estimate_usage_cost("claude-opus-4-7", usage, provider="anthropic")
        assert result.label.startswith("~$")
        assert "5.00" in result.label  # $5 for 1M input

    def test_has_known_pricing_true(self):
        assert has_known_pricing("claude-opus-4-7", provider="anthropic") is True

    def test_has_known_pricing_false(self):
        assert has_known_pricing("totally-unknown-model") is False

    def test_has_known_pricing_subscription(self):
        assert has_known_pricing("claude-opus-4-7", provider="claude-code-oauth") is True


# ---------------------------------------------------------------------------
# usage_pricing — normalize_usage (3 API shapes)
# ---------------------------------------------------------------------------


class TestNormalizeUsage:
    def test_anthropic_shape(self):
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        }
        usage = normalize_usage(raw)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 10

    def test_openai_chat_completions_shape(self):
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
        usage = normalize_usage(raw)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        assert usage.cache_read_tokens == 30
        assert usage.reasoning_tokens == 5

    def test_codex_responses_shape(self):
        raw = {
            "input_tokens": 200,
            "output_tokens": 80,
            "input_tokens_details": {"cached_tokens": 30},
            "output_tokens_details": {"reasoning_tokens": 5},
        }
        usage = normalize_usage(raw)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        assert usage.cache_read_tokens == 30
        assert usage.reasoning_tokens == 5

    def test_none_returns_empty(self):
        usage = normalize_usage(None)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_fallback_to_anthropic_top_level_for_openai_compat(self):
        """OpenAI-compat proxy routing Claude may put cache fields at top level."""
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0},  # OpenAI says 0
            "cache_read_input_tokens": 40,  # But Anthropic top-level has it
            "cache_creation_input_tokens": 20,
        }
        usage = normalize_usage(raw)
        # Should fall back to top-level Anthropic fields.
        assert usage.cache_read_tokens == 40
        assert usage.cache_write_tokens == 20

    def test_sdk_object_with_model_dump(self):
        class FakeUsage:
            def model_dump(self):
                return {"input_tokens": 100, "output_tokens": 50}

        usage = normalize_usage(FakeUsage())
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


# ---------------------------------------------------------------------------
# usage_pricing — formatting helpers
# ---------------------------------------------------------------------------


class TestFormatDurationCompact:
    def test_seconds(self):
        assert format_duration_compact(45) == "45s"
        assert format_duration_compact(0) == "0s"

    def test_minutes(self):
        assert format_duration_compact(120) == "2m"
        assert format_duration_compact(720) == "12m"

    def test_hours(self):
        assert format_duration_compact(3600) == "1h"
        assert format_duration_compact(11700) == "3h 15m"  # 3h 15m

    def test_days(self):
        assert format_duration_compact(86400) == "1.0d"
        assert format_duration_compact(86400 * 5) == "5.0d"
        assert format_duration_compact(86400 * 15) == "15d"


class TestFormatTokenCountCompact:
    def test_small_numbers(self):
        assert format_token_count_compact(0) == "0"
        assert format_token_count_compact(999) == "999"

    def test_thousands(self):
        assert format_token_count_compact(1500) == "1.5K"
        assert format_token_count_compact(10000) == "10K"

    def test_millions(self):
        assert format_token_count_compact(1_500_000) == "1.5M"
        assert format_token_count_compact(10_000_000) == "10M"

    def test_billions(self):
        assert format_token_count_compact(1_500_000_000) == "1.5B"


# ---------------------------------------------------------------------------
# InsightsEngine — integration tests with a real SQLite DB
# ---------------------------------------------------------------------------


def _create_test_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a test DB with the sessions + messages schema and return a connection."""
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'cli',
            model TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL
        );
    """)
    return conn


def _insert_session(
    conn: sqlite3.Connection,
    *,
    id: str,
    source: str = "cli",
    model: str = "claude-opus-4-7",
    started_at: float,
    ended_at: float | None = None,
    message_count: int = 5,
    tool_call_count: int = 2,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    billing_provider: str = "anthropic",
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (id, source, model, started_at, ended_at, message_count,
                              tool_call_count, input_tokens, output_tokens,
                              cache_read_tokens, cache_write_tokens, billing_provider)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (id, source, model, started_at, ended_at, message_count, tool_call_count,
         input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, billing_provider),
    )


def _insert_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    role: str,
    content: str = "",
    tool_calls: str | None = None,
    tool_name: str | None = None,
    timestamp: float,
) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, tool_calls, tool_name, timestamp),
    )


@pytest.fixture
def insights_engine(tmp_path):
    """Build an InsightsEngine backed by a populated test DB."""
    conn = _create_test_db(tmp_path)
    now = time.time()

    # 3 sessions across 2 sources.
    _insert_session(
        conn, id="sess-1", source="cli", model="claude-opus-4-7",
        started_at=now - 7200, ended_at=now - 7000,  # 200s session
        message_count=10, tool_call_count=3,
        input_tokens=5000, output_tokens=2000,
        cache_read_tokens=1000, cache_write_tokens=500,
    )
    _insert_session(
        conn, id="sess-2", source="cli", model="claude-sonnet-4-6",
        started_at=now - 3600, ended_at=now - 3500,  # 100s session
        message_count=8, tool_call_count=5,
        input_tokens=3000, output_tokens=1500,
    )
    _insert_session(
        conn, id="sess-3", source="telegram", model="claude-opus-4-7",
        started_at=now - 1800, ended_at=now - 1700,  # 100s session
        message_count=15, tool_call_count=8,
        input_tokens=10000, output_tokens=5000,
    )

    # Messages: user / assistant (with tool_calls JSON) / tool.
    _insert_message(conn, session_id="sess-1", role="user", content="hello", timestamp=now - 7100)
    _insert_message(
        conn, session_id="sess-1", role="assistant", content="let me check",
        tool_calls=json.dumps([
            {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}},
            {"id": "call_2", "function": {"name": "skill_view", "arguments": '{"name": "python-help"}'}},
        ]),
        timestamp=now - 7050,
    )
    _insert_message(conn, session_id="sess-1", role="tool", tool_name="read_file", content="file contents", timestamp=now - 7000)

    _insert_message(conn, session_id="sess-2", role="user", content="hi", timestamp=now - 3550)
    _insert_message(
        conn, session_id="sess-2", role="assistant", content="checking",
        tool_calls=json.dumps([
            {"id": "call_3", "function": {"name": "grep", "arguments": '{"pattern": "foo"}'}},
            {"id": "call_4", "function": {"name": "skill_manage", "arguments": '{"name": "python-help", "action": "edit"}'}},
        ]),
        timestamp=now - 3525,
    )
    _insert_message(conn, session_id="sess-2", role="tool", tool_name="grep", content="matches", timestamp=now - 3500)

    conn.commit()

    # Build a fake SessionDB-like object.
    fake_db = MagicMock()
    fake_db._conn = conn

    engine = InsightsEngine(db=fake_db)
    yield engine
    conn.close()


class TestInsightsEngineGenerate:
    def test_empty_db_returns_empty_report(self, tmp_path):
        conn = _create_test_db(tmp_path)
        fake_db = MagicMock()
        fake_db._conn = conn
        engine = InsightsEngine(db=fake_db)
        report = engine.generate(days=30)
        assert report["empty"] is True
        assert report["overview"] == {}
        assert report["models"] == []
        conn.close()

    def test_populated_report_not_empty(self, insights_engine):
        report = insights_engine.generate(days=30)
        assert report["empty"] is False
        assert report["overview"]["total_sessions"] == 3

    def test_overview_totals(self, insights_engine):
        report = insights_engine.generate(days=30)
        ov = report["overview"]
        assert ov["total_sessions"] == 3
        assert ov["total_messages"] == 6  # 2 user + 2 assistant + 2 tool
        assert ov["total_input_tokens"] == 18000  # 5000 + 3000 + 10000
        assert ov["total_output_tokens"] == 8500  # 2000 + 1500 + 5000
        assert ov["total_cache_read_tokens"] == 1000
        assert ov["total_cache_write_tokens"] == 500
        assert ov["total_tokens"] == 28000

    def test_overview_cost_estimation(self, insights_engine):
        report = insights_engine.generate(days=30)
        ov = report["overview"]
        # claude-opus-4-7: $5/M in, $25/M out, $0.50/M cache_read, $6.25/M cache_write
        # sess-1: 5000 in + 2000 out + 1000 cache_read + 500 cache_write
        #   = 0.025 + 0.05 + 0.0005 + 0.003125 = $0.078625
        # sess-3: 10000 in + 5000 out
        #   = 0.05 + 0.125 = $0.175
        # claude-sonnet-4-6: $3/M in, $15/M out
        # sess-2: 3000 in + 1500 out
        #   = 0.009 + 0.0225 = $0.0315
        # Total ≈ $0.285
        assert ov["estimated_cost"] > 0.28
        assert ov["estimated_cost"] < 0.29
        assert ov["unknown_cost_sessions"] == 0

    def test_overview_message_breakdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        ov = report["overview"]
        assert ov["user_messages"] == 2
        assert ov["assistant_messages"] == 2
        assert ov["tool_messages"] == 2

    def test_overview_active_time(self, insights_engine):
        report = insights_engine.generate(days=30)
        ov = report["overview"]
        # sess-1: 200s, sess-2: 100s, sess-3: 100s → total 400s
        assert ov["total_hours"] > 0
        assert ov["avg_session_duration"] > 0

    def test_model_breakdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        models = report["models"]
        assert len(models) == 2  # claude-opus-4-7 + claude-sonnet-4-6
        # claude-opus-4-7 has 2 sessions (sess-1 + sess-3)
        opus = next(m for m in models if m["model"] == "claude-opus-4-7")
        assert opus["sessions"] == 2
        assert opus["input_tokens"] == 15000  # 5000 + 10000
        assert opus["output_tokens"] == 7000  # 2000 + 5000
        assert opus["has_pricing"] is True

    def test_platform_breakdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        platforms = report["platforms"]
        assert len(platforms) == 2  # cli + telegram
        cli = next(p for p in platforms if p["platform"] == "cli")
        assert cli["sessions"] == 2
        telegram = next(p for p in platforms if p["platform"] == "telegram")
        assert telegram["sessions"] == 1

    def test_tool_breakdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        tools = report["tools"]
        tool_names = {t["tool"] for t in tools}
        # read_file + grep from tool_name column; also from tool_calls JSON
        assert "read_file" in tool_names
        assert "grep" in tool_names
        read_file = next(t for t in tools if t["tool"] == "read_file")
        assert read_file["count"] >= 1
        assert read_file["percentage"] > 0

    def test_skill_breakdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        skills = report["skills"]
        top_skills = skills["top_skills"]
        assert len(top_skills) == 1  # python-help
        skill = top_skills[0]
        assert skill["skill"] == "python-help"
        assert skill["view_count"] == 1  # sess-1 skill_view
        assert skill["manage_count"] == 1  # sess-2 skill_manage
        assert skill["total_count"] == 2
        assert skill["last_used_at"] is not None

    def test_activity_patterns(self, insights_engine):
        report = insights_engine.generate(days=30)
        activity = report["activity"]
        assert "by_day" in activity
        assert len(activity["by_day"]) == 7  # Mon-Sun
        assert "by_hour" in activity
        assert len(activity["by_hour"]) == 24
        assert activity["active_days"] >= 1
        assert activity["max_streak"] >= 1

    def test_top_sessions(self, insights_engine):
        report = insights_engine.generate(days=30)
        top = report["top_sessions"]
        # Should have entries for longest / most-messages / most-tokens / most-tools.
        labels = {t["label"] for t in top}
        assert "Longest session" in labels
        assert "Most messages" in labels
        assert "Most tokens" in labels
        assert "Most tool calls" in labels

    def test_source_filter(self, insights_engine):
        report = insights_engine.generate(days=30, source="telegram")
        assert report["overview"]["total_sessions"] == 1

    def test_days_filter_excludes_old_sessions(self, tmp_path):
        conn = _create_test_db(tmp_path)
        now = time.time()
        # Insert a session from 10 days ago.
        _insert_session(
            conn, id="old-sess", started_at=now - 10 * 86400,
            ended_at=now - 10 * 86400 + 100,
        )
        conn.commit()
        fake_db = MagicMock()
        fake_db._conn = conn
        engine = InsightsEngine(db=fake_db)
        # 7-day window should exclude the 10-day-old session.
        report = engine.generate(days=7)
        assert report["empty"] is True
        # 30-day window should include it.
        report_30 = engine.generate(days=30)
        assert report_30["empty"] is False
        conn.close()


# ---------------------------------------------------------------------------
# InsightsEngine — formatting
# ---------------------------------------------------------------------------


class TestInsightsEngineFormatTerminal:
    def test_empty_report(self):
        engine = InsightsEngine(db=MagicMock(_conn=None))
        report = {"empty": True, "days": 30, "source_filter": None}
        result = engine.format_terminal(report)
        assert "No sessions found" in result
        assert "30 days" in result

    def test_populated_report(self, insights_engine):
        report = insights_engine.generate(days=30)
        result = insights_engine.format_terminal(report)
        assert "NIA Insights" in result
        assert "Overview:" in result
        assert "Sessions:" in result
        assert "Models Used:" in result
        assert "Top Tools:" in result

    def test_includes_cost(self, insights_engine):
        report = insights_engine.generate(days=30)
        result = insights_engine.format_terminal(report)
        assert "Est. cost:" in result
        assert "$" in result


class TestInsightsEngineFormatGateway:
    def test_empty_report(self):
        engine = InsightsEngine(db=MagicMock(_conn=None))
        report = {"empty": True, "days": 30, "source_filter": None}
        result = engine.format_gateway(report)
        assert "No sessions found" in result
        assert "30 days" in result

    def test_populated_report_markdown(self, insights_engine):
        report = insights_engine.generate(days=30)
        result = insights_engine.format_gateway(report)
        assert "**NIA Insights**" in result
        assert "**Sessions:**" in result
        assert "**Tokens:**" in result
        # Markdown bold formatting.
        assert result.count("**") >= 4


class TestToJson:
    def test_serializes_report(self, insights_engine):
        report = insights_engine.generate(days=30)
        json_str = to_json(report)
        parsed = json.loads(json_str)
        assert parsed["days"] == 30
        assert parsed["empty"] is False
        assert "overview" in parsed
        assert "models" in parsed

    def test_handles_empty_report(self):
        report = {"empty": True, "days": 7}
        json_str = to_json(report)
        parsed = json.loads(json_str)
        assert parsed["empty"] is True


# ---------------------------------------------------------------------------
# Backwards-compat estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCostBackwardsCompat:
    def test_known_model(self):
        # claude-3-opus: $15/M input, $75/M output.
        assert abs(estimate_cost("claude-3-opus", 1_000_000, 0) - 15.0) < 0.01
        assert abs(estimate_cost("claude-3-opus", 0, 1_000_000) - 75.0) < 0.01

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model", 1_000_000, 0) == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
