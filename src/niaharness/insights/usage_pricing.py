"""Usage pricing — canonical token normalization + per-model cost estimation.

Ported from Hermes Agent's ``agent/usage_pricing.py`` (981 LOC), scoped to
NIA's needs. Provides:

  - :class:`CanonicalUsage` — normalized token bucket from any provider's
    API response (Anthropic / OpenAI Chat Completions / OpenAI Responses).
  - :class:`PricingEntry` — per-model price snapshot (USD per million tokens).
  - :class:`CostResult` — output of :func:`estimate_usage_cost` with amount,
    status (actual/estimated/included/unknown), source, and display label.
  - :func:`estimate_usage_cost` — main cost estimator. Resolves the billing
    route, looks up the pricing entry, computes
    ``Σ (tokens × per_million_rate / 1_000_000)`` for input/output/cache.
  - :func:`resolve_billing_route` — map ``(model, provider, base_url)`` to
    a :class:`BillingRoute` that identifies which pricing table applies.
  - :func:`normalize_usage` — convert a raw API usage object (Anthropic /
    Codex Responses / OpenAI Chat Completions) into a :class:`CanonicalUsage`.

The pricing table (``_OFFICIAL_DOCS_PRICING``) is a static snapshot of
official provider pricing pages, keyed by ``(provider_lowercase,
model_id_lowercase)``. Unknown models return ``CostResult(amount_usd=None,
status="unknown")`` — never a fabricated estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Literal, Optional, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CostStatus = Literal["actual", "estimated", "included", "unknown"]
CostSource = Literal[
    "provider_cost_api",
    "provider_generation_api",
    "provider_models_api",
    "official_docs_snapshot",
    "user_override",
    "custom_contract",
    "none",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE_MILLION = Decimal("1000000")  # Pricing is per-million-tokens.

# Nous Research default base URL (OpenAI-compatible).
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"

_UTC_NOW = lambda: datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalUsage:
    """Normalized token bucket from any provider's API response.

    All token counts are non-negative integers. ``raw_usage`` preserves the
    original API response for debugging / re-normalization. ``request_count``
    defaults to 1 (a single API call); use :meth:`__add__` to merge multiple
    calls (e.g. MoA advisor fan-out + aggregator).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1
    raw_usage: Optional[dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> int:
        """Total prompt-side tokens = input + cache_read + cache_write."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        """Grand total = prompt + output."""
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: "CanonicalUsage") -> "CanonicalUsage":
        """Sum two usage buckets.

        ``raw_usage`` is dropped on the sum — it describes a single API
        response and cannot be meaningfully merged. ``request_count`` adds
        so callers can see how many underlying API calls a combined figure
        covers.
        """
        if not isinstance(other, CanonicalUsage):
            return NotImplemented
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            request_count=self.request_count + other.request_count,
            raw_usage=None,
        )


@dataclass(frozen=True)
class BillingRoute:
    """Identifies which pricing table & lookup path applies to a model.

    Attributes:
        provider: Lowercased provider name (``"anthropic"``, ``"openai"``,
            ``"openrouter"``, ``"bedrock"``, ``"google"``, ``"deepseek"``,
            ``"nous"``, ``"vertex"``, ``"subscription_included"``, etc.).
        model: Model id with vendor prefix stripped (``"claude-opus-4-7"``
            not ``"anthropic/claude-opus-4-7"``).
        base_url: The base URL the call was made against (for OpenRouter /
            Nous / Vertex routing).
        billing_mode: How this route is billed — ``"subscription_included"``
            (zero-cost, e.g. Claude Code OAuth), ``"official_models_api"``
            (live /models endpoint), ``"official_docs_snapshot"`` (static
            pricing table), ``"unknown"``.
    """

    provider: str
    model: str
    base_url: str = ""
    billing_mode: str = "unknown"


@dataclass(frozen=True)
class PricingEntry:
    """Per-model price snapshot. All monetary fields are USD per million tokens.

    Attributes:
        input_cost_per_million: Cost per 1M input tokens.
        output_cost_per_million: Cost per 1M output tokens.
        cache_read_cost_per_million: Cost per 1M cache-read tokens (Anthropic
            prompt caching; ~10x cheaper than input).
        cache_write_cost_per_million: Cost per 1M cache-write tokens
            (Anthropic prompt caching; ~25% more than input).
        request_cost: Per-request flat fee (always None in the snapshot table).
        source: Where the pricing came from (see :data:`CostSource`).
        source_url: URL of the pricing page (for audit).
        pricing_version: Version tag (e.g. ``"anthropic-pricing-2026-05"``).
        fetched_at: When the pricing was fetched (None for static snapshot).
    """

    input_cost_per_million: Optional[Decimal] = None
    output_cost_per_million: Optional[Decimal] = None
    cache_read_cost_per_million: Optional[Decimal] = None
    cache_write_cost_per_million: Optional[Decimal] = None
    request_cost: Optional[Decimal] = None
    source: CostSource = "none"
    source_url: Optional[str] = None
    pricing_version: Optional[str] = None
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class CostResult:
    """Output of :func:`estimate_usage_cost`.

    Attributes:
        amount_usd: The estimated cost in USD, or ``None`` if unknown.
        status: ``"actual"`` (provider-reported), ``"estimated"`` (computed
            from pricing table), ``"included"`` (subscription, zero-cost),
            ``"unknown"`` (no pricing data).
        source: Where the pricing came from (see :data:`CostSource`).
        label: Display string like ``"~$1.23"`` or ``"n/a"`` or ``"included"``.
        fetched_at: When the pricing was fetched (None for static snapshot).
        pricing_version: Version tag (e.g. ``"anthropic-pricing-2026-05"``).
        notes: Additional context tuples (e.g. cache-pricing-unavailable).
    """

    amount_usd: Optional[Decimal]
    status: CostStatus
    source: CostSource
    label: str
    fetched_at: Optional[datetime] = None
    pricing_version: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Coerce any value to ``Decimal``, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_int(value: Any) -> int:
    """Coerce to int, defaulting 0 on None / failure."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


# ---------------------------------------------------------------------------
# Billing-route resolution
# ---------------------------------------------------------------------------


def _base_url_host_matches(base_url: Any, host_substring: str) -> bool:
    """Return True if *base_url*'s host contains *host_substring*."""
    if not base_url:
        return False
    try:
        url = str(base_url)
        if "://" not in url:
            url = f"https://{url}"
        parsed = urlparse(url)
        return host_substring.lower() in (parsed.hostname or "").lower()
    except Exception:
        return False


def resolve_billing_route(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BillingRoute:
    """Map ``(model, provider, base_url)`` → :class:`BillingRoute`.

    Infers provider from the model name prefix if it contains ``/`` and
    starts with ``anthropic``/``openai``/``google``. Routes OpenRouter /
    Nous / Vertex by ``base_url`` host. Strips the vendor prefix from
    ``model`` for downstream pricing lookups.

    For NIA's needs, we recognize these billing modes:
      - ``"subscription_included"`` — provider is ``"claude-code-oauth"`` or
        ``"openai-codex"`` (subscription auth, zero per-token cost).
      - ``"official_docs_snapshot"`` — static pricing table lookup.
      - ``"unknown"`` — no pricing data available.
    """
    model = (model_name or "").strip()
    base_url = (base_url or "").strip()
    provider_lower = (provider or "").strip().lower()

    # Infer provider from model prefix (e.g. "anthropic/claude-opus-4-7").
    if "/" in model and not provider_lower:
        prefix = model.split("/", 1)[0].lower()
        if prefix in {"anthropic", "openai", "google", "deepseek", "bedrock"}:
            provider_lower = prefix
            model = model.split("/", 1)[1]

    # Route by base_url host.
    if not provider_lower:
        if _base_url_host_matches(base_url, "openrouter.ai"):
            provider_lower = "openrouter"
        elif _base_url_host_matches(base_url, "nousresearch.com"):
            provider_lower = "nous"
        elif _base_url_host_matches(base_url, "vertex"):
            provider_lower = "vertex"
        elif _base_url_host_matches(base_url, "bedrock") or _base_url_host_matches(base_url, "aws"):
            provider_lower = "bedrock"

    # Subscription-included routes (zero per-token cost).
    if provider_lower in {"claude-code-oauth", "openai-codex", "subscription_included"}:
        return BillingRoute(
            provider="subscription_included",
            model=model,
            base_url=base_url,
            billing_mode="subscription_included",
        )

    # Default provider inference from model name.
    if not provider_lower:
        model_lower = model.lower()
        if model_lower.startswith("claude-") or model_lower.startswith("anthropic"):
            provider_lower = "anthropic"
        elif model_lower.startswith("gpt-") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            provider_lower = "openai"
        elif model_lower.startswith("gemini-"):
            provider_lower = "google"
        elif model_lower.startswith("deepseek-"):
            provider_lower = "deepseek"
        elif model_lower.startswith("grok-"):
            provider_lower = "xai"
        elif model_lower.startswith("llama-") or model_lower.startswith("mixtral"):
            provider_lower = "openrouter"  # Usually routed via OpenRouter.

    return BillingRoute(
        provider=provider_lower or "unknown",
        model=model,
        base_url=base_url,
        billing_mode="official_docs_snapshot" if provider_lower else "unknown",
    )


# ---------------------------------------------------------------------------
# Model-name normalization
# ---------------------------------------------------------------------------


def _normalize_anthropic_model_name(model: str) -> str:
    """Normalize Anthropic model name variants to canonical form.

    Handles:
      - Strips ``anthropic/`` prefix if present.
      - Dot notation: ``claude-opus-4.7`` → ``claude-opus-4-7``.
    """
    name = (model or "").lower().strip()
    if name.startswith("anthropic/"):
        name = name[len("anthropic/"):]
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    return name


def _normalize_bedrock_model_name(model: str) -> str:
    """Normalize a Bedrock model id to its bare foundation-model form.

    Strips region prefixes (``us.`` / ``global.`` / ``eu.`` / ``ap.`` /
    ``jp.``) and normalizes dot-notation version numbers.
    """
    name = (model or "").lower().strip()
    for prefix in ("us.", "global.", "eu.", "ap.", "jp."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    return name


# ---------------------------------------------------------------------------
# Official-docs pricing table (static snapshot)
# ---------------------------------------------------------------------------

_OFFICIAL_DOCS_PRICING: Dict[Tuple[str, str], PricingEntry] = {
    # ── Anthropic Claude 4.8 ─────────────────────────────────────────────
    ("anthropic", "claude-opus-4-8"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-opus-4-8-fast"): PricingEntry(
        input_cost_per_million=Decimal("10.00"),
        output_cost_per_million=Decimal("50.00"),
        cache_read_cost_per_million=Decimal("1.00"),
        cache_write_cost_per_million=Decimal("12.50"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/anthropic/claude-opus-4.8-fast",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.7 ─────────────────────────────────────────────
    ("anthropic", "claude-opus-4-7"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-opus-4-7-20250507"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-7"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.6 ─────────────────────────────────────────────
    ("anthropic", "claude-opus-4-6"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-opus-4-6-20250414"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-6"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-6-20250514"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.5 ─────────────────────────────────────────────
    ("anthropic", "claude-opus-4-5"): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-5"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-haiku-4-5"): PricingEntry(
        input_cost_per_million=Decimal("1.00"),
        output_cost_per_million=Decimal("5.00"),
        cache_read_cost_per_million=Decimal("0.10"),
        cache_write_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4 / 4.1 ─────────────────────────────────────────
    ("anthropic", "claude-opus-4-1"): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-1"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-sonnet-4-20250514"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-opus-4-20250514"): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 3.5 / 3 ─────────────────────────────────────────
    ("anthropic", "claude-3-5-sonnet"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-3-5-haiku"): PricingEntry(
        input_cost_per_million=Decimal("0.80"),
        output_cost_per_million=Decimal("4.00"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-3-opus"): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    ("anthropic", "claude-3-haiku"): PricingEntry(
        input_cost_per_million=Decimal("0.25"),
        output_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── OpenAI ───────────────────────────────────────────────────────────
    ("openai", "gpt-4o"): PricingEntry(
        input_cost_per_million=Decimal("2.50"),
        output_cost_per_million=Decimal("10.00"),
        cache_read_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "gpt-4o-mini"): PricingEntry(
        input_cost_per_million=Decimal("0.15"),
        output_cost_per_million=Decimal("0.60"),
        cache_read_cost_per_million=Decimal("0.075"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "gpt-4.1"): PricingEntry(
        input_cost_per_million=Decimal("2.00"),
        output_cost_per_million=Decimal("8.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "gpt-4.1-mini"): PricingEntry(
        input_cost_per_million=Decimal("0.40"),
        output_cost_per_million=Decimal("1.60"),
        cache_read_cost_per_million=Decimal("0.10"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "gpt-4.1-nano"): PricingEntry(
        input_cost_per_million=Decimal("0.10"),
        output_cost_per_million=Decimal("0.40"),
        cache_read_cost_per_million=Decimal("0.025"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "o3"): PricingEntry(
        input_cost_per_million=Decimal("10.00"),
        output_cost_per_million=Decimal("40.00"),
        cache_read_cost_per_million=Decimal("2.50"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    ("openai", "o3-mini"): PricingEntry(
        input_cost_per_million=Decimal("1.10"),
        output_cost_per_million=Decimal("4.40"),
        cache_read_cost_per_million=Decimal("0.55"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    # ── DeepSeek ─────────────────────────────────────────────────────────
    ("deepseek", "deepseek-chat"): PricingEntry(
        input_cost_per_million=Decimal("0.14"),
        output_cost_per_million=Decimal("0.28"),
        source="official_docs_snapshot",
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_version="deepseek-pricing-2026-05-12",
    ),
    ("deepseek", "deepseek-reasoner"): PricingEntry(
        input_cost_per_million=Decimal("0.55"),
        output_cost_per_million=Decimal("2.19"),
        source="official_docs_snapshot",
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_version="deepseek-pricing-2026-05-12",
    ),
    # ── Google ───────────────────────────────────────────────────────────
    ("google", "gemini-2.5-pro"): PricingEntry(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("10.00"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    ("google", "gemini-2.5-flash"): PricingEntry(
        input_cost_per_million=Decimal("0.075"),
        output_cost_per_million=Decimal("0.30"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    ("google", "gemini-2.0-flash"): PricingEntry(
        input_cost_per_million=Decimal("0.10"),
        output_cost_per_million=Decimal("0.40"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    # ── Bedrock ──────────────────────────────────────────────────────────
    ("bedrock", "anthropic.claude-opus-4-6"): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    ("bedrock", "anthropic.claude-sonnet-4-6"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    ("bedrock", "anthropic.claude-sonnet-4-5"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    ("bedrock", "anthropic.claude-haiku-4-5"): PricingEntry(
        input_cost_per_million=Decimal("0.80"),
        output_cost_per_million=Decimal("4.00"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    # ── xAI ──────────────────────────────────────────────────────────────
    ("xai", "grok-3"): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        source="official_docs_snapshot",
        source_url="https://x.ai/api",
        pricing_version="xai-pricing-2026-03",
    ),
    ("xai", "grok-3-mini"): PricingEntry(
        input_cost_per_million=Decimal("0.20"),
        output_cost_per_million=Decimal("0.50"),
        source="official_docs_snapshot",
        source_url="https://x.ai/api",
        pricing_version="xai-pricing-2026-03",
    ),
}


def _lookup_official_docs_pricing(route: BillingRoute) -> Optional[PricingEntry]:
    """Direct lookup in the static pricing table with normalization fallback."""
    model = route.model.lower()
    entry = _OFFICIAL_DOCS_PRICING.get((route.provider, model))
    if entry:
        return entry
    # Try Anthropic normalization (dot → dash).
    if route.provider == "anthropic":
        normalized = _normalize_anthropic_model_name(model)
        if normalized != model:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, normalized))
            if entry:
                return entry
    # Try Bedrock normalization (strip region prefix + dot → dash).
    if route.provider == "bedrock":
        normalized = _normalize_bedrock_model_name(model)
        if normalized != model:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, normalized))
            if entry:
                return entry
    return None


def get_pricing_entry(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[PricingEntry]:
    """Top-level pricing lookup.

    Order:
      1. ``subscription_included`` route → zero-cost entry.
      2. Static ``_OFFICIAL_DOCS_PRICING`` snapshot (with Anthropic / Bedrock
         name normalization).

    NIA does not currently fetch live pricing from OpenRouter / endpoint
    ``/models`` APIs (Hermes does, but NIA's architecture differs). If you
    need live pricing, add a fetch step here.
    """
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)

    if route.billing_mode == "subscription_included":
        return PricingEntry(
            source="none",
            pricing_version="included-route",
        )

    return _lookup_official_docs_pricing(route)


def has_known_pricing(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """True if pricing entry exists or route is subscription_included."""
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return True
    return _lookup_official_docs_pricing(route) is not None


# ---------------------------------------------------------------------------
# Usage normalization (raw API response → CanonicalUsage)
# ---------------------------------------------------------------------------


def normalize_usage(
    response_usage: Any,
    *,
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> CanonicalUsage:
    """Convert a raw API usage object into a :class:`CanonicalUsage`.

    Handles three response shapes:
      - **Anthropic Messages** — ``input_tokens``, ``output_tokens``,
        ``cache_read_input_tokens``, ``cache_creation_input_tokens``.
      - **OpenAI Chat Completions** — ``prompt_tokens``, ``completion_tokens``,
        ``prompt_tokens_details.cached_tokens``,
        ``completion_tokens_details.reasoning_tokens``.
      - **OpenAI Responses (Codex)** — ``input_tokens``, ``output_tokens``,
        ``input_tokens_details.cached_tokens``,
        ``output_tokens_details.reasoning_tokens``.

    Also handles OpenAI-compat proxies that route Claude: falls back to
    top-level Anthropic fields (``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``) when ``prompt_tokens_details.cached_tokens``
    is 0.
    """
    if response_usage is None:
        return CanonicalUsage(input_tokens=0, output_tokens=0, request_count=1, raw_usage=None)

    # Accept both dict and SDK object (use model_dump / vars if available).
    if hasattr(response_usage, "model_dump"):
        raw = response_usage.model_dump()
    elif isinstance(response_usage, dict):
        raw = response_usage
    elif hasattr(response_usage, "__dict__"):
        raw = vars(response_usage)
    else:
        raw = {"_raw": str(response_usage)}

    input_tokens = _to_int(raw.get("input_tokens") or raw.get("prompt_tokens"))
    output_tokens = _to_int(raw.get("output_tokens") or raw.get("completion_tokens"))

    # Cache tokens — Anthropic shape.
    cache_read = _to_int(
        raw.get("cache_read_input_tokens")
        or raw.get("cache_read_tokens")
    )
    cache_write = _to_int(
        raw.get("cache_creation_input_tokens")
        or raw.get("cache_write_input_tokens")
        or raw.get("cache_write_tokens")
    )

    # Cache tokens — OpenAI shape (prompt_tokens_details.cached_tokens).
    if cache_read == 0:
        prompt_details = raw.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cache_read = _to_int(prompt_details.get("cached_tokens"))
        elif hasattr(prompt_details, "cached_tokens"):
            cache_read = _to_int(getattr(prompt_details, "cached_tokens"))
    # Codex Responses shape: input_tokens_details.cached_tokens.
    if cache_read == 0:
        input_details = raw.get("input_tokens_details")
        if isinstance(input_details, dict):
            cache_read = _to_int(input_details.get("cached_tokens"))
        elif hasattr(input_details, "cached_tokens"):
            cache_read = _to_int(getattr(input_details, "cached_tokens"))

    # Reasoning tokens.
    reasoning = 0
    output_details = raw.get("output_tokens_details") or raw.get("completion_tokens_details")
    if isinstance(output_details, dict):
        reasoning = _to_int(output_details.get("reasoning_tokens"))
    elif hasattr(output_details, "reasoning_tokens"):
        reasoning = _to_int(getattr(output_details, "reasoning_tokens"))

    return CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        request_count=1,
        raw_usage=raw if isinstance(raw, dict) else None,
    )


# ---------------------------------------------------------------------------
# Cost estimation (main entry point)
# ---------------------------------------------------------------------------


def estimate_usage_cost(
    model_name: str,
    usage: CanonicalUsage,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CostResult:
    """Estimate the cost of a model invocation.

    Resolves the billing route, looks up the pricing entry, computes
    ``Σ (tokens × per_million_rate / 1_000_000)`` for input / output /
    cache_read / cache_write, plus ``request_count × request_cost``.

    Returns:
        :class:`CostResult` with ``status``:
          - ``"included"`` — subscription route, zero cost.
          - ``"estimated"`` — computed from pricing table.
          - ``"unknown"`` — no pricing data for this route.
    """
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)

    if route.billing_mode == "subscription_included":
        return CostResult(
            amount_usd=_ZERO,
            status="included",
            source="none",
            label="included",
            pricing_version="included-route",
        )

    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    if not entry:
        return CostResult(
            amount_usd=None,
            status="unknown",
            source="none",
            label="n/a",
        )

    notes: list[str] = []
    amount = _ZERO

    # Validate that we have pricing for non-zero token buckets.
    if usage.input_tokens and entry.input_cost_per_million is None:
        return CostResult(
            amount_usd=None, status="unknown", source=entry.source, label="n/a",
        )
    if usage.output_tokens and entry.output_cost_per_million is None:
        return CostResult(
            amount_usd=None, status="unknown", source=entry.source, label="n/a",
        )
    if usage.cache_read_tokens and entry.cache_read_cost_per_million is None:
        return CostResult(
            amount_usd=None, status="unknown", source=entry.source, label="n/a",
            notes=("cache-read pricing unavailable for route",),
        )
    if usage.cache_write_tokens and entry.cache_write_cost_per_million is None:
        return CostResult(
            amount_usd=None, status="unknown", source=entry.source, label="n/a",
            notes=("cache-write pricing unavailable for route",),
        )

    # Compute cost.
    if entry.input_cost_per_million is not None:
        amount += Decimal(usage.input_tokens) * entry.input_cost_per_million / _ONE_MILLION
    if entry.output_cost_per_million is not None:
        amount += Decimal(usage.output_tokens) * entry.output_cost_per_million / _ONE_MILLION
    if entry.cache_read_cost_per_million is not None:
        amount += Decimal(usage.cache_read_tokens) * entry.cache_read_cost_per_million / _ONE_MILLION
    if entry.cache_write_cost_per_million is not None:
        amount += Decimal(usage.cache_write_tokens) * entry.cache_write_cost_per_million / _ONE_MILLION
    if entry.request_cost is not None and usage.request_count:
        amount += Decimal(usage.request_count) * entry.request_cost

    status: CostStatus = "estimated"
    label = f"~${amount:.2f}"
    if entry.source == "none" and amount == _ZERO:
        status = "included"
        label = "included"

    if route.provider == "openrouter":
        notes.append("OpenRouter cost is estimated from the models API until reconciled.")

    return CostResult(
        amount_usd=amount,
        status=status,
        source=entry.source,
        label=label,
        fetched_at=entry.fetched_at,
        pricing_version=entry.pricing_version,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration_compact(seconds: float) -> str:
    """Render seconds as ``45s`` / ``12m`` / ``3h 15m`` / ``2.5d``."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        if minutes:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
    days = seconds / 86400
    if days < 10:
        return f"{days:.1f}d"
    return f"{int(days)}d"


def format_token_count_compact(value: int) -> str:
    """Render token counts as K / M / B with adaptive precision.

    Strips trailing ``.0`` from the number BEFORE appending the suffix, so
    ``10000`` → ``"10K"`` (not ``"10.0K"``) and ``1500`` → ``"1.5K"``.
    """
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        num = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{num}K"
    if value < 1_000_000_000:
        num = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{num}M"
    num = f"{value / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
    return f"{num}B"


__all__ = [
    "BillingRoute",
    "CanonicalUsage",
    "CostResult",
    "CostSource",
    "CostStatus",
    "PricingEntry",
    "estimate_usage_cost",
    "format_duration_compact",
    "format_token_count_compact",
    "get_pricing_entry",
    "has_known_pricing",
    "normalize_usage",
    "resolve_billing_route",
]
