"""Auxiliary model fallback chain — multi-provider routing with failover.

Ported from Hermes Agent's agent/auxiliary_client.py (7,161 LOC), scoped
to NIA's needs. Provides:

  - **Provider chain** — ordered list of fallback providers tried when the
    primary aux model fails (OpenRouter → custom endpoint → primary API key)
  - **Error classifiers** — structured detection of payment/auth/rate-limit/
    connection/model-not-found errors, each driving a different fallback policy
  - **Unhealthy-provider cache** — recently-402'd providers are skipped for
    10 minutes to avoid burning RTTs on depleted accounts
  - **Payment fallback** — on 402/credit exhaustion, walks the chain skipping
    the failed provider
  - **Per-task overrides** — ``auxiliary.<task>.provider`` / ``auxiliary.<task>.model``
    config-file overrides (compression, vision, title_generation, etc.)

Usage::

    from niaharness.auxiliary.chain import call_with_fallback

    result = await call_with_fallback(
        primary_client=my_aux_client,
        prompt="Summarize this conversation...",
        task="compression",
        max_tokens=1024,
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unhealthy-provider cache (ported from Hermes)
# ---------------------------------------------------------------------------

_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}

# Map provider names to chain labels.
_AUX_UNHEALTHY_LABEL_ALIASES = {
    "openrouter": "openrouter",
    "custom": "local/custom",
    "local/custom": "local/custom",
    "anthropic": "anthropic",
    "openai": "openai",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "zai": "zai",
}


def _normalize_chain_label(provider: str) -> str:
    """Normalize a provider name to a chain label."""
    if not provider:
        return ""
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)


def mark_provider_unhealthy(provider: str, ttl: Optional[float] = None) -> None:
    """Mark a provider as recently-failed, hidden from chain iteration until TTL expires.

    Ported from Hermes _mark_provider_unhealthy. Called after a confirmed
    payment/auth error to skip the provider on subsequent aux calls.
    """
    label = _normalize_chain_label(provider)
    if not label:
        return
    expires_at = time.time() + (ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS)
    _aux_unhealthy_until[label] = expires_at
    logger.warning(
        "Auxiliary: marking %s unhealthy for %ds. Subsequent calls will skip it.",
        label,
        int(ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS),
    )


def is_provider_unhealthy(label: str) -> bool:
    """True if the provider is in the unhealthy cache and TTL hasn't expired.

    Ported from Hermes _is_provider_unhealthy. Lazily evicts expired entries.
    """
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True


def _log_skip_unhealthy(label: str, task: Optional[str] = None) -> None:
    """Emit a single info-level log per minute when skipping an unhealthy provider."""
    now = time.time()
    last = _aux_unhealthy_logged_at.get(label, 0.0)
    if now - last >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info(
            "Auxiliary %s: skipping %s (recently failed, retry in %ds)",
            task or "call", label, max(0, int(expires_at - now)),
        )


def reset_unhealthy_cache() -> None:
    """Clear the unhealthy cache (for tests / manual reset)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()


# ---------------------------------------------------------------------------
# Error classifiers (ported from Hermes)
# ---------------------------------------------------------------------------


def is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Ported from Hermes _is_payment_error. Returns True for HTTP 402 and for
    429/403/404/None errors whose message indicates billing exhaustion or
    daily quota exhaustion.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    if status in {402, 403, 404, 429, None}:
        if any(kw in err_lower for kw in (
            "credits", "insufficient funds",
            "can only afford", "billing",
            "payment required",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
            "requires a subscription", "upgrade for access",
            "upgrade for higher limits", "reached your session usage limit",
            "quota exceeded", "quota_exceeded",
            "too many tokens per day", "daily limit",
            "tokens per day", "daily quota",
            "resource exhausted",
            "weekly usage limit", "weekly limit",
        )):
            return True
    return False


def is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider fallback.

    Ported from Hermes _is_auth_error.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    if status == 403 and "bad-credentials" in err_lower:
        return True
    if "unauthenticated" in err_lower and "bad-credentials" in err_lower:
        return True
    return False


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors that warrant provider fallback.

    Ported from Hermes _is_rate_limit_error. Distinguishes rate-limit from
    billing (billing keywords are handled by is_payment_error).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    err_lower = str(exc).lower()

    if type(exc).__name__ == "RateLimitError":
        return True

    if status == 429:
        if any(kw in err_lower for kw in (
            "rate limit", "rate_limit", "too many requests",
            "try again", "retry after", "resets in",
        )):
            return True
        # Generic 429 without billing keywords = likely a rate limit
        if not any(kw in err_lower for kw in (
            "credits", "insufficient funds", "billing",
            "payment required", "can only afford",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
        )):
            return True
    return False


def is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Ported from Hermes _is_connection_error. Returns True for DNS failures,
    connection refused, TLS errors, timeouts, streaming premature-close.
    """
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
        "incomplete chunked read",
        "peer closed connection",
        "response ended prematurely",
        "unexpected eof",
        "remoteprotocolerror",
        "localprotocolerror",
    )):
        return True
    return False


def is_model_not_found_error(exc: Exception) -> bool:
    """Detect "model doesn't exist" errors.

    Ported from Hermes _is_model_not_found_error. Excludes billing keywords
    (those belong to is_payment_error).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "free tier", "free-tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return any(kw in err_lower for kw in (
        "model does not exist",
        "does not exist in our configuration",
        "is not a valid model",
        "no such model",
        "model not found",
        "the model `",
        "model_not_found",
        "unknown model",
    ))


def is_transient_transport_error(exc: Exception) -> bool:
    """Return True for a one-off transport blip worth retrying on the same provider.

    Ported from Hermes _is_transient_transport_error. Covers connection errors
    + pure 5xx/408 HTTP status.
    """
    if is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


# ---------------------------------------------------------------------------
# Provider chain (ported from Hermes _get_provider_chain)
# ---------------------------------------------------------------------------


def _get_provider_chain() -> List[Tuple[str, Callable[[], Tuple[Optional[Any], Optional[str]]]]]:
    """Return the ordered provider detection chain.

    Built at call time (not module level) so test patches are picked up.
    Each entry is (label, try_fn) where try_fn returns (client, model) or
    (None, None).

    Ported from Hermes _get_provider_chain, adapted for NIA's providers.
    """
    return [
        ("openrouter", _try_openrouter),
        ("local/custom", _try_custom_endpoint),
        ("api-key", _try_api_key_provider),
    ]


def _try_openrouter() -> Tuple[Optional[Any], Optional[str]]:
    """Try OpenRouter as a fallback aux provider."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None, None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
        )
        # Use a cheap model on OpenRouter
        model = os.environ.get("NIA_AUX_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        return client, model
    except Exception as exc:
        logger.debug("OpenRouter aux fallback failed: %s", exc)
        return None, None


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    """Try a custom endpoint as a fallback aux provider."""
    base_url = os.environ.get("NIA_AUX_BASE_URL", "").strip()
    api_key = os.environ.get("NIA_AUX_API_KEY", "").strip()
    model = os.environ.get("NIA_AUX_MODEL", "").strip()
    if not base_url or not api_key or not model:
        return None, None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        return client, model
    except Exception as exc:
        logger.debug("Custom endpoint aux fallback failed: %s", exc)
        return None, None


def _try_api_key_provider() -> Tuple[Optional[Any], Optional[str]]:
    """Try using any available API key provider as a fallback.

    Checks all known API key env vars and creates a client for the first
    one that has both a key and a known model.
    """
    # Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=key, max_retries=0)
            return client, "claude-3-haiku-20240307"
        except Exception:
            pass

    # OpenAI
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key, max_retries=0)
            return client, "gpt-4o-mini"
        except Exception:
            pass

    # DeepSeek
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=key, base_url="https://api.deepseek.com/v1", max_retries=0
            )
            return client, "deepseek-chat"
        except Exception:
            pass

    # Groq
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=key, base_url="https://api.groq.com/openai/v1", max_retries=0
            )
            return client, "llama-3.3-70b-versatile"
        except Exception:
            pass

    return None, None


# ---------------------------------------------------------------------------
# Payment fallback (ported from Hermes _try_payment_fallback)
# ---------------------------------------------------------------------------


def try_payment_fallback(
    failed_provider: str,
    task: str = "default",
    reason: str = "payment error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit or connection error.

    Ported from Hermes _try_payment_fallback. Iterates the provider chain,
    skipping the failed provider and any unhealthy entries.

    Args:
        failed_provider: The provider that just failed.
        task: The auxiliary task name (for logging).
        reason: Why the fallback was triggered (for logging).

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    skip = (failed_provider or "").lower().strip()
    skip_labels = {skip}
    # Map common resolved_provider values back to chain labels
    for alias, label in _AUX_UNHEALTHY_LABEL_ALIASES.items():
        if skip == alias:
            skip_labels.add(label)

    tried: List[str] = []
    for label, try_fn in _get_provider_chain():
        if label in skip_labels:
            continue
        if is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — falling back to %s (%s)",
                task, reason, failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: %s on %s and no fallback available (tried: %s)",
        task, reason, failed_provider, ", ".join(tried),
    )
    return None, None, ""


# ---------------------------------------------------------------------------
# Per-task config overrides (ported from Hermes auxiliary.<task>.provider)
# ---------------------------------------------------------------------------


def get_task_config(task: Optional[str] = None) -> Optional[Tuple[str, str, Optional[str], str]]:
    """Resolve per-task auxiliary config override.

    Ported from Hermes per-task overrides. Checks:
      1. ``NIA_AUX_<TASK>_MODEL`` / ``NIA_AUX_<TASK>_API_KEY`` env vars
      2. ``auxiliary.<task>.model`` / ``auxiliary.<task>.api_key`` in config

    Returns (model, api_key, base_url, provider) or None if no override.
    """
    if not task:
        return None

    task_upper = task.upper()
    model = os.environ.get(f"NIA_AUX_{task_upper}_MODEL", "").strip()
    api_key = os.environ.get(f"NIA_AUX_{task_upper}_API_KEY", "").strip()
    base_url = os.environ.get(f"NIA_AUX_{task_upper}_BASE_URL", "").strip() or None
    provider = os.environ.get(f"NIA_AUX_{task_upper}_PROVIDER", "").strip() or "openai"

    if model and api_key:
        return model, api_key, base_url, provider

    # Config file
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        aux_section = getattr(settings, "auxiliary", None) or {}
        if isinstance(aux_section, dict):
            task_section = aux_section.get(task, {})
            if isinstance(task_section, dict):
                model = task_section.get("model", "").strip()
                api_key = task_section.get("api_key", "").strip()
                base_url = task_section.get("base_url", "").strip() or None
                provider = task_section.get("provider", "openai").strip()
                if model and api_key:
                    return model, api_key, base_url, provider
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Call with fallback (the main entry point)
# ---------------------------------------------------------------------------


async def call_with_fallback(
    primary_client: Any,
    prompt: str,
    *,
    task: str = "default",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: Optional[str] = None,
    provider_label: str = "",
) -> Optional[str]:
    """Call the primary aux client with automatic fallback on failure.

    Ported from Hermes call_llm fallback pattern. On payment/auth/rate-limit/
    connection errors, tries fallback providers from the chain.

    Args:
        primary_client: The primary AuxiliaryClient instance.
        prompt: The prompt to send.
        task: The task name (for logging + per-task config).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        system: Optional system prompt.
        provider_label: The provider label of the primary client (for skip logic).

    Returns:
        The completion text, or None if all providers fail.
    """
    # Try the primary client first.
    try:
        return await primary_client.complete(
            prompt, max_tokens=max_tokens, temperature=temperature, system=system
        )
    except Exception as exc:
        # Classify the error.
        if is_payment_error(exc):
            logger.warning(
                "Auxiliary %s: payment error on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
            mark_provider_unhealthy(provider_label or "primary")
        elif is_auth_error(exc):
            logger.warning(
                "Auxiliary %s: auth error on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
            mark_provider_unhealthy(provider_label or "primary")
        elif is_rate_limit_error(exc):
            logger.warning(
                "Auxiliary %s: rate limit on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
            # Don't mark unhealthy — rate limits are transient
        elif is_connection_error(exc):
            logger.warning(
                "Auxiliary %s: connection error on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
            # Don't mark unhealthy — connection errors are transient
        elif is_model_not_found_error(exc):
            logger.warning(
                "Auxiliary %s: model not found on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
        elif is_transient_transport_error(exc):
            # Retry once on transient errors before falling back
            logger.warning(
                "Auxiliary %s: transient error on %s, retrying: %s",
                task, provider_label or "primary", str(exc)[:200],
            )
            try:
                return await primary_client.complete(
                    prompt, max_tokens=max_tokens, temperature=temperature, system=system
                )
            except Exception:
                pass  # Fall through to provider fallback
        else:
            logger.warning(
                "Auxiliary %s: unclassified error on %s: %s",
                task, provider_label or "primary", str(exc)[:200],
            )

    # Try fallback providers.
    fallback_client, fallback_model, fallback_label = try_payment_fallback(
        provider_label or "primary", task=task, reason="aux error"
    )

    if fallback_client is not None:
        try:
            # Build a temporary AuxiliaryClient for the fallback
            from niaharness.auxiliary import AuxConfig, AuxiliaryClient

            fallback_config = AuxConfig(
                model=fallback_model or "gpt-4o-mini",
                api_key=None,  # The client is already constructed
                provider="openai",  # Fallback clients are always OpenAI-compatible
            )
            # HACK: inject the pre-built client directly
            temp_aux = AuxiliaryClient(fallback_config)
            temp_aux._client = fallback_client
            return await temp_aux.complete(
                prompt, max_tokens=max_tokens, temperature=temperature, system=system
            )
        except Exception as exc2:
            logger.warning(
                "Auxiliary %s: fallback %s also failed: %s",
                task, fallback_label, str(exc2)[:200],
            )
            if is_payment_error(exc2):
                mark_provider_unhealthy(fallback_label)

    logger.error("Auxiliary %s: all providers failed", task)
    return None


__all__ = [
    "call_with_fallback",
    "get_task_config",
    "is_auth_error",
    "is_connection_error",
    "is_model_not_found_error",
    "is_payment_error",
    "is_rate_limit_error",
    "is_transient_transport_error",
    "is_provider_unhealthy",
    "mark_provider_unhealthy",
    "reset_unhealthy_cache",
    "try_payment_fallback",
]
