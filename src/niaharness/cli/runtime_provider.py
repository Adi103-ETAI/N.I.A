"""Runtime provider resolution.

Focused extraction from hermes-agent/hermes_cli/runtime_provider.py (2,058 LOC).

The full Hermes module has OAuth device-code flows, credential pools, Azure
Foundry, Vertex AI, Codex Responses API, and Nous Portal. This port covers
the essential ``resolve_runtime_provider()`` API that ``setup.runtime_check``
depends on — it resolves the configured provider's API key + base_url from
config.yaml + env vars.

For OAuth / credential-pool / Azure / Vertex providers, it falls back to
checking env vars directly (same as the tui_gateway deep-port's fallback).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _getenv(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_model_config() -> Dict[str, Any]:
    """Load the model config from config.yaml."""
    try:
        from niaharness.tui_gateway.server import _load_cfg
        cfg = _load_cfg()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def _get_provider_config() -> Dict[str, Any]:
    """Load the agent/provider config from config.yaml."""
    try:
        from niaharness.tui_gateway.server import _load_cfg
        cfg = _load_cfg()
        agent_cfg = cfg.get("agent", {})
        if isinstance(agent_cfg, dict):
            return agent_cfg
    except Exception:
        pass
    return {}


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve the requested provider string to a canonical provider id.

    Ported from hermes-agent/hermes_cli/runtime_provider.py (simplified).
    """
    if requested and requested.strip():
        return requested.strip().lower()

    # Check config.yaml for explicit provider.
    model_cfg = _get_model_config()
    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    if cfg_provider and cfg_provider != "auto":
        return cfg_provider

    # Auto-detect from env vars.
    from niaharness.cli.auth import PROVIDER_REGISTRY, has_usable_secret

    for pid, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        for env_var in pconfig.api_key_env_vars:
            val = _getenv(env_var)
            if has_usable_secret(val):
                return pid

    return "auto"


def _resolve_api_key_provider(
    provider: str,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve an API-key provider's runtime credentials.

    Returns a runtime dict or None if the provider isn't an api_key type.
    """
    from niaharness.cli.auth import PROVIDER_REGISTRY, has_usable_secret

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None or pconfig.auth_type != "api_key":
        return None

    # Explicit overrides take priority.
    api_key = (explicit_api_key or "").strip()
    base_url = (explicit_base_url or "").strip()
    source = "explicit"

    if not api_key:
        for env_var in pconfig.api_key_env_vars:
            val = _getenv(env_var)
            if has_usable_secret(val):
                api_key = val
                source = f"env:{env_var}"
                break

    if not base_url and pconfig.base_url_env_var:
        base_url = _getenv(pconfig.base_url_env_var)

    if not base_url:
        base_url = pconfig.inference_base_url

    return {
        "provider": provider,
        "api_mode": "chat_completions",
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "source": source,
        "model": _get_model_config().get("default", ""),
    }


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution.

    Ported from hermes-agent/hermes_cli/runtime_provider.py line 1509
    (simplified — covers api_key providers; OAuth/pool paths fall back
    to env-var scanning).

    Returns a dict with keys: provider, api_mode, base_url, api_key, source,
    model, requested_provider.
    """
    requested_provider = resolve_requested_provider(requested)

    # MoA virtual provider.
    if requested_provider == "moa":
        return {
            "provider": "moa",
            "api_mode": "chat_completions",
            "base_url": "moa://local",
            "api_key": "moa-virtual-provider",
            "source": "moa-virtual-provider",
            "requested_provider": requested_provider,
            "model": target_model or "",
        }

    # API-key providers.
    runtime = _resolve_api_key_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if runtime is not None:
        runtime["requested_provider"] = requested_provider
        if target_model:
            runtime["model"] = target_model
        return runtime

    # Custom provider (base_url + api_key from config).
    model_cfg = _get_model_config()
    cfg_base_url = str(model_cfg.get("base_url") or "").strip()
    cfg_api_key = str(model_cfg.get("api_key") or "").strip()

    if cfg_base_url or explicit_base_url:
        base_url = (explicit_base_url or cfg_base_url).rstrip("/")
        api_key = (explicit_api_key or cfg_api_key).strip()
        if not api_key:
            # Try common env vars.
            for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NIA_API_KEY"):
                val = _getenv(env_var)
                if val:
                    api_key = val
                    break
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": "config" if cfg_base_url else "explicit",
            "model": target_model or model_cfg.get("default", ""),
            "requested_provider": requested_provider,
        }

    # Auto-detected but nothing found.
    return {
        "provider": requested_provider,
        "api_mode": "chat_completions",
        "base_url": "",
        "api_key": "",
        "source": "none",
        "model": target_model or model_cfg.get("default", ""),
        "requested_provider": requested_provider,
    }


def _has_any_provider_configured() -> bool:
    """Return True if any provider credentials are discoverable.

    Ported from hermes-agent/hermes_cli/main.py (simplified).
    """
    from niaharness.cli.auth import PROVIDER_REGISTRY, has_usable_secret

    # Check env vars for any known provider.
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            for env_var in pconfig.api_key_env_vars:
                if has_usable_secret(_getenv(env_var)):
                    return True

    # Check config.yaml for a custom provider with api_key.
    model_cfg = _get_model_config()
    if model_cfg.get("api_key") and has_usable_secret(str(model_cfg["api_key"])):
        return True

    # Check auth.json for OAuth credentials.
    try:
        from niaharness.cli.auth import _load_auth_store
        store = _load_auth_store()
        providers = store.get("providers", {})
        if isinstance(providers, dict) and providers:
            return True
    except Exception:
        pass

    return False


__all__ = [
    "resolve_runtime_provider",
    "resolve_requested_provider",
    "_has_any_provider_configured",
]
