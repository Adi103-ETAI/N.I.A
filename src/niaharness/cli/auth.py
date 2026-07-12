"""Provider authentication registry + credential management.

Ported from hermes-agent/hermes_cli/auth.py (8,127 LOC) — focused extraction
of the essential APIs that the tui_gateway deep-ports depend on:
  - PROVIDER_REGISTRY (ProviderConfig dataclass + all known providers)
  - has_usable_secret (placeholder detection)
  - clear_provider_auth (auth.json credential clearing)
  - _load_auth_store / _save_auth_store (auth.json I/O)

The full Hermes auth.py has OAuth device-code flows, token refresh, credential
pools, Z.AI endpoint detection, and more. This port covers the API surface
needed by model.disconnect + setup.runtime_check.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AUTH_STORE_VERSION = 1

# ---------------------------------------------------------------------------
# ProviderConfig + PROVIDER_REGISTRY
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Describes a known inference provider.

    Ported from hermes-agent/hermes_cli/auth.py line 160.
    """

    id: str
    name: str
    auth_type: str  # "oauth_device_code", "oauth_external", "api_key", "external_process"
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    api_key_env_vars: tuple = ()
    base_url_env_var: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "openai-api": ProviderConfig(
        id="openai-api",
        name="OpenAI API",
        auth_type="api_key",
        inference_base_url="https://api.openai.com/v1",
        api_key_env_vars=("OPENAI_API_KEY",),
        base_url_env_var="OPENAI_BASE_URL",
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        auth_type="api_key",
        inference_base_url="https://api.anthropic.com",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        auth_type="api_key",
        inference_base_url="https://api.deepseek.com/v1",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
    ),
    "xai": ProviderConfig(
        id="xai",
        name="xAI (Grok)",
        auth_type="api_key",
        inference_base_url="https://api.x.ai/v1",
        api_key_env_vars=("XAI_API_KEY",),
    ),
    "gemini": ProviderConfig(
        id="gemini",
        name="Google AI Studio",
        auth_type="api_key",
        inference_base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        base_url_env_var="GEMINI_BASE_URL",
    ),
    "zai": ProviderConfig(
        id="zai",
        name="Z.AI / GLM",
        auth_type="api_key",
        inference_base_url="https://api.z.ai/api/paas/v4",
        api_key_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        base_url_env_var="GLM_BASE_URL",
    ),
    "kimi-coding": ProviderConfig(
        id="kimi-coding",
        name="Kimi / Moonshot",
        auth_type="api_key",
        inference_base_url="https://api.moonshot.ai/v1",
        api_key_env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        base_url_env_var="KIMI_BASE_URL",
    ),
    "lmstudio": ProviderConfig(
        id="lmstudio",
        name="LM Studio",
        auth_type="api_key",
        inference_base_url="http://127.0.0.1:1234/v1",
        api_key_env_vars=("LM_API_KEY",),
        base_url_env_var="LM_BASE_URL",
    ),
    "copilot": ProviderConfig(
        id="copilot",
        name="GitHub Copilot",
        auth_type="api_key",
        inference_base_url="https://models.inference.ai.azure.com",
        api_key_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
        base_url_env_var="COPILOT_API_BASE_URL",
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq Cloud",
        auth_type="api_key",
        inference_base_url="https://api.groq.com/openai/v1",
        api_key_env_vars=("GROQ_API_KEY",),
    ),
    "fireworks": ProviderConfig(
        id="fireworks",
        name="Fireworks AI",
        auth_type="api_key",
        inference_base_url="https://api.fireworks.ai/inference/v1",
        api_key_env_vars=("FIREWORKS_API_KEY", "FW_API_KEY"),
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        auth_type="api_key",
        inference_base_url="https://openrouter.ai/api/v1",
        api_key_env_vars=("OPENROUTER_API_KEY",),
    ),
    "together": ProviderConfig(
        id="together",
        name="Together AI",
        auth_type="api_key",
        inference_base_url="https://api.together.xyz/v1",
        api_key_env_vars=("TOGETHER_API_KEY",),
    ),
    "arcee": ProviderConfig(
        id="arcee",
        name="Arcee AI",
        auth_type="api_key",
        inference_base_url="https://api.arcee.ai/v1",
        api_key_env_vars=("ARCEE_API_KEY",),
    ),
    "novita": ProviderConfig(
        id="novita",
        name="Novita AI",
        auth_type="api_key",
        inference_base_url="https://api.novita.ai/v3/openai",
        api_key_env_vars=("NOVITA_API_KEY",),
    ),
    "minimax": ProviderConfig(
        id="minimax",
        name="MiniMax",
        auth_type="api_key",
        inference_base_url="https://api.minimax.chat/v1",
        api_key_env_vars=("MINIMAX_API_KEY",),
    ),
    "alibaba": ProviderConfig(
        id="alibaba",
        name="Alibaba (DashScope)",
        auth_type="api_key",
        inference_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env_vars=("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    ),
    "stepfun": ProviderConfig(
        id="stepfun",
        name="StepFun",
        auth_type="api_key",
        inference_base_url="https://api.stepfun.com/v1",
        api_key_env_vars=("STEPFUN_API_KEY",),
    ),
    "gmi": ProviderConfig(
        id="gmi",
        name="GMI",
        auth_type="api_key",
        inference_base_url="https://api.gmi-serving.com/v1",
        api_key_env_vars=("GMI_API_KEY",),
    ),
    "perplexity": ProviderConfig(
        id="perplexity",
        name="Perplexity",
        auth_type="api_key",
        inference_base_url="https://api.perplexity.ai",
        api_key_env_vars=("PPLX_API_KEY", "PERPLEXITY_API_KEY"),
    ),
    "mistral": ProviderConfig(
        id="mistral",
        name="Mistral AI",
        auth_type="api_key",
        inference_base_url="https://api.mistral.ai/v1",
        api_key_env_vars=("MISTRAL_API_KEY",),
    ),
    "cohere": ProviderConfig(
        id="cohere",
        name="Cohere",
        auth_type="api_key",
        inference_base_url="https://api.cohere.com/v1",
        api_key_env_vars=("COHERE_API_KEY",),
    ),
    "hyperbolic": ProviderConfig(
        id="hyperbolic",
        name="Hyperbolic",
        auth_type="api_key",
        inference_base_url="https://api.hyperbolic.xyz/v1",
        api_key_env_vars=("HYPERBOLIC_API_KEY",),
    ),
}


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_SECRET_VALUES = {
    "*", "**", "***", "changeme", "your_api_key", "your_api_key_here",
    "your-api-key", "placeholder", "example", "dummy", "null", "none",
}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder.

    Ported from hermes-agent/hermes_cli/auth.py line 556.
    """
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return False
    return True


# ---------------------------------------------------------------------------
# Auth store (auth.json) I/O
# ---------------------------------------------------------------------------


def _nia_home() -> Path:
    try:
        from niaharness.prompts.soul import get_nia_home
        return get_nia_home()
    except Exception:
        return Path(os.path.expanduser("~/.nia"))


def _auth_file_path() -> Path:
    """Return the auth.json path under NIA_HOME."""
    return _nia_home() / "auth.json"


@contextlib.contextmanager
def _auth_store_lock():
    """Context manager for auth.json access (thread lock, not file lock).

    The full Hermes implementation uses fcntl file locks; this port uses a
    threading lock for simplicity. For single-process TUI gateway use this
    is sufficient.
    """
    import threading
    lock = getattr(_auth_store_lock, "_lock", None)
    if lock is None:
        lock = threading.RLock()
        _auth_store_lock._lock = lock  # type: ignore[attr-defined]
    with lock:
        yield


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load the auth store from auth.json.

    Ported from hermes-agent/hermes_cli/auth.py line 1066.
    """
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    try:
        raw = json.loads(auth_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("auth: failed to parse %s (%s) — starting with empty store", auth_file, exc)
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict)
        or isinstance(raw.get("credential_pool"), dict)
    ):
        raw.setdefault("providers", {})
        raw.setdefault("credential_pool", {})
        return raw

    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _save_auth_store(auth_store: Dict[str, Any], target_path: Optional[Path] = None) -> Path:
    """Save the auth store to auth.json atomically.

    Ported from hermes-agent/hermes_cli/auth.py line 1108.
    """
    auth_file = target_path if target_path is not None else _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(auth_store, indent=2) + "\n"
    tmp_path = auth_file.with_name(f"{auth_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        # Write to temp file then rename for atomicity.
        tmp_path.write_text(payload, encoding="utf-8")
        # Set restrictive permissions on POSIX.
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        tmp_path.replace(auth_file)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return auth_file


# ---------------------------------------------------------------------------
# Provider auth clearing
# ---------------------------------------------------------------------------


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """Clear auth state for a provider.

    Ported from hermes-agent/hermes_cli/auth.py line 1501.

    If provider_id is None, clears the active provider.
    Returns True if something was cleared.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False

        providers = auth_store.get("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            auth_store["providers"] = providers

        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool

        cleared = False
        if target in providers:
            del providers[target]
            cleared = True
        if target in pool:
            del pool[target]
            cleared = True

        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True

        if not cleared:
            return False
        _save_auth_store(auth_store)
    return True


def deactivate_provider() -> None:
    """Clear active_provider in auth.json without deleting credentials.

    Ported from hermes-agent/hermes_cli/auth.py line 1541.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


__all__ = [
    "ProviderConfig",
    "PROVIDER_REGISTRY",
    "has_usable_secret",
    "clear_provider_auth",
    "deactivate_provider",
    "_load_auth_store",
    "_save_auth_store",
    "_auth_file_path",
    "AUTH_STORE_VERSION",
]
