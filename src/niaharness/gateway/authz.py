"""P1 Gateway per-platform authorization.

Ported from Hermes Agent's ``gateway/authz_mixin.py`` (692 LOC), scoped to
NIA's architecture. Provides:

  - :class:`AuthorizationPolicy` — resolved per-platform allowlist policy.
  - :func:`policy_for_source` — resolve the policy for an incoming message.
  - :func:`is_user_authorized` — check if a user/chat is allowed to talk
    to the agent on a given platform.

Two access modes per platform:
  - ``allow_from`` — user IDs that may talk to the agent (DM scope).
  - ``group_allow_from`` — user IDs that may talk in groups.

When ``allow_from`` is empty/unset, the platform is "open" (anyone can
talk). When set, only listed users + the gateway owner may talk.

Configuration (config.yaml ``gateway.platforms.<platform>`` section):
  - ``allow_from``: list of user IDs (DM scope).
  - ``group_allow_from``: list of user IDs (group scope).
  - ``owner_id``: the gateway owner's user ID (always authorized).
  - ``dm_policy``: "open" (default) or "allowlist".
  - ``group_policy``: "open" (default) or "allowlist".

Usage::

    from niaharness.gateway.authz import policy_for_source, is_user_authorized

    if not is_user_authorized(source, config):
        return "Unauthorized"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorizationPolicy:
    """Resolved authorization policy for a single (platform, scope) pair.

    Attributes:
        enabled: True if the allowlist is active for this scope.
        owner_id: The gateway owner's user ID (always authorized).
        allowed_user_ids: Frozenset of user IDs on the allowlist.
        adapter_enforces_own_policy: True if the adapter gates at intake
            (e.g. WeCom, Weixin). When True, the gateway trusts the
            adapter's decision and skips its own check.
        authorization_is_upstream: True if the adapter delegates authz to
            a trusted upstream (e.g. relay). When True, the gateway
            honors the upstream decision directly.
    """

    enabled: bool
    owner_id: str = ""
    allowed_user_ids: FrozenSet[str] = field(default_factory=frozenset)
    adapter_enforces_own_policy: bool = False
    authorization_is_upstream: bool = False

    def is_authorized(self, user_id: Optional[str]) -> bool:
        """Return True if the user is authorized under this policy."""
        if not self.enabled:
            return True  # Open platform — anyone can talk.
        if not user_id:
            return False
        uid = str(user_id)
        # Owner is always authorized.
        if self.owner_id and uid == self.owner_id:
            return True
        # Allowlist.
        return uid in self.allowed_user_ids


# ---------------------------------------------------------------------------
# Scope classification
# ---------------------------------------------------------------------------


_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})


def _scope_for_chat_type(chat_type: Optional[str]) -> str:
    """Return 'dm' or 'group' based on chat_type."""
    if chat_type and chat_type.lower() in _DM_CHAT_TYPES:
        return "dm"
    return "group"


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def _coerce_id_list(raw: Any) -> FrozenSet[str]:
    """Normalize a YAML-loaded ID list into a frozenset of strings."""
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return frozenset(out)


def _platform_config(gateway_config: Any, platform: str) -> dict[str, Any]:
    """Extract the platform config dict from the gateway config."""
    if gateway_config is None:
        return {}
    platforms = getattr(gateway_config, "platforms", None)
    if platforms is None:
        return {}
    try:
        cfg = platforms.get(platform, {})
    except (AttributeError, TypeError):
        return {}
    if isinstance(cfg, dict):
        return cfg
    # Object with attributes.
    return {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if not k.startswith("_") and not callable(getattr(cfg, k))
    }


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def policy_for_source(
    gateway_config: Any,
    source: Any,
) -> AuthorizationPolicy:
    """Resolve the authorization policy for an incoming message source.

    Args:
        gateway_config: The gateway config object (with .platforms dict).
        source: The message source (with .platform, .chat_type, .user_id).

    Returns:
        An AuthorizationPolicy. Returns a "disabled" (open) policy when:
          - gateway_config is None
          - the platform has no config
          - the platform's allow_from is empty (open platform)
    """
    if gateway_config is None or source is None:
        return AuthorizationPolicy(enabled=False)

    platform = getattr(source, "platform", None) or getattr(source, "platform_name", None)
    if not platform:
        return AuthorizationPolicy(enabled=False)

    cfg = _platform_config(gateway_config, str(platform))
    if not cfg:
        return AuthorizationPolicy(enabled=False)

    scope = _scope_for_chat_type(getattr(source, "chat_type", None))

    # Pick the right allowlist key for the scope.
    if scope == "group":
        allow_key = "group_allow_from"
        policy_key = "group_policy"
    else:
        allow_key = "allow_from"
        policy_key = "dm_policy"

    allowed = _coerce_id_list(cfg.get(allow_key))
    owner_id = str(cfg.get("owner_id", "")).strip()
    policy_mode = str(cfg.get(policy_key, "open")).strip().lower()

    # The allowlist is "enabled" when:
    #   - policy_mode is "allowlist", OR
    #   - allow_from is non-empty (implicitly switches to allowlist mode).
    enabled = policy_mode == "allowlist" or bool(allowed)

    # Adapter-level flags.
    adapter_enforces = bool(cfg.get("enforces_own_access_policy", False))
    upstream = bool(cfg.get("authorization_is_upstream", False))

    return AuthorizationPolicy(
        enabled=enabled,
        owner_id=owner_id,
        allowed_user_ids=allowed,
        adapter_enforces_own_policy=adapter_enforces,
        authorization_is_upstream=upstream,
    )


def is_user_authorized(
    source: Any,
    gateway_config: Any,
) -> bool:
    """Check if the user who sent *source* is authorized to talk to the agent.

    This is the main entry point. It resolves the policy and checks the
    user_id against it. When the adapter enforces its own policy or
    delegates to an upstream, the gateway trusts that decision.
    """
    policy = policy_for_source(gateway_config, source)

    # If the adapter enforces its own access policy, trust it.
    if policy.adapter_enforces_own_policy:
        return True
    # If authz is upstream (e.g. relay), trust it.
    if policy.authorization_is_upstream:
        return True

    user_id = getattr(source, "user_id", None) or getattr(source, "platform_user_id", None)
    return policy.is_authorized(user_id)


__all__ = [
    "AuthorizationPolicy",
    "is_user_authorized",
    "policy_for_source",
]
