"""P1 Cron origin tracking + mirror delivery.

Ported from Hermes Agent's ``cron/scheduler.py`` origin functions
(lines 465-856).

Origin tracking: when a cron job is created from a live gateway chat,
the chat is stamped as the job's ``origin`` (platform + chat_id +
thread_id). This lets the scheduler:
  1. Resolve ``deliver=origin`` to the originating chat.
  2. Mirror the cron's output into the origin chat's session transcript
     so the user's next reply sees the brief in context.

Mirror delivery: when ``attach_to_session`` is True (per-job) or
``cron.mirror_delivery`` is True (global config), the cron's final
output is appended to the target session as an assistant turn via the
gateway's mirror primitive. This makes the brief visible in the chat's
history so the user can reply to it naturally.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_origin(job: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Treats non-dict origins (free-form provenance strings, ints, lists
    from migration scripts or hand-edited jobs.json) as missing instead
    of crashing.

    Returns:
        The origin dict (with platform + chat_id) if valid, None otherwise.
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def cron_mirror_delivery_enabled(
    job: dict[str, Any],
    cfg: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether a cron delivery should also be mirrored into the target chat's
    gateway session transcript.

    Default OFF — preserves the historical isolation guarantee (cron
    deliveries live only in the cron job's own session, never the target
    chat's history).

    Precedence (first decisive value wins):
      1. Per-job ``attach_to_session`` (bool) — set via the cronjob tool.
      2. Global ``cron.mirror_delivery`` (bool) in config.yaml.
      3. False.
    """
    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            from niaharness.config.settings import load_settings
            settings = load_settings()
            cfg = getattr(settings, "cron", None) or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def target_matches_origin(
    origin: dict[str, Any],
    platform_name: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    """True when a delivery target is the job's own origin conversation.

    Mirroring is scoped to the origin session by design. A job created
    from a live gateway chat stamps that chat as ``origin``, and that
    session is guaranteed to exist. Fan-out targets (deliver=all,
    explicit platform:chat_id to some other chat) are deliberately NOT
    mirrored: they are broadcasts, not a continuation of a conversation,
    and may point at a chat the user never opened an agent session in.
    """
    if not origin:
        return False
    if str(origin.get("platform", "")).lower() != str(platform_name).lower():
        return False
    if str(origin.get("chat_id", "")) != str(chat_id):
        return False
    # thread_id must match when the origin pins one (topic-scoped chats).
    origin_thread = origin.get("thread_id")
    if origin_thread is not None and str(origin_thread) != str(thread_id or ""):
        return False
    return True


def maybe_mirror_cron_delivery(
    job: dict[str, Any],
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> bool:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (resolved once by the caller). Also no-op
    if the target is NOT the origin conversation (mirror is scoped to
    the origin session by design).

    When enabled + target == origin, the cron's final output is appended
    to the target session as an assistant turn via the gateway's mirror
    primitive, so the next user reply in that chat sees the brief in
    context.

    Returns True if the mirror was attempted (success or failure), False
    if it was skipped (not enabled or target != origin).
    """
    if not enabled:
        return False

    origin = resolve_origin(job)
    if not origin:
        return False

    if not target_matches_origin(origin, platform_name, chat_id, thread_id):
        return False

    text = (mirror_text or "").strip()
    if not text:
        return False

    # Best-effort: append to the session transcript via the gateway's
    # session store. Failures are logged + swallowed — a delivery that
    # already succeeded is never failed by a seeding problem.
    try:
        from niaharness.gateway.session import SessionStore
        store = SessionStore()
        session_key = f"{platform_name}:{chat_id}"
        if thread_id:
            session_key += f":{thread_id}"
        store.append_to_transcript(
            session_key,
            {"role": "assistant", "content": text},
        )
        logger.debug(
            "Mirrored cron delivery into origin session %s (job=%s)",
            session_key, job.get("id"),
        )
        return True
    except Exception as exc:
        logger.debug("Mirror delivery failed (non-fatal): %s", exc)
        return False


def cron_job_origin_log_suffix(job: dict[str, Any]) -> str:
    """Build a log-line suffix describing the job's origin.

    e.g. ``" (origin: telegram:123456)"`` or ``""`` if no origin.
    """
    origin = resolve_origin(job)
    if not origin:
        return ""
    platform = origin.get("platform", "")
    chat_id = origin.get("chat_id", "")
    thread_id = origin.get("thread_id", "")
    suffix = f" (origin: {platform}:{chat_id}"
    if thread_id:
        suffix += f":{thread_id}"
    suffix += ")"
    return suffix


__all__ = [
    "cron_job_origin_log_suffix",
    "cron_mirror_delivery_enabled",
    "maybe_mirror_cron_delivery",
    "resolve_origin",
    "target_matches_origin",
]
