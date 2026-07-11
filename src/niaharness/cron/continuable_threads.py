"""P1 Cron continuable threads — open a dedicated thread for continuable jobs.

Ported from Hermes Agent's ``cron/scheduler.py`` continuable-thread
functions (lines 622-832).

A "continuable" cron job is one where the user might want to reply to
the cron's output in-thread (e.g. a morning briefing that the user
might ask a follow-up about). For platforms that support threads
(Telegram, Discord, Slack), the scheduler opens a dedicated thread for
the cron's output so the user's follow-up stays in the same context.

For platforms that DON'T support threads (WhatsApp, Signal, SMS), the
scheduler falls back to the origin-DM mirror — the cron's output is
mirrored into the origin chat's session transcript.

Usage::

    from niaharness.cron.continuable_threads import open_continuable_cron_thread

    thread_id = await open_continuable_cron_thread(job, adapter, chat_id, loop)
    if thread_id is None:
        # Fall back to origin-DM mirror.
        maybe_mirror_cron_delivery(job, platform, chat_id, text)
    else:
        # Deliver to the new thread.
        await adapter.send_message(OutgoingMessage(
            platform_chat_id=chat_id, thread_id=thread_id, text=text,
        ))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def open_continuable_cron_thread(
    job: dict[str, Any],
    adapter: Any,
    chat_id: str,
    loop: Optional[asyncio.AbstractEventLoop],
) -> Optional[str]:
    """Open a dedicated thread for a continuable cron job.

    Returns the new ``thread_id`` on success, or ``None`` when:
      - The platform has no thread primitive (WhatsApp/Signal/SMS).
      - Thread creation failed.
      - No event loop is available.

    The ``None`` return is the caller's signal to fall back to the
    origin-DM mirror.

    Args:
        job: The cron job dict.
        adapter: The platform adapter instance.
        chat_id: The chat to open the thread in.
        loop: The event loop (for scheduling the adapter's async call
            from a sync context).
    """
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None

    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"NIA — {task_name}"

    try:
        coro = create_thread(str(chat_id), thread_name)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as exc:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back "
            "to DM-session mirror: %s",
            job.get("id", "?"),
            getattr(adapter, "platform_name", "?"),
            exc,
        )
        return None


def seed_cron_thread_session(
    job: dict[str, Any],
    adapter: Any,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
) -> bool:
    """Seed the freshly-opened cron thread's session with the brief.

    Without this, the brief is *visible* in the new thread but absent
    from any transcript, so the user's first reply in-thread would hit
    a session with no record of it. We create the thread-keyed session
    and append the brief as an assistant turn.

    Best-effort — a delivery that already succeeded is never failed by
    a seeding problem.

    Returns True if the seed succeeded, False otherwise.
    """
    text = (mirror_text or "").strip()
    if not text:
        return False

    try:
        from niaharness.gateway.session import SessionStore
        store = getattr(adapter, "_session_store", None) or SessionStore()
        session_key = f"{platform_name}:{chat_id}:{thread_id}"
        store.append_to_transcript(
            session_key,
            {"role": "assistant", "content": text},
        )
        logger.debug(
            "Seeded cron thread session %s for job %s",
            session_key, job.get("id"),
        )
        return True
    except Exception as exc:
        logger.debug("Seed cron thread session failed (non-fatal): %s", exc)
        return False


async def deliver_to_thread_or_mirror(
    job: dict[str, Any],
    adapter: Any,
    platform_name: str,
    chat_id: str,
    text: str,
    loop: Optional[asyncio.AbstractEventLoop],
    *,
    mirror_enabled: bool = False,
) -> bool:
    """Deliver a cron result to a continuable thread, or fall back to mirror.

    This is the main entry point for continuable cron delivery. It:
      1. Tries to open a dedicated thread via ``open_continuable_cron_thread``.
      2. If successful, delivers the text to the thread + seeds the session.
      3. If thread creation fails, falls back to mirroring the text into
         the origin chat's session (if mirror_enabled).

    Returns True if delivery succeeded (to thread or mirror), False if
    all paths failed.
    """
    # Try the thread path.
    thread_id = await open_continuable_cron_thread(job, adapter, chat_id, loop)
    if thread_id is not None:
        try:
            from niaharness.gateway import OutgoingMessage
            await adapter.send_message(OutgoingMessage(
                platform_chat_id=chat_id,
                text=text,
                metadata={"thread_id": thread_id},
            ))
            seed_cron_thread_session(
                job, adapter, platform_name, chat_id, thread_id, text,
            )
            return True
        except Exception as exc:
            logger.error("Thread delivery failed: %s", exc)
            # Fall through to mirror.

    # Fall back to mirror.
    if mirror_enabled:
        from niaharness.cron.origin import maybe_mirror_cron_delivery
        return maybe_mirror_cron_delivery(
            job, platform_name, chat_id, text, enabled=True,
        )

    return False


__all__ = [
    "deliver_to_thread_or_mirror",
    "open_continuable_cron_thread",
    "seed_cron_thread_session",
]
