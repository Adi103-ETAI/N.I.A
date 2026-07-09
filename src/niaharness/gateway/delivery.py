"""Gateway delivery — cron→channel routing + dead target registry.

Ported from Hermes Agent's ``gateway/delivery.py`` (557 LOC) +
``gateway/dead_targets.py`` (144 LOC), scoped to NIA's architecture.
Provides:

  - :class:`DeliveryTarget` — a single delivery destination (platform +
    chat_id + thread_id). Parses ``"origin"``, ``"local"``,
    ``"telegram"``, ``"telegram:123456"`` format strings.
  - :class:`DeliveryRouter` — routes messages to destinations, dispatching
    via per-platform adapters. Handles oversized output (truncation +
    audit save), silence-narration filtering, and dead-target skipping.
  - :class:`DeadTargetRegistry` — persistent set of confirmed-dead
    targets (deleted group, bot kicked/blocked, deactivated user).
    Self-healing: a successful send clears the flag.

The router is platform-agnostic — it routes by platform name and
dispatches via ``adapters[platform]``. NIA currently ships Telegram;
Discord/Slack/Signal/Matrix adapters can be plugged in without
modifying the router.

Usage::

    from niaharness.gateway.delivery import DeliveryRouter, DeliveryTarget

    router = DeliveryRouter(adapters={"telegram": tg_adapter})
    targets = [DeliveryTarget.parse("origin", origin=source), DeliveryTarget.parse("local")]
    result = await router.deliver("Cron output here", targets, job_id="morning-summary")
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from niaharness.gateway.session import SessionSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum chars for a single platform output (Telegram's hard limit is 4096;
# 4000 leaves room for the truncation footer).
MAX_PLATFORM_OUTPUT = 4000

# Error kinds that indicate a permanently-dead target.
_DEAD_ERROR_KINDS = frozenset({"forbidden", "not_found"})

# Silence-narration regex (anchored, ≤64 chars).
_SILENCE_NARRATION_RE = re.compile(
    r"^[\s*_~`]*\(?\s*(silent|silence|no\s+response|no\s+reply)\s*\.?\)?[\s*_~`]*$"
    r"|^[\s*_~`]*[\U0001F507\.\u2026]+[\s*_~`]*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_silence_narration(content: Optional[str]) -> bool:
    """True if content is only a silence token (e.g. *(silent)*, 🔇, bare '.')."""
    if not content or len(content) > 64:
        return False
    return bool(_SILENCE_NARRATION_RE.match(content.strip()))


def _normalize(platform: str, chat_id: str) -> str:
    """Canonical key for a (platform, chat_id) pair."""
    return f"{str(platform).strip().lower()}:{str(chat_id).strip()}"


def _classify_dead_from_error_text(error_text: Optional[str]) -> Optional[str]:
    """Best-effort recovery of error_kind from an exception string.

    Returns one of :data:`_DEAD_ERROR_KINDS` or None.
    """
    if not error_text:
        return None
    text = error_text.lower()
    # Chat-level not_found (deleted group, etc.).
    if "chat not found" in text or "chat_id not found" in text:
        return "not_found"
    # Forbidden (bot kicked/blocked).
    if "forbidden" in text or "bot was blocked" in text or "bot kicked" in text:
        return "forbidden"
    return None


# ---------------------------------------------------------------------------
# DeliveryTarget
# ---------------------------------------------------------------------------


@dataclass
class DeliveryTarget:
    """A single delivery destination.

    Attributes:
        platform: Platform name (e.g. "telegram", "local").
        chat_id: Chat ID (None = use home channel).
        thread_id: Thread/topic ID (optional).
        is_origin: True if this target is the originating chat.
        is_explicit: True if chat_id was explicitly specified.
    """

    platform: str
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    is_origin: bool = False
    is_explicit: bool = False

    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource] = None) -> "DeliveryTarget":
        """Parse a delivery target string.

        Formats:
          - ``"origin"`` → back to source
          - ``"local"`` → local files only
          - ``"telegram"`` → Telegram home channel
          - ``"telegram:123456"`` → specific Telegram chat
          - ``"telegram:123456:789"`` → specific chat + thread
        """
        target_stripped = target.strip()
        target_lower = target_stripped.lower()

        if target_lower == "origin":
            if origin:
                return cls(
                    platform=origin.platform,
                    chat_id=origin.chat_id,
                    thread_id=origin.thread_id,
                    is_origin=True,
                )
            return cls(platform="local", is_origin=True)

        if target_lower == "local":
            return cls(platform="local")

        # platform:chat_id[:thread_id] format.
        if ":" in target_stripped:
            parts = target_stripped.split(":", 2)
            platform_str = parts[0].lower()
            chat_id = parts[1] if len(parts) > 1 else None
            thread_id = parts[2] if len(parts) > 2 else None
            return cls(platform=platform_str, chat_id=chat_id, thread_id=thread_id, is_explicit=True)

        # Bare platform name → use home channel.
        return cls(platform=target_lower)

    def to_string(self) -> str:
        """Convert back to string format."""
        if self.is_origin:
            return "origin"
        if self.platform == "local":
            return "local"
        if self.chat_id and self.thread_id:
            return f"{self.platform}:{self.chat_id}:{self.thread_id}"
        if self.chat_id:
            return f"{self.platform}:{self.chat_id}"
        return self.platform


# ---------------------------------------------------------------------------
# DeadTargetRegistry
# ---------------------------------------------------------------------------


class DeadTargetRegistry:
    """Thread-safe, persistent set of confirmed-dead delivery targets.

    Keyed on ``platform:chat_id``. Self-healing: :meth:`clear` (called on
    a successful send) removes the flag. No TTL — the flag stays set
    indefinitely until a successful send clears it.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._dead: Dict[str, Dict[str, Any]] = {}
        if path is not None:
            self._path = path
        else:
            try:
                from niaharness.prompts.soul import get_nia_home

                self._path = get_nia_home() / "gateway" / "dead_targets.json"
            except Exception:
                self._path = Path(os.path.expanduser("~/.nia/gateway/dead_targets.json"))
        self._load()

    def _load(self) -> None:
        """Load from disk (best-effort)."""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._dead = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError) as exc:
            logger.debug("dead_targets: could not load %s (%s)", self._path, exc)
            self._dead = {}

    def _flush_locked(self) -> None:
        """Persist to disk (best-effort)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._dead, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            logger.debug("dead_targets: could not persist %s (%s)", self._path, exc)

    @staticmethod
    def is_dead_error_kind(error_kind: Optional[str]) -> bool:
        """Return True when error_kind denotes a permanent whole-chat death."""
        return bool(error_kind) and error_kind in _DEAD_ERROR_KINDS

    def is_dead(self, platform: str, chat_id: Optional[str]) -> bool:
        """Check if a target is confirmed-dead."""
        if not chat_id:
            return False
        with self._lock:
            return _normalize(platform, chat_id) in self._dead

    def mark_dead(self, platform: str, chat_id: Optional[str], reason: str = "") -> bool:
        """Record a target as confirmed-dead. Returns True if newly added."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            existed = key in self._dead
            self._dead[key] = {
                "platform": str(platform).strip().lower(),
                "chat_id": str(chat_id),
                "reason": str(reason)[:200],
                "marked_at": time.time(),
            }
            self._flush_locked()
        if not existed:
            logger.info(
                "dead_targets: marked %s as unreachable (%s) — future deliveries skipped",
                key, reason or "no reason given",
            )
        return not existed

    def clear(self, platform: str, chat_id: Optional[str]) -> bool:
        """Remove a target's dead flag (self-healing). Returns True if it was set."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            if key in self._dead:
                del self._dead[key]
                self._flush_locked()
                logger.info("dead_targets: cleared %s (delivery succeeded)", key)
                return True
        return False

    def all_dead(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot of the current dead set (for diagnostics)."""
        with self._lock:
            return {k: dict(v) for k, v in self._dead.items()}


# ---------------------------------------------------------------------------
# DeliveryRouter
# ---------------------------------------------------------------------------


class DeliveryRouter:
    """Routes messages to appropriate destinations.

    Handles:
      - Resolving delivery targets (origin, local, platform, platform:chat_id).
      - Dispatching to per-platform adapters.
      - Oversized output handling (truncation + audit save).
      - Silence-narration filtering (anti-loop guard for bot-to-bot channels).
      - Dead-target skipping (confirmed-unreachable targets are short-circuited).
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, Any]] = None,
        *,
        dead_targets: Optional[DeadTargetRegistry] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.adapters = adapters or {}
        self.dead_targets = dead_targets or DeadTargetRegistry()
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            try:
                from niaharness.prompts.soul import get_nia_home

                self.output_dir = get_nia_home() / "cron" / "output"
            except Exception:
                self.output_dir = Path(os.path.expanduser("~/.nia/cron/output"))

    async def deliver(
        self,
        content: str,
        targets: List[DeliveryTarget],
        *,
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Deliver content to all specified targets.

        Args:
            content: The message/output to deliver.
            targets: List of delivery targets.
            job_id: Optional cron job ID (for local file naming).
            job_name: Optional human-readable job name.
            metadata: Additional metadata.

        Returns:
            Dict mapping target string → {success, result/error}.
        """
        results: Dict[str, Any] = {}

        for target in targets:
            # Skip known-dead targets.
            if (
                target.platform != "local"
                and target.chat_id
                and self.dead_targets.is_dead(target.platform, target.chat_id)
            ):
                logger.info(
                    "Skipping delivery to known-dead target %s:%s",
                    target.platform, target.chat_id,
                )
                results[target.to_string()] = {
                    "success": False,
                    "skipped": "dead_target",
                    "error": "target previously confirmed unreachable",
                }
                continue

            try:
                if target.platform == "local":
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)
                    # Clear dead flag on success.
                    if target.chat_id and result.get("success", True):
                        self.dead_targets.clear(target.platform, target.chat_id)

                results[target.to_string()] = {"success": True, "result": result}
            except Exception as exc:
                # Check if this is a dead-target error.
                if target.platform != "local" and target.chat_id:
                    dead_kind = _classify_dead_from_error_text(str(exc))
                    if dead_kind:
                        self.dead_targets.mark_dead(
                            target.platform, target.chat_id,
                            reason=f"{dead_kind}: {str(exc)[:120]}",
                        )
                results[target.to_string()] = {"success": False, "error": str(exc)}

        return results

    def _deliver_local(
        self,
        content: str,
        job_id: Optional[str],
        job_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Save content to a local Markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if job_id:
            output_path = self.output_dir / job_id / f"{timestamp}.md"
        else:
            output_path = self.output_dir / "misc" / f"{timestamp}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(f"# {job_name}" if job_name else "# Delivery Output")
        lines.append("")
        lines.append(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if job_id:
            lines.append(f"**Job ID:** {job_id}")
        if metadata:
            for key, value in metadata.items():
                lines.append(f"**{key}:** {value}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(content)

        output_path.write_text("\n".join(lines), encoding="utf-8")

        return {"path": str(output_path), "timestamp": timestamp}

    def _save_full_output(self, content: str, job_id: str) -> Path:
        """Save full cron output to disk (audit trail for oversized content)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{job_id}_{timestamp}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _filter_silence_narration_enabled(self) -> bool:
        """Whether the silence-narration filter is active (default True)."""
        env = os.getenv("NIA_FILTER_SILENCE_NARRATION")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return True  # Default: filter on.

    async def _deliver_to_platform(
        self,
        target: DeliveryTarget,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deliver content to a messaging platform via its adapter."""
        adapter = self.adapters.get(target.platform)
        if not adapter:
            raise ValueError(f"No adapter configured for {target.platform}")
        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform} delivery")

        job_id = (metadata or {}).get("job_id", "unknown")
        saved_path: Optional[Path] = None

        # Oversized output handling.
        if len(content) > MAX_PLATFORM_OUTPUT:
            try:
                saved_path = self._save_full_output(content, job_id)
            except OSError as exc:
                logger.warning("Audit save failed for cron output: %s", exc)

            # Truncate for non-chunking adapters.
            if not getattr(adapter, "splits_long_messages", False):
                if saved_path is None:
                    saved_path = self._save_full_output(content, job_id)
                footer = f"\n\n... [truncated, full output saved to {saved_path}]"
                visible = max(0, MAX_PLATFORM_OUTPUT - len(footer))
                content = content[:visible] + footer

        # Silence-narration filter (anti-loop guard).
        if self._filter_silence_narration_enabled() and _is_silence_narration(content):
            logger.warning(
                "Dropped silence-narration outbound to %s:%s: %r",
                target.platform, target.chat_id, content[:40],
            )
            return {"success": True, "filtered": "silence_narration", "delivered": False}

        # Dispatch to adapter.
        send_metadata = dict(metadata or {})
        if target.thread_id and "thread_id" not in send_metadata:
            send_metadata["thread_id"] = target.thread_id

        result = await adapter.send(target.chat_id, content, metadata=send_metadata or None)

        # Check for failure.
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(result.get("error", f"{target.platform} delivery failed"))

        return result if isinstance(result, dict) else {"success": True, "result": str(result)}


__all__ = [
    "DeadTargetRegistry",
    "DeliveryRouter",
    "DeliveryTarget",
    "MAX_PLATFORM_OUTPUT",
    "_is_silence_narration",
]
