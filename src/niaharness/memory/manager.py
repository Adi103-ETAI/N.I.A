"""Memory manager — orchestrates providers, streaming scrubber, context block builder.

Ported from Hermes Agent's ``agent/memory_manager.py`` (1,086 LOC), scoped
to NIA's architecture. Provides:

  - :class:`MemoryManager` — orchestrates the built-in provider plus at
    most one external provider. Handles prefetch fan-out, background sync
    via daemon ThreadPoolExecutor, tool-call routing, lifecycle hooks.
  - :class:`StreamingContextScrubber` — stateful scrubber that strips
    ``<memory-context>`` blocks from streaming model output across chunk
    boundaries. Prevents the agent from echoing its own memory context
    back to the user.
  - :func:`build_memory_context_block` — wraps prefetched memory in a
    fenced ``<memory-context>`` block with a ``[System note: ...]``
    preamble so the model treats it as reference data, not new input.
  - :func:`sanitize_context` — one-shot strip of fence tags + system-note
    lines from provider output.

The built-in provider (:class:`BuiltinJsonMemoryProvider`) wraps NIA's
existing file-based Memory class. External providers (vector DB, Honcho)
implement the :class:`MemoryProvider` protocol and drop in without
touching the agent loop.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from niaharness.memory.provider import MemoryProvider
from niaharness.memory.threat_patterns import first_threat_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYNC_DRAIN_TIMEOUT_S = 5.0

# Regexes for one-shot context sanitization.
_FENCE_TAG_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>", re.IGNORECASE
)
_INTERNAL_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*"
    r"NOT new user input\.\s*Treat as "
    r"(?:informational background data|authoritative reference data[^\]]*)\.\]\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Sanitize + context block builder
# ---------------------------------------------------------------------------


def sanitize_context(text: str) -> str:
    """One-shot strip of fence tags, full context blocks, and system-note lines.

    Does NOT survive chunk boundaries — use :class:`StreamingContextScrubber`
    for streaming text.
    """
    if not text:
        return text
    text = _INTERNAL_CONTEXT_RE.sub("", text)
    text = _FENCE_TAG_RE.sub("", text)
    text = _INTERNAL_NOTE_RE.sub("", text)
    return text.strip()


def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in a fenced block with system note.

    Returns ``""`` if *raw_context* is empty. The preamble tells the model:
    this is RECALLED memory, NOT new user input, treat as AUTHORITATIVE
    REFERENCE DATA.

    The :class:`StreamingContextScrubber` strips this block from streaming
    model output so the agent doesn't echo its own context back to the user.
    """
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )


# ---------------------------------------------------------------------------
# StreamingContextScrubber
# ---------------------------------------------------------------------------


class StreamingContextScrubber:
    """Stateful scrubber for streaming text with split memory-context spans.

    The one-shot :func:`sanitize_context` regex cannot survive chunk
    boundaries: a ``<memory-context>`` opened in one delta and closed in
    a later delta leaks its payload to the UI. This scrubber runs a small
    state machine across deltas, holding back partial-tag tails and
    discarding everything inside a span.

    Usage::

        scrubber = StreamingContextScrubber()
        for delta in stream:
            visible = scrubber.feed(delta)
            if visible:
                emit(visible)
        trailing = scrubber.flush()
        if trailing:
            emit(trailing)
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        """Clear state (call at new turn)."""
        self._in_span = False
        self._buf = ""
        self._at_block_boundary = True

    def feed(self, text: str) -> str:
        """Return the visible portion of *text* after scrubbing."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                buf = buf[idx + len(self._CLOSE_TAG):]
                self._in_span = False
            else:
                idx = self._find_boundary_open_tag(buf)
                if idx == -1:
                    held = (
                        self._max_pending_open_suffix(buf)
                        or self._max_partial_suffix(buf, self._OPEN_TAG)
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                if idx > 0:
                    self._append_visible(out, buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG):]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream."""
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    # -- Private helpers --------------------------------------------------

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix."""
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span."""
        buf_lower = buf.lower()
        search_start = 0
        while True:
            idx = buf_lower.find(self._OPEN_TAG, search_start)
            if idx == -1:
                return -1
            if self._is_block_boundary(buf, idx) and self._has_block_opener_suffix(buf, idx):
                return idx
            search_start = idx + 1

    def _max_pending_open_suffix(self, buf: str) -> int:
        if not buf.lower().endswith(self._OPEN_TAG):
            return 0
        idx = len(buf) - len(self._OPEN_TAG)
        if not self._is_block_boundary(buf, idx):
            return 0
        return len(self._OPEN_TAG)

    def _has_block_opener_suffix(self, buf: str, idx: int) -> bool:
        after_idx = idx + len(self._OPEN_TAG)
        if after_idx >= len(buf):
            return False
        return buf[after_idx] in "\r\n"

    def _is_block_boundary(self, buf: str, idx: int) -> bool:
        if idx == 0:
            return self._at_block_boundary
        preceding = buf[:idx]
        last_newline = preceding.rfind("\n")
        if last_newline == -1:
            return self._at_block_boundary and preceding.strip() == ""
        return preceding[last_newline + 1:].strip() == ""

    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            self._at_block_boundary = text[last_newline + 1:].strip() == ""
        else:
            self._at_block_boundary = self._at_block_boundary and text.strip() == ""


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider.

    The builtin provider is always first. Only one non-builtin (external)
    provider is allowed. Failures in one provider never block the other.

    Usage::

        manager = MemoryManager()
        manager.add_provider(BuiltinJsonMemoryProvider(...))
        # Optionally: manager.add_provider(ExternalProvider(...))

        # Before each turn:
        context = manager.prefetch_all(user_message)
        prompt_with_context = build_memory_context_block(context) + "\\n\\n" + user_message

        # After each turn (background, non-blocking):
        manager.sync_all(user_content, assistant_content, session_id=session_id)
    """

    _MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}

    def __init__(self) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False
        self._sync_executor: Optional[ThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()

    # -- Registration --------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a memory provider.

        Built-in provider (name ``"builtin"``) is always accepted. Only
        **one** external provider is allowed — a second attempt is rejected.
        """
        is_builtin = provider.name == "builtin"

        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if p.name != "builtin"), "unknown"
                )
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider allowed.",
                    provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # Index tool names → provider for routing.
        for raw_schema in provider.get_tool_schemas():
            schema = _normalize_tool_schema(raw_schema)
            if schema is None:
                continue
            tool_name = schema["name"]
            if tool_name and tool_name not in self._tool_to_provider:
                self._tool_to_provider[tool_name] = provider
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s",
                    tool_name, self._tool_to_provider[tool_name].name,
                )

        logger.info(
            "Memory provider '%s' registered (%d tools)",
            provider.name, len(provider.get_tool_schemas()),
        )

    @property
    def providers(self) -> List[MemoryProvider]:
        """Returns a copy of all providers in order."""
        return list(self._providers)

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """Lookup by name."""
        return next((p for p in self._providers if p.name == name), None)

    # -- System prompt -------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Concatenate non-empty system_prompt_block() from every provider."""
        parts = [p.system_prompt_block() for p in self._providers if p.system_prompt_block()]
        return "\n\n".join(parts)

    # -- Prefetch ------------------------------------------------------------

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers (inline, synchronous)."""
        if not query:
            return ""
        parts: List[str] = []
        for provider in self._providers:
            try:
                result = provider.prefetch(query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug("Memory provider '%s' prefetch failed: %s", provider.name, e)
        return "\n\n".join(parts)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Background prefetch for next turn."""
        providers = list(self._providers)
        if not providers:
            return

        def _run() -> None:
            for provider in providers:
                try:
                    provider.queue_prefetch(query, session_id=session_id)
                except Exception as e:
                    logger.debug("Memory provider '%s' queue_prefetch failed: %s", provider.name, e)

        self._submit_background(_run)

    # -- Sync (background) ---------------------------------------------------

    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Sync a completed turn to all providers (background, non-blocking).

        Runs on a background worker thread so a slow or broken provider
        can't stall the turn.
        """
        providers = list(self._providers)
        if not providers or not user_content:
            return

        def _run() -> None:
            for provider in providers:
                try:
                    if messages is not None and self._provider_sync_accepts_messages(provider):
                        provider.sync_turn(
                            user_content, assistant_content,
                            session_id=session_id, messages=messages,
                        )
                    else:
                        provider.sync_turn(
                            user_content, assistant_content,
                            session_id=session_id,
                        )
                except Exception as e:
                    logger.warning("Memory provider '%s' sync_turn failed: %s", provider.name, e)

        self._submit_background(_run)

    def flush_pending(self, timeout: Optional[float] = None) -> bool:
        """Block until all submitted background tasks complete. True on success."""
        if self._sync_executor is None:
            return True
        try:
            future = self._sync_executor.submit(lambda: None)
            future.result(timeout=timeout)
            return True
        except Exception:
            return False

    # -- Tool routing --------------------------------------------------------

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect bare function schemas from every provider."""
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for provider in self._providers:
            for raw_schema in provider.get_tool_schemas():
                schema = _normalize_tool_schema(raw_schema)
                if schema is None:
                    continue
                name = schema.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    result.append(schema)
        return result

    def get_all_tool_names(self) -> set[str]:
        """All routable tool names."""
        return set(self._tool_to_provider.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is routable."""
        return tool_name in self._tool_to_provider

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Route a tool call to the registered provider."""
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return json_error(f"Unknown memory tool: {tool_name}")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.warning("Memory tool '%s' failed: %s", tool_name, e)
            return json_error(str(e))

    # -- Lifecycle hooks -----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Notify all providers of a new turn."""
        for provider in self._providers:
            try:
                provider.on_turn_start(turn_number, message, **kwargs)
            except Exception as e:
                logger.debug("Memory provider '%s' on_turn_start failed: %s", provider.name, e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Notify all providers of session end."""
        for provider in self._providers:
            try:
                provider.on_session_end(messages)
            except Exception as e:
                logger.debug("Memory provider '%s' on_session_end failed: %s", provider.name, e)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        """Notify all providers of session ID rotation."""
        for provider in self._providers:
            try:
                provider.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:
                logger.debug("Memory provider '%s' on_session_switch failed: %s", provider.name, e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Collect pre-compression text from all providers."""
        parts: List[str] = []
        for provider in self._providers:
            try:
                result = provider.on_pre_compress(messages)
                if result:
                    parts.append(result)
            except Exception as e:
                logger.debug("Memory provider '%s' on_pre_compress failed: %s", provider.name, e)
        return "\n\n".join(parts)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror a built-in memory-tool write to all external providers."""
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                provider.on_memory_write(action, target, content, metadata=metadata)
            except Exception as e:
                logger.debug("Memory provider '%s' on_memory_write failed: %s", provider.name, e)

    def shutdown_all(self) -> None:
        """Shut down all providers (reverse order)."""
        self._drain_sync_executor()
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning("Memory provider '%s' shutdown failed: %s", provider.name, e)

    def initialize_all(self, session_id: str, **kwargs: Any) -> None:
        """Initialize all providers."""
        try:
            from niaharness.prompts.soul import get_nia_home
            kwargs.setdefault("nia_home", str(get_nia_home()))
        except Exception:
            pass
        for provider in self._providers:
            try:
                provider.initialize(session_id, **kwargs)
            except Exception as e:
                logger.warning("Memory provider '%s' initialize failed: %s", provider.name, e)

    # -- Private: background executor ----------------------------------------

    def _submit_background(self, fn: Callable) -> None:
        """Submit fn to the daemon executor (inline fallback on failure)."""
        executor = self._get_sync_executor()
        if executor is None:
            try:
                fn()
            except Exception:
                pass
            return
        try:
            executor.submit(fn)
        except RuntimeError:
            # Executor shutting down — run inline.
            try:
                fn()
            except Exception:
                pass

    def _get_sync_executor(self) -> Optional[ThreadPoolExecutor]:
        """Lazily create a single-worker daemon ThreadPoolExecutor."""
        if self._sync_executor is not None:
            return self._sync_executor
        with self._sync_executor_lock:
            if self._sync_executor is None:
                try:
                    self._sync_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="mem-sync",
                    )
                except Exception:
                    return None
        return self._sync_executor

    def _drain_sync_executor(self) -> None:
        """Shut down the executor with a bounded wait."""
        with self._sync_executor_lock:
            executor = self._sync_executor
            self._sync_executor = None
        if executor is None:
            return
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
        # Bounded wait on a daemon thread.
        drainer = threading.Thread(
            target=executor.shutdown, kwargs={"wait": True},
            daemon=True, name="mem-sync-drain",
        )
        drainer.start()
        drainer.join(timeout=_SYNC_DRAIN_TIMEOUT_S)

    @staticmethod
    def _provider_sync_accepts_messages(provider: MemoryProvider) -> bool:
        """Check if provider.sync_turn accepts a 'messages' kwarg."""
        try:
            sig = inspect.signature(provider.sync_turn)
            return "messages" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_tool_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Normalize a tool schema to bare {name, description, parameters}."""
    if not isinstance(schema, dict):
        return None
    if "function" in schema and isinstance(schema["function"], dict):
        schema = schema["function"]
    if "name" not in schema:
        return None
    return schema


def json_error(msg: str) -> str:
    """Return a JSON error string."""
    import json
    return json.dumps({"success": False, "error": msg})


# ---------------------------------------------------------------------------
# Backwards-compat: preserve existing file-based memory helpers
# ---------------------------------------------------------------------------

# Re-export the existing helpers so callers don't break.
from niaharness.memory.paths import get_memory_entrypoint, get_project_memory_dir  # noqa: E402
from pathlib import Path  # noqa: E402
from re import sub as _re_sub  # noqa: E402


def list_memory_files(cwd: str | Path) -> list[Path]:
    """List memory markdown files for the project."""
    memory_dir = get_project_memory_dir(cwd)
    return sorted(path for path in memory_dir.glob("*.md"))


def add_memory_entry(cwd: str | Path, title: str, content: str) -> Path:
    """Create a memory file and append it to MEMORY.md."""
    memory_dir = get_project_memory_dir(cwd)
    slug = _re_sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_") or "memory"
    path = memory_dir / f"{slug}.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")

    entrypoint = get_memory_entrypoint(cwd)
    existing = entrypoint.read_text(encoding="utf-8") if entrypoint.exists() else "# Memory Index\n"
    if path.name not in existing:
        existing = existing.rstrip() + f"\n- [{title}]({path.name})\n"
        entrypoint.write_text(existing, encoding="utf-8")
    return path


def remove_memory_entry(cwd: str | Path, name: str) -> bool:
    """Delete a memory file and remove its index entry."""
    memory_dir = get_project_memory_dir(cwd)
    matches = [path for path in memory_dir.glob("*.md") if path.stem == name or path.name == name]
    if not matches:
        return False
    path = matches[0]
    if path.exists():
        path.unlink()
    entrypoint = get_memory_entrypoint(cwd)
    if entrypoint.exists():
        lines = [
            line for line in entrypoint.read_text(encoding="utf-8").splitlines()
            if path.name not in line
        ]
        entrypoint.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[MemoryManager] = None
_default_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Return the process-wide MemoryManager singleton."""
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = MemoryManager()
        return _default_manager


def reset_memory_manager() -> None:
    """Reset the singleton (for tests)."""
    global _default_manager
    with _default_manager_lock:
        if _default_manager is not None:
            _default_manager.shutdown_all()
        _default_manager = None


__all__ = [
    "MemoryManager",
    "StreamingContextScrubber",
    "add_memory_entry",
    "build_memory_context_block",
    "get_memory_manager",
    "json_error",
    "list_memory_files",
    "remove_memory_entry",
    "reset_memory_manager",
    "sanitize_context",
]
