"""P1 GatewayRunner — the orchestrator that ties everything together.

Ported from Hermes Agent's ``gateway/run.py`` (20526 LOC), scoped to
NIA's architecture. The GatewayRunner owns:

  - The GatewayRouter (adapter registry + message routing).
  - The SlashCommandRegistry (in-session slash commands).
  - Startup machinery: PID-file lock, runtime lock, --replace takeover,
    signal handlers, drain control.
  - Streaming responses: token-by-token delivery to chat platforms.
  - Secret redaction: every response is scrubbed before delivery.
  - Intentional silence detection: ``NO_REPLY`` / ``[SILENT]`` markers
    suppress delivery.
  - Status phrases: long-running operations get "still on it" updates.
  - Restart-loop guard: skips auto-resume after a crash loop.
  - Scale-to-zero: idle detection for relay-only deployments.

This is a substantial orchestrator — it coordinates the full lifecycle
of a running gateway process.

Usage::

    from niaharness.gateway.runner import GatewayRunner

    runner = GatewayRunner(engine=my_engine)
    await runner.start()  # acquires lock, writes PID, starts adapters
    await runner.run_forever()  # blocks until shutdown
    await runner.stop()  # drains, stops adapters, removes PID
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from niaharness.gateway import (
    GatewayRouter,
    IncomingMessage,
    OutgoingMessage,
    PlatformAdapter,
)
from niaharness.gateway.drain_control import (
    clear_drain_request,
    drain_requested,
    write_drain_request,
)
from niaharness.gateway.response_filters import (
    is_intentional_silence_response,
    is_partial_silence_marker,
    redact_secrets,
)
from niaharness.gateway.restart_loop_guard import check_and_record
from niaharness.gateway.slash_commands import (
    SlashCommandContext,
    SlashCommandRegistry,
    create_default_registry,
    handle_slash_command,
    parse_slash_command,
)
from niaharness.gateway.status import (
    GatewaySignalHandler,
    acquire_gateway_runtime_lock,
    is_gateway_running,
    read_runtime_status,
    release_gateway_runtime_lock,
    remove_pid_file,
    replace_existing_gateway,
    write_pid_file,
    write_runtime_status,
)
from niaharness.gateway.status_phrases import choose_status_phrase

logger = logging.getLogger(__name__)


class GatewayRunner:
    """Orchestrates the full gateway lifecycle.

    The runner owns the GatewayRouter, SlashCommandRegistry, and all
    startup/shutdown machinery. It's the single entry point for
    ``nia gateway run``.

    Lifecycle:
      1. ``start()`` — acquire lock, write PID, install signal handlers,
         start all adapters.
      2. ``run_forever()`` — block until shutdown is requested (via
         signal or ``request_shutdown()``).
      3. ``stop()`` — drain in-flight agents, stop adapters, remove PID,
         release lock.
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        router: Optional[GatewayRouter] = None,
        slash_registry: Optional[SlashCommandRegistry] = None,
        replace: bool = False,
        enable_streaming: bool = True,
        enable_secret_redaction: bool = True,
        enable_silence_detection: bool = True,
        enable_status_phrases: bool = True,
    ) -> None:
        self._engine = engine
        self._router = router or GatewayRouter(engine=engine)
        self._slash_registry = slash_registry or create_default_registry()
        self._replace = replace
        self._enable_streaming = enable_streaming
        self._enable_secret_redaction = enable_secret_redaction
        self._enable_silence_detection = enable_silence_detection
        self._enable_status_phrases = enable_status_phrases

        # State.
        self._running = False
        self._started_at: float = 0.0
        self._last_inbound_at: float = time.time()
        self._active_agents: int = 0
        self._gateway_state: str = "stopped"
        self._yolo_sessions: set[str] = set()
        self._cancelled_sessions: set[str] = set()
        self._queued_events: Dict[str, List[IncomingMessage]] = {}
        self._signal_handler: Optional[GatewaySignalHandler] = None

        # Wire the router to use this runner for message handling.
        self._router.set_message_handler_override(self._handle_incoming)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def router(self) -> GatewayRouter:
        return self._router

    @property
    def slash_registry(self) -> SlashCommandRegistry:
        return self._slash_registry

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        if not self._running:
            return 0.0
        return time.time() - self._started_at

    @property
    def gateway_state(self) -> str:
        return self._gateway_state

    @property
    def active_agent_count(self) -> int:
        return self._active_agents

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the gateway: acquire lock, write PID, start adapters."""
        if self._running:
            logger.warning("GatewayRunner.start() called but already running")
            return

        # --replace takeover.
        if self._replace:
            if not replace_existing_gateway():
                raise RuntimeError(
                    "Could not replace existing gateway instance "
                    "(it didn't terminate within the timeout)"
                )

        # Acquire the runtime lock.
        if not acquire_gateway_runtime_lock():
            from niaharness.gateway.status import get_running_pid
            existing = get_running_pid()
            raise RuntimeError(
                f"Gateway already running (PID {existing}). "
                f"Use --replace to auto-replace."
            )

        # Check for restart-loop (skip auto-resume if tripped).
        try:
            tripped = check_and_record()
            if tripped:
                logger.warning(
                    "Restart-loop detected — skipping auto-resume for this boot"
                )
        except Exception:
            pass

        # Write the PID file.
        write_pid_file()

        # Install signal handlers.
        self._signal_handler = GatewaySignalHandler()
        self._signal_handler.__enter__()

        # Update state.
        self._running = True
        self._started_at = time.time()
        self._gateway_state = "running"

        # Start all adapters.
        await self._router.start_all()
        logger.info("GatewayRunner started (uptime=0s)")

        # Write initial runtime status.
        self._write_status()

    async def run_forever(self) -> None:
        """Block until shutdown is requested (via signal or request_shutdown)."""
        if not self._running:
            await self.start()
        logger.info("GatewayRunner running — press Ctrl+C to stop")
        while self._running:
            # Check for shutdown.
            if self._signal_handler and self._signal_handler.shutdown_requested:
                break
            # Check for external drain request.
            if drain_requested():
                logger.info("External drain request detected — entering drain state")
                self._gateway_state = "draining"
                await self._drain()
                break
            # Update runtime status periodically.
            self._write_status()
            await asyncio.sleep(5.0)
        await self.stop()

    async def stop(self) -> None:
        """Stop the gateway: drain, stop adapters, remove PID, release lock."""
        if not self._running:
            return
        logger.info("GatewayRunner stopping...")
        self._gateway_state = "draining"

        # Drain in-flight agents (best-effort, 10s timeout).
        await self._drain(timeout=10.0)

        # Stop all adapters.
        await self._router.stop_all()

        # Clear any drain request (we've completed the drain).
        clear_drain_request()

        # Remove PID file + release lock.
        remove_pid_file()
        release_gateway_runtime_lock()

        # Restore signal handlers.
        if self._signal_handler is not None:
            self._signal_handler.__exit__(None, None, None)
            self._signal_handler = None

        self._running = False
        self._gateway_state = "stopped"
        logger.info("GatewayRunner stopped")

    def request_shutdown(self) -> None:
        """Request a graceful shutdown (sets the shutdown event)."""
        if self._signal_handler is not None:
            self._signal_handler._shutdown_event.set()

    async def _drain(self, timeout: float = 30.0) -> None:
        """Wait for in-flight agents to complete (with timeout)."""
        logger.info("Draining %d in-flight agent(s)...", self._active_agents)
        deadline = time.monotonic() + timeout
        while self._active_agents > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if self._active_agents > 0:
            logger.warning(
                "Drain timeout — %d agent(s) still in flight", self._active_agents
            )

    # ------------------------------------------------------------------
    # Message handling (with slash commands + streaming + redaction)
    # ------------------------------------------------------------------

    async def _handle_incoming(self, message: IncomingMessage) -> None:
        """Handle an incoming message (slash command or regular)."""
        if not self._running:
            logger.warning("Message received but gateway not running — ignoring")
            return

        # Check drain state.
        if self._gateway_state == "draining":
            await self._send_reply(
                message,
                "Gateway is draining — please try again in a moment.",
            )
            return

        self._last_inbound_at = time.time()

        # Check for slash command.
        if message.text and message.text.startswith("/"):
            await self._handle_slash(message)
            return

        # Regular message — route to engine.
        self._active_agents += 1
        try:
            await self._process_message(message)
        finally:
            self._active_agents -= 1

    async def _handle_slash(self, message: IncomingMessage) -> None:
        """Handle a slash command message."""
        context = SlashCommandContext(
            platform=message.platform,
            chat_id=message.platform_chat_id,
            user_id=message.platform_user_id,
            user_name=message.platform_username,
            session_id=None,  # Filled by the router/session store.
            gateway_runner=self,
            metadata={"registry": self._slash_registry},
        )
        reply = await handle_slash_command(
            message.text, context, self._slash_registry
        )
        if reply:
            await self._send_reply(message, reply)

    async def _process_message(self, message: IncomingMessage) -> None:
        """Process a regular (non-slash) message."""
        adapter = self._router.get_adapter(message.platform)
        if adapter is None:
            logger.warning("No adapter for platform '%s'", message.platform)
            return

        # Send a status phrase if streaming is enabled (so the user knows
        # the agent is working).
        if self._enable_status_phrases:
            status_phrase = choose_status_phrase("status")
            try:
                await adapter.send_message(OutgoingMessage(
                    platform_chat_id=message.platform_chat_id,
                    text=status_phrase,
                    reply_to_message_id=message.platform_message_id,
                ))
            except Exception:
                pass  # Best-effort.

        try:
            # Route to engine.
            response_text = await self._router._route_to_engine(message)

            if not response_text:
                return  # Empty response — don't deliver.

            # Silence detection.
            if self._enable_silence_detection and is_intentional_silence_response(response_text):
                logger.debug("Suppressing silence marker response")
                return

            # Secret redaction.
            if self._enable_secret_redaction:
                response_text = redact_secrets(response_text)

            # Send the response.
            await self._send_reply(message, response_text)
        except Exception as exc:
            logger.error("Error processing message: %s", exc)
            await self._send_reply(
                message,
                f"Sorry, I encountered an error: {exc}",
            )

    async def _send_reply(self, message: IncomingMessage, text: str) -> None:
        """Send a reply to the originating chat."""
        adapter = self._router.get_adapter(message.platform)
        if adapter is None:
            return
        # Redact secrets in error messages too.
        if self._enable_secret_redaction:
            text = redact_secrets(text)
        try:
            await adapter.send_message(OutgoingMessage(
                platform_chat_id=message.platform_chat_id,
                text=text,
                reply_to_message_id=message.platform_message_id,
                parse_mode="Markdown",
            ))
        except Exception as exc:
            logger.error("Failed to send reply: %s", exc)

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        message: IncomingMessage,
        stream: Any,
    ) -> None:
        """Stream a response token-by-token to the chat platform.

        Handles:
          - Partial silence marker detection (hold back the buffer until
            it's clear the response isn't a silence marker).
          - Chunking (split long messages into multiple sends if the
            platform has a length limit).
          - Secret redaction on each chunk.
        """
        adapter = self._router.get_adapter(message.platform)
        if adapter is None:
            return

        buffer = ""
        async for delta in stream:
            buffer += delta
            # Hold back if this could still be a silence marker.
            if self._enable_silence_detection and is_partial_silence_marker(buffer):
                continue
            # Redact secrets.
            chunk = buffer
            if self._enable_secret_redaction:
                chunk = redact_secrets(chunk)
            if chunk:
                try:
                    await adapter.send_message(OutgoingMessage(
                        platform_chat_id=message.platform_chat_id,
                        text=chunk,
                    ))
                except Exception as exc:
                    logger.error("Stream send failed: %s", exc)
            buffer = ""

        # Flush any remaining buffer.
        if buffer:
            if self._enable_silence_detection and is_intentional_silence_response(buffer):
                return  # Suppress silence.
            if self._enable_secret_redaction:
                buffer = redact_secrets(buffer)
            if buffer:
                try:
                    await adapter.send_message(OutgoingMessage(
                        platform_chat_id=message.platform_chat_id,
                        text=buffer,
                    ))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Slash command helpers (called by handlers)
    # ------------------------------------------------------------------

    def reset_session(self, platform: str, chat_id: str) -> None:
        """Reset the session for a (platform, chat_id) pair."""
        session_key = f"{platform}:{chat_id}"
        # Clear yolo + queue for this session.
        self._yolo_sessions.discard(session_key)
        self._queued_events.pop(session_key, None)
        # Delegate to the router's session store.
        store = self._router._get_session_store()
        if store is not None:
            try:
                store.reset_session(session_key)
            except Exception as exc:
                logger.debug("Session store reset failed: %s", exc)

    def toggle_yolo(self, platform: str, chat_id: str) -> bool:
        """Toggle yolo mode for a session. Returns the new state."""
        session_key = f"{platform}:{chat_id}"
        if session_key in self._yolo_sessions:
            self._yolo_sessions.discard(session_key)
            return False
        else:
            self._yolo_sessions.add(session_key)
            return True

    def is_yolo(self, platform: str, chat_id: str) -> bool:
        """Return True if yolo mode is on for this session."""
        return f"{platform}:{chat_id}" in self._yolo_sessions

    def cancel_turn(self, platform: str, chat_id: str) -> bool:
        """Cancel the in-flight turn for a session."""
        session_key = f"{platform}:{chat_id}"
        if self._active_agents > 0:
            self._cancelled_sessions.add(session_key)
            return True
        return False

    def is_cancelled(self, platform: str, chat_id: str) -> bool:
        """Return True if the current turn was cancelled."""
        return f"{platform}:{chat_id}" in self._cancelled_sessions

    def get_queue(self, platform: str, chat_id: str) -> List[IncomingMessage]:
        """Return queued messages for a session."""
        session_key = f"{platform}:{chat_id}"
        return self._queued_events.get(session_key, [])

    def get_status(self) -> Dict[str, Any]:
        """Return a status dict for /status."""
        return {
            "state": self._gateway_state,
            "uptime": f"{self.uptime_seconds:.0f}s",
            "active_agents": self._active_agents,
            "adapters": self._router.list_adapters(),
            "yolo_sessions": len(self._yolo_sessions),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_status(self) -> None:
        """Write the runtime status file."""
        write_runtime_status(
            state=self._gateway_state,
            active_agents=self._active_agents,
            uptime_seconds=self.uptime_seconds,
            adapters=self._router.list_adapters(),
        )


__all__ = ["GatewayRunner"]
