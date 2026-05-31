"""N.I.A Backend Host - JSON-lines backend for the React frontend.

This is the Python process that the React frontend spawns.
It reads requests from stdin, processes them via N.I.A's brain,
and writes events to stdout with the OHJSON: prefix.
"""

from __future__ import annotations

import asyncio
import json
import sys
import logging
from dataclasses import dataclass
from typing import Any

from agents.nia.nia import NIA
from agents.nia.ui.protocol import (
    BackendEvent,
    FrontendRequest,
    TranscriptItem,
    build_state_payload,
)

logger = logging.getLogger(__name__)

OHJSON_PREFIX = "OHJSON:"


@dataclass
class BackendHostConfig:
    """Configuration for the backend host."""
    working_directory: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


class NIABackendHost:
    """N.I.A's backend host for the React frontend.

    Protocol:
    - Reads JSON requests from stdin (one per line)
    - Processes via N.I.A brain (LLM) + OpenHarness tools
    - Writes OHJSON-prefixed events to stdout
    """

    def __init__(self, config: BackendHostConfig) -> None:
        self._config = config
        self._nia: NIA | None = None
        self._request_queue: asyncio.Queue[FrontendRequest] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._running = False

    async def run(self) -> None:
        """Main run loop."""
        # Initialize N.I.A
        self._nia = NIA(working_directory=self._config.working_directory)
        await self._nia.initialize()

        # Configure provider if specified
        if self._config.provider:
            self._nia.switch_provider(self._config.provider, self._config.model)

        self._running = True

        # Emit ready event
        state = self._build_state()
        self._emit(BackendEvent(
            type="ready",
            state=state,
            commands=["/connect", "/models", "/provider", "/status", "/clear", "/help"],
        ))

        self._emit(BackendEvent(type="state_snapshot", state=state))

        # Start stdin reader
        reader_task = asyncio.create_task(self._read_requests())

        # Main processing loop
        try:
            while self._running:
                try:
                    request = await asyncio.wait_for(self._request_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self._process_request(request)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            reader_task.cancel()
            if self._nia:
                self._nia.shutdown()

    async def _read_requests(self) -> None:
        """Read requests from stdin in a thread."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    # EOF
                    await self._request_queue.put(FrontendRequest(type="shutdown"))
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    request = FrontendRequest(**data)
                    await self._request_queue.put(request)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from frontend: {line[:100]}")
            except Exception as e:
                logger.error(f"Error reading stdin: {e}")
                break

    async def _process_request(self, request: FrontendRequest) -> None:
        """Process a single request from the frontend."""
        if request.type == "shutdown":
            self._running = False
            self._emit(BackendEvent(type="shutdown"))
            return

        if request.type == "submit_line":
            await self._process_line(request.line)
            return

        # Other request types can be added here

    async def _process_line(self, line: str) -> None:
        """Process a user line through N.I.A."""
        if not self._nia:
            return

        # Emit user message to transcript
        self._emit(BackendEvent(
            type="transcript_item",
            item=TranscriptItem(role="user", text=line),
        ))

        try:
            # Get response from N.I.A brain
            response = await self._nia.process(line)

            # Emit assistant response
            self._emit(BackendEvent(
                type="transcript_item",
                item=TranscriptItem(role="assistant", text=response),
            ))

            # Emit streaming delta (full response at once for simplicity)
            self._emit(BackendEvent(type="assistant_delta", message=response))
            self._emit(BackendEvent(type="assistant_complete", message=response))

        except Exception as e:
            error_msg = f"Error processing: {e}"
            logger.error(error_msg)
            self._emit(BackendEvent(
                type="transcript_item",
                item=TranscriptItem(role="system", text=error_msg, is_error=True),
            ))

        # Emit final state
        self._emit(BackendEvent(type="state_snapshot", state=self._build_state()))
        self._emit(BackendEvent(type="line_complete"))

    def _build_state(self) -> dict[str, Any]:
        """Build current state snapshot."""
        provider_id = ""
        model = ""

        if self._nia and self._nia._provider_registry:
            provider_id = self._nia._provider_registry._active_provider_id or ""
            model = self._nia._provider_registry.get_active_model() or ""

        return build_state_payload(
            provider_id=provider_id,
            model=model,
            cwd=self._config.working_directory,
            auth_status="configured" if provider_id else "missing",
        )

    def _emit(self, event: BackendEvent) -> None:
        """Emit an event to stdout with OHJSON prefix."""
        try:
            line = OHJSON_PREFIX + event.model_dump_json() + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error emitting event: {e}")


async def run_nia_backend(config: BackendHostConfig) -> None:
    """Run the N.I.A backend host."""
    host = NIABackendHost(config)
    await host.run()
