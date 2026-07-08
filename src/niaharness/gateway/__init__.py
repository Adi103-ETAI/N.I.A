"""Messaging Gateway — chat platform adapters for NIA.

Ported from the reference project's gateway/ package (41,192 lines),
providing a pluggable system for connecting NIA to chat platforms
(Telegram, Discord, Slack, WhatsApp, etc.).

The gateway receives messages from a chat platform, routes them to NIA's
conversation engine, and delivers NIA's responses back to the platform.
Each platform is implemented as a ``PlatformAdapter`` that translates
between the platform's API and NIA's internal message format.

Architecture
------------
::

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Telegram    │     │   Discord    │     │    Slack     │
    │  Adapter     │     │   Adapter    │     │   Adapter    │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           └────────────┬───────┴────────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │   Gateway    │
                │   Router     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  NIA Engine  │
                └──────────────┘

Currently supported platforms:
  - **Telegram** (MVP) — long-polling bot API

Planned platforms (stubs only):
  - Discord, Slack, WhatsApp, Matrix, Signal

Why this matters
----------------
Without a gateway, NIA is limited to the terminal UI. The gateway lets
users interact with NIA from any chat platform — turning NIA into a
Jarvis-like assistant that's always available on their phone.

Usage::

    from niaharness.gateway import TelegramAdapter, GatewayRouter

    router = GatewayRouter(engine=my_engine)
    telegram = TelegramAdapter(token="...", router=router)
    await telegram.start()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class IncomingMessage:
    """A message received from a chat platform.

    Attributes:
        platform: The platform name (e.g. "telegram").
        platform_message_id: The message ID on the platform.
        platform_chat_id: The chat/channel ID on the platform.
        platform_user_id: The user ID on the platform.
        platform_username: The user's display name (optional).
        text: The message text.
        timestamp: When the message was sent (UTC).
        reply_to_message_id: The message ID being replied to (optional).
        metadata: Platform-specific metadata.
    """

    platform: str
    platform_message_id: str
    platform_chat_id: str
    platform_user_id: str
    platform_username: Optional[str] = None
    text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reply_to_message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    """A message to send to a chat platform.

    Attributes:
        platform_chat_id: The chat/channel ID to send to.
        text: The message text.
        reply_to_message_id: The message ID to reply to (optional).
        parse_mode: The parse mode (e.g. "Markdown", "HTML").
        metadata: Platform-specific metadata.
    """

    platform_chat_id: str
    text: str
    reply_to_message_id: Optional[str] = None
    parse_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Platform adapter ABC
# ---------------------------------------------------------------------------


class PlatformAdapter(ABC):
    """Abstract base for chat platform adapters.

    Each adapter translates between a platform's API and NIA's internal
    message format. Adapters are responsible for:
      - Polling or receiving messages from the platform
      - Translating platform messages to ``IncomingMessage``
      - Translating ``OutgoingMessage`` to platform API calls
      - Handling platform-specific auth (bot tokens, OAuth, etc.)
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """The platform name (e.g. "telegram")."""

    @abstractmethod
    async def start(self) -> None:
        """Start receiving messages from the platform."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving messages and clean up resources."""

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> str:
        """Send a message to the platform. Returns the platform message ID."""

    def set_message_handler(
        self, handler: Callable[[IncomingMessage], Any]
    ) -> None:
        """Set the callback for incoming messages.

        The handler is called for each incoming message. It should be async
        and return the response text (or None for no response).
        """
        self._message_handler = handler


# ---------------------------------------------------------------------------
# Gateway router
# ---------------------------------------------------------------------------


class GatewayRouter:
    """Routes messages between platform adapters and NIA's engine.

    The router maintains a registry of platform adapters and routes
    incoming messages to the NIA engine. Responses are sent back via
    the originating adapter.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._adapters: Dict[str, PlatformAdapter] = {}

    def register_adapter(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter."""
        self._adapters[adapter.platform_name] = adapter
        adapter.set_message_handler(self._handle_incoming)
        logger.info("Registered gateway adapter: %s", adapter.platform_name)

    def unregister_adapter(self, platform_name: str) -> None:
        """Unregister a platform adapter."""
        adapter = self._adapters.pop(platform_name, None)
        if adapter is not None:
            logger.info("Unregistered gateway adapter: %s", platform_name)

    def get_adapter(self, platform_name: str) -> Optional[PlatformAdapter]:
        """Get a registered adapter by platform name."""
        return self._adapters.get(platform_name)

    def list_adapters(self) -> List[str]:
        """Return a list of registered platform names."""
        return list(self._adapters.keys())

    async def start_all(self) -> None:
        """Start all registered adapters."""
        for adapter in self._adapters.values():
            try:
                await adapter.start()
            except Exception as exc:
                logger.error("Failed to start adapter %s: %s", adapter.platform_name, exc)

    async def stop_all(self) -> None:
        """Stop all registered adapters."""
        for adapter in self._adapters.values():
            try:
                await adapter.stop()
            except Exception as exc:
                logger.error("Failed to stop adapter %s: %s", adapter.platform_name, exc)

    async def _handle_incoming(self, message: IncomingMessage) -> None:
        """Handle an incoming message from a platform adapter.

        Routes the message to the NIA engine, then sends the response back
        via the originating adapter.
        """
        adapter = self._adapters.get(message.platform)
        if adapter is None:
            logger.warning("No adapter for platform '%s'", message.platform)
            return

        try:
            response_text = await self._route_to_engine(message)
            if response_text:
                outgoing = OutgoingMessage(
                    platform_chat_id=message.platform_chat_id,
                    text=response_text,
                    reply_to_message_id=message.platform_message_id,
                    parse_mode="Markdown",
                )
                await adapter.send_message(outgoing)
        except Exception as exc:
            logger.error("Error handling incoming message: %s", exc)
            # Send error message to user.
            try:
                await adapter.send_message(OutgoingMessage(
                    platform_chat_id=message.platform_chat_id,
                    text=f"Sorry, I encountered an error processing your message: {exc}",
                    reply_to_message_id=message.platform_message_id,
                ))
            except Exception:
                pass

    async def _route_to_engine(self, message: IncomingMessage) -> Optional[str]:
        """Route an incoming message to the NIA engine and return the response.

        Override this method to integrate with the actual NIA engine.
        The default implementation returns a stub response.
        """
        if self._engine is None:
            return f"[NIA Gateway] Received your message: {message.text[:100]}"
        # Delegate to the engine's process_message method.
        if hasattr(self._engine, "process_gateway_message"):
            return await self._engine.process_gateway_message(
                platform=message.platform,
                chat_id=message.platform_chat_id,
                user_id=message.platform_user_id,
                text=message.text,
            )
        return None


# ---------------------------------------------------------------------------
# Telegram adapter (MVP)
# ---------------------------------------------------------------------------


class TelegramAdapter(PlatformAdapter):
    """Telegram Bot API adapter (long-polling MVP).

    Uses Telegram's getUpdates long-polling API to receive messages.
    For production, consider switching to webhooks (requires a public URL).

    Configuration:
      - ``NIA_TELEGRAM_BOT_TOKEN`` env var, or
      - ``gateway.telegram.bot_token`` in config.yaml

    Usage::

        adapter = TelegramAdapter(token="123:abc...")
        router = GatewayRouter(engine=my_engine)
        router.register_adapter(adapter)
        await router.start_all()  # starts long-polling
    """

    API_BASE = "https://api.telegram.org"

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        router: Optional[GatewayRouter] = None,
    ) -> None:
        import os

        self._token = token or os.environ.get("NIA_TELEGRAM_BOT_TOKEN", "")
        self._router = router
        self._message_handler: Optional[Callable[[IncomingMessage], Any]] = None
        self._running = False
        self._offset = 0
        self._http_client = None

    @property
    def platform_name(self) -> str:
        return "telegram"

    def set_message_handler(self, handler: Callable[[IncomingMessage], Any]) -> None:
        self._message_handler = handler

    async def start(self) -> None:
        """Start long-polling for messages."""
        if not self._token:
            raise ValueError(
                "Telegram bot token required. Set NIA_TELEGRAM_BOT_TOKEN env var "
                "or pass token to TelegramAdapter."
            )
        self._running = True
        logger.info("Telegram adapter started (long-polling)")
        # The actual polling loop runs in the gateway's start_all() context.
        # Callers should await run_polling() in a background task.
        import asyncio

        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling and clean up."""
        self._running = False
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("Telegram adapter stopped")

    async def _get_http_client(self):
        """Lazily initialize the HTTP client."""
        if self._http_client is None:
            import httpx

            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def _poll_loop(self) -> None:
        """Long-polling loop for getUpdates."""
        logger.info("Telegram polling loop started")
        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._process_update(update)
            except Exception as exc:
                logger.error("Telegram polling error: %s", exc)
                import asyncio

                await asyncio.sleep(5)  # Back off on error

    async def _get_updates(self) -> List[Dict[str, Any]]:
        """Call getUpdates with long-polling (30s timeout)."""
        client = await self._get_http_client()
        response = await client.post(
            f"{self.API_BASE}/bot{self._token}/getUpdates",
            json={
                "offset": self._offset,
                "timeout": 30,  # long-poll timeout
                "allowed_updates": ["message"],
            },
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return []
        updates = data.get("result", [])
        if updates:
            # Update offset to avoid re-receiving the same updates.
            self._offset = updates[-1]["update_id"] + 1
        return updates

    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single update from Telegram."""
        message = update.get("message")
        if not message:
            return

        incoming = IncomingMessage(
            platform="telegram",
            platform_message_id=str(message.get("message_id", "")),
            platform_chat_id=str(message.get("chat", {}).get("id", "")),
            platform_user_id=str(message.get("from", {}).get("id", "")),
            platform_username=message.get("from", {}).get("username"),
            text=message.get("text", ""),
            timestamp=datetime.fromtimestamp(
                message.get("date", 0), tz=timezone.utc
            ),
            reply_to_message_id=str(message.get("reply_to_message", {}).get("message_id", ""))
            if message.get("reply_to_message")
            else None,
            metadata={"raw": message},
        )

        if self._message_handler is not None:
            import asyncio

            if asyncio.iscoroutinefunction(self._message_handler):
                await self._message_handler(incoming)
            else:
                self._message_handler(incoming)

    async def send_message(self, message: OutgoingMessage) -> str:
        """Send a message via the Telegram Bot API."""
        client = await self._get_http_client()
        payload: Dict[str, Any] = {
            "chat_id": message.platform_chat_id,
            "text": message.text,
        }
        if message.reply_to_message_id:
            payload["reply_to_message_id"] = int(message.reply_to_message_id)
        if message.parse_mode:
            payload["parse_mode"] = message.parse_mode

        response = await client.post(
            f"{self.API_BASE}/bot{self._token}/sendMessage",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data.get('description', 'unknown')}")
        return str(data.get("result", {}).get("message_id", ""))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_gateway_router(engine: Any = None) -> GatewayRouter:
    """Create a GatewayRouter with all configured platform adapters.

    Reads config to determine which platforms to enable. Currently only
    Telegram is supported (MVP).
    """
    router = GatewayRouter(engine=engine)

    # Telegram (if token is configured).
    import os

    telegram_token = os.environ.get("NIA_TELEGRAM_BOT_TOKEN", "")
    if telegram_token:
        router.register_adapter(TelegramAdapter(token=telegram_token))

    return router


__all__ = [
    "GatewayRouter",
    "IncomingMessage",
    "OutgoingMessage",
    "PlatformAdapter",
    "TelegramAdapter",
    "create_gateway_router",
]
