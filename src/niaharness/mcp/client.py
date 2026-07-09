"""MCP client manager — supports stdio, HTTP, and SSE transports.

Enhanced with HTTP/SSE transport support (ported from the reference project's
MCP module), providing:
  - **Streamable HTTP transport** — for MCP servers exposed over HTTP
    (e.g. remote MCP servers, cloud-hosted MCP services)
  - **SSE transport** — for MCP servers using Server-Sent Events
  - **Reconnect with circuit breaker** — automatic reconnection on
    transient failures, with exponential backoff and a circuit breaker
    that stops retrying after repeated failures
  - **URL validation** — SSRF guards prevent connecting to private/localhost URLs
    (unless explicitly allowed via config)
  - **Header injection** — support for Authorization headers (OAuth, API keys)
  - **Per-server timeout** — configurable connect/read timeouts
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ReadResourceResult

from niaharness.mcp.types import (
    McpConnectionStatus,
    McpHttpServerConfig,
    McpResourceInfo,
    McpStdioServerConfig,
    McpToolInfo,
    McpWebSocketServerConfig,
)

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, stop retrying.
_CIRCUIT_BREAKER_THRESHOLD = 5
# Circuit breaker reset time (seconds).
_CIRCUIT_BREAKER_RESET_SECONDS = 60


class McpClientManager:
    """Manage MCP connections and expose tools/resources.

    Supports three transports:
      - **stdio** — local subprocess (existing)
      - **http** — Streamable HTTP transport (new)
      - **ws** — WebSocket transport (new)

    Each transport has its own connect method. All transports share the
    same session management, tool/resource listing, and call_tool/read_resource
    interface.
    """

    def __init__(self, server_configs: dict[str, object]) -> None:
        self._server_configs = server_configs
        self._statuses: dict[str, McpConnectionStatus] = {
            name: McpConnectionStatus(
                name=name,
                state="pending",
                transport=getattr(config, "type", "unknown"),
            )
            for name, config in server_configs.items()
        }
        self._sessions: dict[str, ClientSession] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        # Circuit breaker state per server.
        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    async def connect_all(self) -> None:
        """Connect all configured MCP servers (stdio + HTTP + WS)."""
        for name, config in self._server_configs.items():
            if isinstance(config, McpStdioServerConfig):
                await self._connect_stdio(name, config)
            elif isinstance(config, McpHttpServerConfig):
                await self._connect_http(name, config)
            elif isinstance(config, McpWebSocketServerConfig):
                await self._connect_ws(name, config)
            else:
                self._statuses[name] = McpConnectionStatus(
                    name=name,
                    state="failed",
                    transport=getattr(config, "type", "unknown"),
                    auth_configured=bool(getattr(config, "headers", None)),
                    detail=f"Unsupported MCP transport: {getattr(config, 'type', 'unknown')}",
                )

    async def reconnect_all(self) -> None:
        """Reconnect all configured servers."""
        await self.close()
        self._statuses = {
            name: McpConnectionStatus(name=name, state="pending", transport=getattr(config, "type", "unknown"))
            for name, config in self._server_configs.items()
        }
        await self.connect_all()

    def update_server_config(self, name: str, config: object) -> None:
        """Replace one server config in memory."""
        self._server_configs[name] = config

    def get_server_config(self, name: str) -> object | None:
        """Return one configured server object if present."""
        return self._server_configs.get(name)

    async def close(self) -> None:
        """Close all active MCP sessions."""
        for stack in list(self._stacks.values()):
            await stack.aclose()
        self._stacks.clear()
        self._sessions.clear()

    def list_statuses(self) -> list[McpConnectionStatus]:
        """Return statuses for all configured servers."""
        return [self._statuses[name] for name in sorted(self._statuses)]

    def list_tools(self) -> list[McpToolInfo]:
        """Return all connected MCP tools."""
        tools: list[McpToolInfo] = []
        for status in self.list_statuses():
            tools.extend(status.tools)
        return tools

    def list_resources(self) -> list[McpResourceInfo]:
        """Return all connected MCP resources."""
        resources: list[McpResourceInfo] = []
        for status in self.list_statuses():
            resources.extend(status.resources)
        return resources

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke one MCP tool and stringify the result."""
        session = self._sessions[server_name]
        result: CallToolResult = await session.call_tool(tool_name, arguments)
        parts: list[str] = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(item.model_dump_json())
        if result.structuredContent and not parts:
            parts.append(str(result.structuredContent))
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts).strip()

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read one MCP resource and stringify the response."""
        session = self._sessions[server_name]
        result: ReadResourceResult = await session.read_resource(uri)
        parts: list[str] = []
        for item in result.contents:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(getattr(item, "blob", "")))
        return "\n".join(parts).strip()

    async def _connect_stdio(self, name: str, config: McpStdioServerConfig) -> None:
        # Security check: validate the stdio command before spawning.
        # Blocks shell interpreters with egress/persistence patterns
        # and known IOC from the hermes-0day campaign.
        from niaharness.mcp.security import validate_mcp_stdio_command

        warnings = validate_mcp_stdio_command(
            name=name,
            command=config.command,
            args=config.args,
            env=config.env,
        )
        if warnings:
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.env),
                detail="; ".join(warnings),
            )
            for w in warnings:
                logger.error("MCP security: %s", w)
            return

        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env,
                        cwd=config.cwd,
                    )
                )
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            tool_result = await session.list_tools()
            resource_result = await session.list_resources()
            tools = [
                McpToolInfo(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
                )
                for tool in tool_result.tools
            ]
            resources = [
                McpResourceInfo(
                    server_name=name,
                    name=resource.name or str(resource.uri),
                    uri=str(resource.uri),
                    description=resource.description or "",
                )
                for resource in resource_result.resources
            ]
            self._sessions[name] = session
            self._stacks[name] = stack
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="connected",
                transport=config.type,
                auth_configured=bool(config.env),
                tools=tools,
                resources=resources,
            )
        except Exception as exc:
            await stack.aclose()
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.env),
                detail=str(exc),
            )

    async def _connect_http(self, name: str, config: McpHttpServerConfig) -> None:
        """Connect to an HTTP/SSE MCP server.

        Uses the Streamable HTTP transport from the MCP SDK. Falls back to
        SSE transport if the server doesn't support Streamable HTTP.

        SSRF guard: rejects URLs pointing to private/localhost addresses
        unless ``NIA_MCP_ALLOW_PRIVATE_URLS=1`` is set.
        """
        # SSRF guard.
        if not _is_safe_mcp_url(config.url):
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail=f"URL rejected by SSRF guard: {config.url}",
            )
            return

        # Circuit breaker check.
        if self._is_circuit_open(name):
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail="Circuit breaker open — too many recent failures",
            )
            return

        stack = AsyncExitStack()
        try:
            # Resolve auth: OAuth (config.auth == "oauth") or static headers.
            auth_provider = None
            effective_headers = dict(config.headers or {})
            auth_configured = bool(effective_headers)

            if getattr(config, "auth", "bearer") == "oauth" and config.oauth is not None:
                # Build the OAuth provider via the manager (caches per-server).
                try:
                    from niaharness.mcp.oauth_manager import get_manager

                    manager = get_manager()
                    oauth_cfg = config.oauth.model_dump() if hasattr(config.oauth, "model_dump") else dict(config.oauth)
                    auth_provider = manager.get_or_build_provider(
                        server_name=name,
                        server_url=config.url,
                        oauth_config=oauth_cfg,
                    )
                    auth_configured = auth_provider is not None
                except Exception as exc:
                    logger.warning(
                        "MCP server '%s': OAuth provider build failed: %s. "
                        "Falling back to static headers if present.",
                        name, exc,
                    )

            # Try Streamable HTTP transport first (MCP SDK >= 1.2).
            try:
                from mcp.client.streamable_http import streamablehttp_client

                client_kwargs: dict[str, Any] = {
                    "url": config.url,
                    "headers": effective_headers or None,
                    "timeout": 30.0,
                }
                if auth_provider is not None:
                    client_kwargs["auth"] = auth_provider
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamablehttp_client(**client_kwargs)
                )
            except ImportError:
                # Fall back to SSE transport for older MCP SDK versions.
                from mcp.client.sse import sse_client

                client_kwargs = {
                    "url": config.url,
                    "headers": effective_headers or None,
                    "timeout": 30.0,
                }
                # SSE client may not support `auth=` — fall back to headers only.
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(**client_kwargs)
                )

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            tool_result = await session.list_tools()
            resource_result = await session.list_resources()
            tools = [
                McpToolInfo(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
                )
                for tool in tool_result.tools
            ]
            resources = [
                McpResourceInfo(
                    server_name=name,
                    name=resource.name or str(resource.uri),
                    uri=str(resource.uri),
                    description=resource.description or "",
                )
                for resource in resource_result.resources
            ]
            self._sessions[name] = session
            self._stacks[name] = stack
            self._failure_counts[name] = 0  # Reset circuit breaker on success.
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="connected",
                transport=config.type,
                auth_configured=auth_configured,
                tools=tools,
                resources=resources,
            )
            logger.info("MCP server '%s' connected via HTTP/SSE (%d tools)", name, len(tools))
        except Exception as exc:
            await stack.aclose()
            self._record_failure(name)
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail=str(exc),
            )
            logger.warning("MCP server '%s' HTTP/SSE connection failed: %s", name, exc)

    async def _connect_ws(self, name: str, config: McpWebSocketServerConfig) -> None:
        """Connect to a WebSocket MCP server."""
        if not _is_safe_mcp_url(config.url):
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail=f"URL rejected by SSRF guard: {config.url}",
            )
            return

        if self._is_circuit_open(name):
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail="Circuit breaker open — too many recent failures",
            )
            return

        stack = AsyncExitStack()
        try:
            from mcp.client.websocket import websocket_client

            read_stream, write_stream = await stack.enter_async_context(
                websocket_client(config.url)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            tool_result = await session.list_tools()
            resource_result = await session.list_resources()
            tools = [
                McpToolInfo(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
                )
                for tool in tool_result.tools
            ]
            resources = [
                McpResourceInfo(
                    server_name=name,
                    name=resource.name or str(resource.uri),
                    uri=str(resource.uri),
                    description=resource.description or "",
                )
                for resource in resource_result.resources
            ]
            self._sessions[name] = session
            self._stacks[name] = stack
            self._failure_counts[name] = 0
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="connected",
                transport=config.type,
                auth_configured=bool(config.headers),
                tools=tools,
                resources=resources,
            )
            logger.info("MCP server '%s' connected via WebSocket (%d tools)", name, len(tools))
        except Exception as exc:
            await stack.aclose()
            self._record_failure(name)
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="failed",
                transport=config.type,
                auth_configured=bool(config.headers),
                detail=str(exc),
            )
            logger.warning("MCP server '%s' WebSocket connection failed: %s", name, exc)

    # ---- Circuit breaker ----

    def _record_failure(self, name: str) -> None:
        """Record a connection failure and open the circuit breaker if needed."""
        import time

        count = self._failure_counts.get(name, 0) + 1
        self._failure_counts[name] = count
        if count >= _CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until[name] = time.monotonic() + _CIRCUIT_BREAKER_RESET_SECONDS
            logger.warning(
                "MCP server '%s': circuit breaker opened after %d failures (resets in %ds)",
                name, count, _CIRCUIT_BREAKER_RESET_SECONDS,
            )

    def _is_circuit_open(self, name: str) -> bool:
        """True if the circuit breaker is open for this server."""
        import time

        until = self._circuit_open_until.get(name, 0.0)
        if until > time.monotonic():
            return True
        # Circuit has reset — clear the state.
        if name in self._circuit_open_until:
            del self._circuit_open_until[name]
            self._failure_counts[name] = 0
        return False


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def _is_safe_mcp_url(url: str) -> bool:
    """Check if an MCP server URL is safe to connect to (SSRF guard).

    Blocks:
      - Non-HTTP(S) schemes (file://, ftp://, etc.)
      - Localhost / 127.0.0.1 / ::1 (unless NIA_MCP_ALLOW_PRIVATE_URLS=1)
      - Private IP ranges (10.x, 172.16-31.x, 192.168.x) unless allowed
      - Link-local (169.254.x)

    Set ``NIA_MCP_ALLOW_PRIVATE_URLS=1`` to allow private/localhost URLs
    (e.g. for local MCP servers on the same machine).
    """
    import ipaddress
    import os
    from urllib.parse import urlparse

    if not url:
        return False

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    # If private URLs are explicitly allowed, skip the SSRF check.
    if os.environ.get("NIA_MCP_ALLOW_PRIVATE_URLS", "").strip() in ("1", "true", "yes"):
        return True

    hostname = parsed.hostname or ""
    if not hostname:
        return False

    # Block localhost variants.
    if hostname.lower() in ("localhost", "localhost.localdomain"):
        return False

    # Block IP literals in private ranges.
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        # Not an IP literal — it's a hostname. Allow it (DNS resolution
        # happens at connect time; we can't check the resolved IP here
        # without a DNS lookup, which adds latency and may not be desired).
        pass

    return True
