"""MCP configuration and state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class McpStdioServerConfig(BaseModel):
    """stdio MCP server configuration."""

    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class OAuthConfig(BaseModel):
    """OAuth 2.1 + PKCE configuration for an MCP server.

    When ``auth="oauth"`` is set on a :class:`McpHttpServerConfig`, this
    config drives the OAuth flow. All fields are optional — sensible
    defaults are applied by :func:`niaharness.mcp.oauth.build_oauth_auth`.
    """

    client_id: str | None = None
    """Pre-registered client ID (skips RFC 7591 dynamic registration)."""

    client_secret: str | None = None
    """Optional client secret for confidential clients."""

    client_name: str = "NIA Agent"
    """Client name sent to the authorization server."""

    scope: str | None = None
    """Space-delimited scopes to request (e.g. ``"read write"``)."""

    redirect_port: int = 0
    """Callback port. 0 = auto-pick a free port."""

    timeout: float = 300.0
    """Authorization flow timeout in seconds (default 5 minutes)."""


class McpHttpServerConfig(BaseModel):
    """HTTP MCP server configuration.

    Supports three auth modes:
      - ``auth="bearer"`` (default): static Bearer token via ``headers["Authorization"]``.
      - ``auth="oauth"``: OAuth 2.1 + PKCE flow via :mod:`niaharness.mcp.oauth`.
        Requires the ``oauth`` field to be set.
      - ``auth="none"``: no auth (anonymous server).
    """

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    auth: Literal["bearer", "oauth", "none"] = "bearer"
    oauth: OAuthConfig | None = None


class McpWebSocketServerConfig(BaseModel):
    """WebSocket MCP server configuration."""

    type: Literal["ws"] = "ws"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


McpServerConfig = McpStdioServerConfig | McpHttpServerConfig | McpWebSocketServerConfig


class McpJsonConfig(BaseModel):
    """Config file shape used by plugins and project files."""

    mcpServers: dict[str, McpServerConfig] = Field(default_factory=dict)


@dataclass(frozen=True)
class McpToolInfo:
    """Tool metadata exposed by an MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class McpResourceInfo:
    """Resource metadata exposed by an MCP server."""

    server_name: str
    name: str
    uri: str
    description: str = ""


@dataclass
class McpConnectionStatus:
    """Runtime status for one MCP server."""

    name: str
    state: Literal["connected", "failed", "pending", "disabled"]
    detail: str = ""
    transport: str = "unknown"
    auth_configured: bool = False
    tools: list[McpToolInfo] = field(default_factory=list)
    resources: list[McpResourceInfo] = field(default_factory=list)
