"""NIA TUI Gateway — JSON-RPC server with WebSocket + stdio transport.

Ported from Hermes Agent's ``tui_gateway/`` package (16,129 LOC across
11 Python files). Provides:

  - :mod:`transport` — Transport abstraction (StdioTransport, TeeTransport).
  - :mod:`server` — 118 RPC methods covering session management, prompt
    submission, image/file attachment, config, model, slash commands,
    completion, approval/clarify, voice, billing, process, MCP/env reload,
    projects, handoff, insights, rollback, browser, plugins, tools,
    agents, cron, learning, skills, shell, terminal, pet.
  - :mod:`ws` — WebSocket transport with token coalescing.
  - :mod:`entry` — Entry point with signal handling + crash logging.
  - :mod:`git_probe` — Git working-tree probing with single-flight cache.
  - :mod:`project_tree` — Project → repo → lane → session tree builder.
  - :mod:`slash_worker` — Persistent slash-command worker subprocess.
  - :mod:`event_publisher` — WebSocket back-channel for dashboard mirroring.
  - :mod:`loop_noise` — Suppress benign event-loop teardown noise.
  - :mod:`render` — Rendering bridge for Python-side rich output.
"""

from niaharness.tui_gateway.transport import (
    DROP_TRANSPORT,
    StdioTransport,
    TeeTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)
from niaharness.tui_gateway.server import (
    dispatch,
    handle_request,
    resolve_skin,
    write_json,
)
from niaharness.tui_gateway.ws import WSTransport, handle_ws
from niaharness.tui_gateway.loop_noise import install_loop_noise_filter
from niaharness.tui_gateway.git_probe import branch, repo_root, resolve, warm_roots
from niaharness.tui_gateway.project_tree import build_tree
from niaharness.tui_gateway.event_publisher import WsPublisherTransport

__all__ = [
    "DROP_TRANSPORT",
    "StdioTransport",
    "TeeTransport",
    "Transport",
    "WsPublisherTransport",
    "WSTransport",
    "bind_transport",
    "branch",
    "build_tree",
    "current_transport",
    "dispatch",
    "handle_request",
    "handle_ws",
    "install_loop_noise_filter",
    "repo_root",
    "reset_transport",
    "resolve",
    "resolve_skin",
    "warm_roots",
    "write_json",
]
