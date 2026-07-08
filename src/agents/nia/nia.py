"""N.I.A — Neural Intelligence Assistant.

NIA is the agent. niaharness is its runtime.

NIA owns: identity (SOUL.md), memory, personality, proactive review.
niaharness owns: tool execution (57 tools), permissions, hooks, MCP, cost tracking.

There is ONE LLM call per turn — the QueryEngine's. NIA does not make a
separate "thinking" LLM call. The LLM IS the brain. This mirrors Hermes's
AIAgent architecture (one class, no separate brain).

Architecture:
┌─────────────────────────────────────────────────┐
│                   N.I.A (agent)                  │
│  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │
│  │ Memory   │ │ Context  │ │ Personality    │   │
│  │ (file)   │ │ (env)    │ │ (SOUL.md tone) │   │
│  └──────────┘ └──────────┘ └────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │ QueryEngine (niaharness — the runtime)   │   │
│  │  • LLM call (THE brain)                  │   │
│  │  • Tool execution (57 tools)             │   │
│  │  • Permissions, hooks, MCP               │   │
│  │  • Background review (proactive)         │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from niaharness.config.settings import Settings
from niaharness.engine.query_engine import QueryEngine
from niaharness.engine.stream_events import StreamEvent
from niaharness.hooks import HookExecutor, HookExecutionContext, HookRegistry
from niaharness.permissions import PermissionChecker
from niaharness.tools import create_default_tool_registry, register_nia_tools

from agents.nia.core.memory import Memory
from agents.nia.core.context import Context
from agents.nia.core.personality import Personality, PersonalityConfig

logger = logging.getLogger(__name__)


class NIA:
    """N.I.A — the agent. niaharness is its runtime.

    Use::

        nia = NIA(working_directory="/path/to/project")
        await nia.initialize(api_key="sk-...", model="claude-3-opus")
        async for event in nia.chat("Read main.py and summarize it"):
            print(event)
        await nia.shutdown()

    NIA does NOT make a separate "thinking" LLM call. The QueryEngine's
    LLM call IS the brain. NIA's job is to own identity, memory, and
    proactive behavior — then hand each turn to niaharness for execution.
    """

    def __init__(
        self,
        working_directory: str | None = None,
        personality_config: PersonalityConfig | None = None,
    ) -> None:
        self._working_directory = working_directory or str(Path.cwd())
        self._personality = Personality(personality_config)
        self._memory = Memory(storage_path=Path.home() / ".nia" / "memory.json")
        self._context = Context()
        self._engine: QueryEngine | None = None
        self._mcp_manager: Any = None
        self._hook_executor: HookExecutor | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_client: Any = None,
        mcp_manager: Any = None,
    ) -> str:
        """Boot NIA. Returns the greeting string.

        Args:
            api_key: API key (defaults to settings-resolved key).
            model: Model name (defaults to settings model).
            base_url: Optional base URL override.
            api_client: Pre-built API client (skips internal client creation).
            mcp_manager: Pre-built MCP manager (skips internal MCP connection).
        """
        # Load NIA's identity layer.
        self._memory.load()
        self._context.detect_environment(self._working_directory)

        # Resolve settings.
        settings = Settings()

        # Build the API client (the body's voice).
        if api_client is not None:
            resolved_api_client = api_client
        else:
            resolved_api_client = self._build_api_client(
                settings, api_key=api_key, base_url=base_url
            )

        # Build the MCP manager (the body's external tools).
        if mcp_manager is not None:
            self._mcp_manager = mcp_manager
        else:
            self._mcp_manager = await self._build_mcp_manager(settings)

        # Build the tool registry (the body's hands).
        tool_registry = create_default_tool_registry(self._mcp_manager)

        # Wire NIA's memory + context into the nia_memory/nia_context tools.
        # This is the one line that makes those tools functional.
        register_nia_tools(tool_registry, self._memory, self._context, self._engine)

        # Build the hook executor (the body's reflexes).
        resolved_model = model or settings.model
        self._hook_executor = HookExecutor(
            HookRegistry(),
            HookExecutionContext(
                cwd=Path(self._working_directory).resolve(),
                api_client=resolved_api_client,
                default_model=resolved_model,
            ),
        )

        # Build the QueryEngine (the body — does the actual LLM + tool loop).
        self._engine = QueryEngine(
            api_client=resolved_api_client,
            tool_registry=tool_registry,
            permission_checker=PermissionChecker(settings.permission),
            cwd=self._working_directory,
            model=resolved_model,
            system_prompt=self._build_system_prompt(),
            max_tokens=settings.max_tokens,
            hook_executor=self._hook_executor,
            tool_metadata={
                "mcp_manager": self._mcp_manager,
                "api_client": resolved_api_client,
                "model": resolved_model,
                "max_tokens": settings.max_tokens,
            },
            memory=self._memory,  # enables background review (proactive layer)
        )

        self._initialized = True
        logger.info(
            "N.I.A ready. Model: %s, Tools: %d",
            resolved_model,
            len(tool_registry._tools) if hasattr(tool_registry, "_tools") else 0,
        )
        return self._greet()

    async def chat(self, message: str) -> AsyncIterator[StreamEvent]:
        """Send a message to NIA. Yields streaming events.

        This is the main entry point for a conversation turn. The
        QueryEngine handles the LLM call + tool execution loop + background
        review spawning. NIA does NOT make a separate "thinking" call.

        Args:
            message: The user's message.

        Yields:
            StreamEvent: Assistant text deltas, tool events, turn-complete.
        """
        if not self._initialized or self._engine is None:
            raise RuntimeError("NIA not initialized. Call await nia.initialize() first.")
        async for event in self._engine.submit_message(message):
            yield event

    async def shutdown(self) -> None:
        """Clean shutdown. Saves memory, closes MCP."""
        self._initialized = False
        try:
            self._memory.save()
        except Exception as exc:
            logger.warning("Memory save failed during shutdown: %s", exc)
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.close()
            except Exception as exc:
                logger.warning("MCP close failed during shutdown: %s", exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> QueryEngine | None:
        """The underlying QueryEngine (niaharness runtime)."""
        return self._engine

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def context(self) -> Context:
        return self._context

    @property
    def personality(self) -> Personality:
        return self._personality

    @property
    def initialized(self) -> bool:
        return self._initialized

    def get_status(self) -> dict[str, Any]:
        """Return a status dict for UIs / monitoring."""
        return {
            "state": "ready" if self._initialized else "uninitialized",
            "model": self._engine._model if self._engine else None,
            "memory": self._memory.get_stats() if self._memory else None,
            "tools": (
                len(self._engine._tool_registry._tools)
                if self._engine and hasattr(self._engine._tool_registry, "_tools")
                else 0
            ),
            "cwd": self._working_directory,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_api_client(
        self, settings: Settings, *, api_key: str | None, base_url: str | None
    ) -> Any:
        """Build the API client (Anthropic or OpenAI-compatible)."""
        resolved_key = api_key or settings.resolve_api_key()
        resolved_base = base_url or settings.base_url

        if getattr(settings, "api_format", "anthropic") == "openai":
            from niaharness.api.openai_client import OpenAICompatibleClient

            return OpenAICompatibleClient(api_key=resolved_key, base_url=resolved_base)
        from niaharness.api.client import AnthropicApiClient

        return AnthropicApiClient(api_key=resolved_key, base_url=resolved_base)

    async def _build_mcp_manager(self, settings: Settings) -> Any:
        """Build and connect the MCP client manager."""
        from niaharness.mcp.client import McpClientManager
        from niaharness.mcp.config import load_mcp_server_configs

        mcp_servers = load_mcp_server_configs(settings, [])
        manager = McpClientManager(mcp_servers)
        try:
            await manager.connect_all()
        except Exception as exc:
            logger.warning("MCP connect failed (non-fatal): %s", exc)
        return manager

    def _build_system_prompt(self) -> str:
        """Build NIA's merged system prompt.

        Layout (highest priority first):
          1. SOUL.md (NIA's identity — loaded from ~/.nia/SOUL.md)
          2. Personality (Jarvis tone)
          3. Memory summary (continuity across sessions)
          4. niaharness base (tool instructions, safety rules, environment)

        SOUL.md and the niaharness base come from
        ``niaharness.prompts.system_prompt.build_system_prompt``, which
        already prepends SOUL.md. We inject personality + memory before it.
        """
        from niaharness.prompts.system_prompt import build_system_prompt

        # build_system_prompt() returns: SOUL.md + base rules + environment.
        base = build_system_prompt(cwd=self._working_directory)

        # Personality block (Jarvis tone).
        personality_block = (
            "# Personality\n"
            "Tone: Professional, confident, slightly witty. "
            "Style: Direct and efficient. Voice: Calm authority.\n"
            "When appropriate, use dry wit — never forced."
        )

        # Memory block (continuity).
        memory_block = ""
        try:
            stats = self._memory.get_stats()
            if stats.get("total_memories", 0) > 0:
                memory_block = (
                    f"# Memory\nYou have {stats['total_memories']} stored memories. "
                    "Use the nia_memory tool to search and recall them."
                )
        except Exception:
            pass

        # Inject personality + memory BEFORE the base (which already has SOUL.md at top).
        # build_system_prompt returns SOUL.md + base; we want SOUL.md first,
        # then personality, then memory, then base rules.
        # Split base into SOUL.md part and rest.
        soul_marker = "---"
        if soul_marker in base:
            soul_part, _, rest = base.partition(soul_marker)
            return f"{soul_part.strip()}\n\n---\n\n{personality_block}\n\n{memory_block}\n\n{rest}".strip()
        return f"{personality_block}\n\n{memory_block}\n\n{base}".strip()

    def _greet(self) -> str:
        """Return NIA's greeting string."""
        greeting = self._personality.greet(self._context.time_of_day.value)
        if self._engine is not None:
            greeting += f"\n\nModel: {self._engine._model}"
        return greeting


__all__ = ["NIA"]
