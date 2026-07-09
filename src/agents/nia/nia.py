"""N.I.A — Neural Intelligence Assistant.

NIA is the agent. niaharness is its runtime.

NIA owns: identity (SOUL.md), memory, personality, proactive review.
niaharness owns: tool execution (57 tools), permissions, hooks, MCP, cost tracking.

There is ONE LLM call per turn — the QueryEngine's. NIA does not make a
separate "thinking" LLM call. The LLM IS the brain. This mirrors the reference project's
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

        # Profile-aware paths: memory, SOUL.md, config, skills, credentials
        # all respect the active profile (~/.nia/ for default, ~/.nia/profiles/<name>/ for named).
        from niaharness.profiles import get_profile_home
        profile_home = get_profile_home()
        self._memory = Memory(storage_path=profile_home / "memory.json")
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
        # The system prompt is built fresh each time _build_system_prompt is
        # called, so background_review memory writes are reflected on the next
        # turn. We store the builder and rebuild the prompt before each turn.
        self._engine = QueryEngine(
            api_client=resolved_api_client,
            tool_registry=tool_registry,
            permission_checker=PermissionChecker(settings.permission),
            cwd=self._working_directory,
            model=resolved_model,
            system_prompt=self._build_system_prompt(),
            max_tokens=settings.max_tokens,
            max_turns=settings.max_turns,
            max_budget_usd=settings.max_budget_usd,
            token_budget=settings.token_budget,
            hook_executor=self._hook_executor,
            tool_metadata={
                "mcp_manager": self._mcp_manager,
                "api_client": resolved_api_client,
                "model": resolved_model,
                "max_tokens": settings.max_tokens,
            },
            memory=self._memory,  # enables background review (proactive layer)
        )

        # Wire NIA's memory + context into the nia_memory/nia_context tools.
        # P0 fix: this must be called AFTER self._engine is created, so
        # nia_session tool's set_engine() receives the real engine (not None).
        register_nia_tools(tool_registry, self._memory, self._context, self._engine)

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

        Before each turn, the system prompt is refreshed so that memory
        writes from background_review are visible to the model.

        Args:
            message: The user's message.

        Yields:
            StreamEvent: Assistant text deltas, tool events, turn-complete.
        """
        if not self._initialized or self._engine is None:
            raise RuntimeError("NIA not initialized. Call await nia.initialize() first.")

        # Refresh the system prompt so background_review memory writes
        # are reflected. The prompt includes SOUL.md + personality +
        # memory summary — if memory changed, the prompt changes.
        self._engine.set_system_prompt(self._build_system_prompt())

        async for event in self._engine.submit_message(message):
            yield event

    async def process_gateway_message(
        self,
        *,
        platform: str,
        chat_id: str,
        user_id: str,
        text: str,
    ) -> str:
        """Process a message from a chat platform (Telegram, Discord, etc.).

        Each chat gets its own QueryEngine with a fresh message list,
        so conversations from different chats don't bleed into each other.
        The engines share the same api_client, tool_registry, and
        permission_checker (cost-efficient), but have isolated histories.

        Args:
            platform: The platform name (e.g. "telegram").
            chat_id: The chat/channel ID on the platform.
            user_id: The user ID on the platform.
            text: The message text.

        Returns:
            The assistant's response text.
        """
        session_key = f"{platform}:{chat_id}"

        if not hasattr(self, "_gateway_sessions"):
            self._gateway_sessions: dict[str, QueryEngine] = {}

        # Create a truly isolated QueryEngine per chat session.
        # Each engine has its own message list but shares the api_client,
        # tool_registry, permission_checker, and hook_executor.
        if session_key not in self._gateway_sessions and self._engine is not None:
            from niaharness.engine.query_engine import QueryEngine as _QE

            isolated_engine = _QE(
                api_client=self._engine._api_client,
                tool_registry=self._engine._tool_registry,
                permission_checker=self._engine._permission_checker,
                cwd=self._engine._cwd,
                model=self._engine._model,
                system_prompt=self._engine._system_prompt,
                max_tokens=self._engine._max_tokens,
                max_turns=self._engine._max_turns,
                max_budget_usd=self._engine._max_budget_usd,
                token_budget=self._engine._token_budget,
                hook_executor=self._engine._hook_executor,
                tool_metadata=self._engine._tool_metadata,
                memory=self._engine._memory,
            )
            self._gateway_sessions[session_key] = isolated_engine

        engine = self._gateway_sessions.get(session_key, self._engine)
        if engine is None:
            return "Error: NIA not initialized."

        # Collect the response text from the stream.
        from niaharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete

        response_text = ""
        async for event in engine.submit_message(text):
            if isinstance(event, AssistantTextDelta):
                response_text += event.text

        return response_text.strip()

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
        """Build the API client (Anthropic or OpenAI-compatible).

        Resolution order:
        1. If an explicit API key is provided, use it.
        2. If no API key but OAuth tokens exist (from /oauth login), use
           the OAuth access token as the API key.
        3. If neither, fall back to settings.resolve_api_key().

        Wraps the AnthropicApiClient in a FailoverAnthropicClient so
        401/402/403/429 errors trigger credential-pool rotation.
        """
        resolved_key = api_key or settings.resolve_api_key()
        resolved_base = base_url or settings.base_url

        if getattr(settings, "api_format", "anthropic") == "openai":
            from niaharness.api.openai_client import OpenAICompatibleClient

            return OpenAICompatibleClient(api_key=resolved_key, base_url=resolved_base)

        # If no API key found, try OAuth tokens (from /oauth login).
        if not resolved_key:
            try:
                from niaharness.providers.anthropic import OAuthTokenManager

                oauth_mgr = OAuthTokenManager()
                oauth_token = oauth_mgr.get_valid_token()
                if oauth_token:
                    logger.info("Using Anthropic OAuth token for API access")
                    resolved_key = oauth_token
            except Exception:
                pass

        # Use the failover client (wraps AnthropicApiClient with credential rotation).
        # Falls back to bare AnthropicApiClient if the failover module is unavailable.
        try:
            from niaharness.api.failover_client import create_failover_client

            return create_failover_client(
                "anthropic",
                base_url=resolved_base,
            )
        except Exception as exc:
            logger.debug("Failover client unavailable, using bare client: %s", exc)
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

        P0 fix: the old implementation used `base.partition("---")` to
        split SOUL.md from the base prompt, but SOUL.md itself contains
        `---` separators (horizontal rules in Markdown), so the split
        fired at the wrong location and corrupted the prompt layout.
        Now we load SOUL.md directly and build the prompt in the correct
        order without string surgery.
        """
        from niaharness.prompts.soul import load_soul_md
        from niaharness.prompts.system_prompt import build_system_prompt
        from niaharness.prompts.environment import get_environment_info

        # Load SOUL.md directly (slot 1).
        parts: list[str] = []
        try:
            soul = load_soul_md()
            if soul:
                parts.append(soul)
        except Exception:
            pass

        # Personality block (slot 2 — Jarvis tone).
        parts.append(
            "# Personality\n"
            "Tone: Professional, confident, slightly witty. "
            "Style: Direct and efficient. Voice: Calm authority.\n"
            "When appropriate, use dry wit — never forced."
        )

        # Memory block (slot 3 — continuity).
        # P2 fix: unify the two memory systems. We now inject the actual
        # memory content (preferences, facts, patterns) into the prompt,
        # not just a count. The project-scoped markdown memory files from
        # niaharness/memory/ are loaded separately by the context builder.
        try:
            stats = self._memory.get_stats()
            if stats.get("total_memories", 0) > 0:
                # Inject the actual memory summary (not just a count).
                memory_summary = self._memory.get_summary_for_prompt()
                if memory_summary:
                    parts.append(memory_summary)
                else:
                    parts.append(
                        f"# Memory\nYou have {stats['total_memories']} stored memories. "
                        "Use the nia_memory tool to search and recall them."
                    )
        except Exception:
            pass

        # niaharness base (slot 4 — tool instructions, safety rules, environment).
        # build_system_prompt with include_soul=False gives us just the base
        # rules + environment, without SOUL.md (which we already loaded above).
        base = build_system_prompt(cwd=self._working_directory, include_soul=False)
        parts.append(base)

        return "\n\n".join(parts).strip()

    def _greet(self) -> str:
        """Return NIA's greeting string."""
        greeting = self._personality.greet(self._context.time_of_day.value)
        if self._engine is not None:
            greeting += f"\n\nModel: {self._engine._model}"
        return greeting


__all__ = ["NIA"]
