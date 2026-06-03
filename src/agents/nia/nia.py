"""N.I.A - Neural Intelligence Assistant.

Unified architecture: NIA is the soul (personality, reasoning, memory),
niaharness is the body (tools, execution, permissions, hooks, swarm).

NIA Brain decides WHAT to do.
niaharness QueryEngine handles HOW to do it (tool execution, permissions, cost tracking).
NIA Personality formats HOW it speaks.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from niaharness.config.settings import PermissionSettings, Settings
from niaharness.engine.query_engine import QueryEngine
from niaharness.hooks import HookExecutor, HookExecutionContext, HookRegistry
from niaharness.permissions import PermissionChecker
from niaharness.tools import create_default_tool_registry, register_nia_tools

from agents.nia.core.brain import NIABrain, BrainResponse
from agents.nia.core.personality import Personality, PersonalityConfig
from agents.nia.core.memory import Memory
from agents.nia.core.context import Context
from agents.nia.communication.listener import Listener
from agents.nia.communication.speaker import Speaker
from agents.nia.orchestration.state import StateManager, SystemState
from agents.nia.config import ConfigManager
from agents.nia.providers.adapter import NIAProviderAdapter
from agents.nia.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class NIA:
    """N.I.A - Neural Intelligence Assistant.

    Architecture (UNIFIED):
    ┌──────────────────────────────────────────────────────┐
    │                    N.I.A (THE SOUL)                   │
    │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
    │  │  Brain   │  │  Memory  │  │    Personality     │  │
    │  │ (reason) │  │ (file)   │  │ (JARVIS tone)      │  │
    │  └────┬─────┘  └──────────┘  └────────────────────┘  │
    │       │                                               │
    │  ┌────▼──────────────────────────────────────────┐    │
    │  │         QueryEngine (THE BODY)                │    │
    │  │  • Conversation loop                          │    │
    │  │  • Tool orchestration (38+ tools)             │    │
    │  │  • Permission checks                          │    │
    │  │  • Pre/post hooks                             │    │
    │  │  • Cost tracking                              │    │
    │  │  • File state cache                           │    │
    │  │  • Abort controller                           │    │
    │  │  • MCP integration                            │    │
    │  └───────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        working_directory: str | None = None,
        personality_config: PersonalityConfig | None = None,
    ) -> None:
        self._config_manager = ConfigManager()
        self._working_directory = working_directory or str(Path.cwd())

        # Provider system
        self._provider_registry = ProviderRegistry(self._config_manager)

        # Core NIA systems (the soul)
        self._brain = NIABrain()
        self._personality = Personality(personality_config)
        self._memory = Memory(
            storage_path=Path.home() / ".nia" / "memory.json"
        )
        self._context = Context()

        # Communication
        self._listener = Listener()
        self._speaker = Speaker()

        # Orchestration
        self._state = StateManager()

        # QueryEngine (the body) - created during initialize()
        self._engine: QueryEngine | None = None
        self._mcp_manager: Any = None

        # State
        self._initialized = False

        logger.info("N.I.A initialized (unified architecture)")

    async def initialize(self) -> str:
        """Initialize N.I.A and all subsystems."""
        self._state.system_state = SystemState.INITIALIZING

        # Initialize provider registry
        await self._provider_registry.initialize()

        # Set up brain with active provider
        active_provider = self._provider_registry.get_active_provider()
        active_model = self._provider_registry.get_active_model()
        if active_provider:
            self._brain.set_provider(active_provider, active_model)

        # Detect environment
        self._context.detect_environment(self._working_directory)

        # Load memory
        self._memory.load()

        # Build the QueryEngine (the body)
        await self._build_engine(active_provider, active_model)

        # Get greeting
        time_str = self._context.time_of_day.value
        greeting = self._personality.greet(time_str)

        # Add provider info to greeting
        if active_provider:
            provider_name = (
                active_provider.config.label
                if hasattr(active_provider, "config")
                else "Unknown"
            )
            greeting += f"\nConnected to {provider_name}/{active_model or 'default model'}"
        else:
            greeting += self._get_setup_prompt()

        self._state.system_state = SystemState.READY
        self._initialized = True

        logger.info(
            f"N.I.A ready. Provider: "
            f"{active_provider.config.name if active_provider and hasattr(active_provider, 'config') else 'none'}"
        )
        return greeting

    async def _build_engine(
        self,
        active_provider: Any,
        active_model: str | None,
    ) -> None:
        """Build the QueryEngine with NIA's merged system prompt."""
        # Create the adapter: NIA provider → SupportsStreamingMessages
        if active_provider is None:
            logger.warning("No provider available, QueryEngine will not work")
            return

        adapter = NIAProviderAdapter(active_provider, active_model)

        # MCP integration: create manager and pass to tool registry
        mcp_manager = None
        try:
            from niaharness.mcp.client import McpClientManager
            from niaharness.mcp.config import load_mcp_server_configs
            mcp_servers = load_mcp_server_configs(Settings(), [])
            if mcp_servers:
                mcp_manager = McpClientManager(mcp_servers)
                await mcp_manager.connect_all()
                logger.info(f"MCP connected: {sum(1 for s in mcp_manager.list_statuses() if s.state == 'connected')} servers")
        except Exception as e:
            logger.debug(f"MCP not available: {e}")

        # Create tool registry with all 38+ niaharness tools + MCP tools
        tool_registry = create_default_tool_registry(mcp_manager)

        # Wire NIA's memory, context, and engine into the NIA-specific tools
        register_nia_tools(tool_registry, self._memory, self._context, self._engine)

        # Build merged system prompt: niaharness base + NIA personality + context
        system_prompt = self._build_merged_system_prompt()

        # Permission checker (niaharness handles permissions now)
        permission_checker = PermissionChecker(PermissionSettings())

        # Hook executor (niaharness handles hooks now)
        hook_executor = HookExecutor(
            HookRegistry(),
            HookExecutionContext(
                cwd=Path(self._working_directory).resolve(),
                api_client=adapter,
                default_model=active_model or "unknown",
            ),
        )

        # Store MCP manager for cleanup
        self._mcp_manager = mcp_manager

        # Create the QueryEngine
        self._engine = QueryEngine(
            api_client=adapter,
            tool_registry=tool_registry,
            permission_checker=permission_checker,
            cwd=self._working_directory,
            model=active_model or "unknown",
            system_prompt=system_prompt,
            max_tokens=4096,
            hook_executor=hook_executor,
            tool_metadata={"mcp_manager": mcp_manager} if mcp_manager else None,
        )

        logger.info("QueryEngine built with NIA merged prompt + MCP")

    def _build_merged_system_prompt(self) -> str:
        """Build a merged system prompt: niaharness base + NIA personality + context.

        This gives the QueryEngine:
        - niaharness's tool instructions and safety rules
        - NIA's JARVIS personality and tone
        - NIA's context awareness
        """
        sections = []

        # NIA personality and identity (top priority)
        sections.append("# Identity\nYou are N.I.A (Neural Intelligence Assistant), "
                        "an AI partner inspired by JARVIS. You think, plan, and execute "
                        "with calm authority. You are proactive, precise, and always ready.")

        if self._personality:
            personality_desc = self._personality.get_stats()
            sections.append(f"# Personality\nTone: Professional, confident, slightly witty. "
                          f"Style: Direct and efficient. Voice: Calm authority. "
                          f"When appropriate, use dry wit — never forced.")

        # NIA's context
        context_data = self._context.get_full_context()
        if context_data:
            ctx_lines = []
            if context_data.get("time_of_day"):
                ctx_lines.append(f"Time: {context_data['time_of_day']}")
            if context_data.get("working_directory"):
                ctx_lines.append(f"Working directory: {context_data['working_directory']}")
            if context_data.get("git_branch"):
                ctx_lines.append(f"Git branch: {context_data['git_branch']}")
            if context_data.get("project_type"):
                ctx_lines.append(f"Project type: {context_data['project_type']}")
            if ctx_lines:
                sections.append("# Environment Context\n" + "\n".join(ctx_lines))

        # NIA's memory summary
        if self._memory:
            stats = self._memory.get_stats()
            if stats.get("total_conversations", 0) > 0:
                sections.append(f"# Memory\nPrevious conversations: {stats['total_conversations']}. "
                              "Use memory to maintain continuity across sessions.")

        # niaharness base system prompt (tools, safety, instructions)
        from niaharness.prompts.system_prompt import build_system_prompt
        niaharness_prompt = build_system_prompt(cwd=self._working_directory)
        sections.append(niaharness_prompt)

        # NIA's tool delegation instructions
        sections.append("# Delegation\n"
                        "You are the head — you decide WHAT needs to happen. "
                        "niaharness tools are your hands — they execute your decisions. "
                        "Use tools precisely: specify file paths, exact content, and clear commands. "
                        "For complex multi-step tasks, think through each step before acting.")

        return "\n\n".join(sections)

    def _get_setup_prompt(self) -> str:
        """Generate first-run setup prompt."""
        providers = self._provider_registry.list_providers()
        lines = ["\n"]
        lines.append("No provider configured. Let's set one up.\n")
        lines.append("Available providers:")
        lines.append("")
        for i, p in enumerate(providers, 1):
            lines.append(f"  {i}. {p.name:<20} ({p.id})")
        lines.append("")
        lines.append("Quick start:")
        lines.append("  /connect anthropic api_key=sk-ant-...")
        lines.append("  /connect openai api_key=sk-...")
        lines.append("  /connect ollama")
        lines.append("")
        lines.append("Or set environment variables:")
        lines.append("  export ANTHROPIC_API_KEY=sk-ant-...")
        lines.append("  export OPENAI_API_KEY=sk-...")
        return "\n".join(lines)

    async def process(self, user_input: str) -> str:
        """Process user input and return response.

        Unified flow:
        1. Listen (parse input)
        2. Brain pre-processes (optional: intent detection for complex tasks)
        3. QueryEngine handles conversation + tool execution (with permissions, hooks, cost)
        4. Speak (format response with personality)
        """
        if not self._initialized:
            await self.initialize()

        # Handle slash commands
        if user_input.startswith("/"):
            from agents.nia.commands import handle_command
            parts = user_input[1:].split()
            command = parts[0] if parts else "help"
            args = {}
            for i, part in enumerate(parts[1:], 1):
                if "=" in part:
                    key, value = part.split("=", 1)
                    args[key] = value
                else:
                    args[str(i)] = part
                    args["0"] = part if i == 1 else args.get("0", "")
            return await handle_command(self, command, args)

        # Track activity
        self._context.track_activity()
        self._memory.add_conversation("user", user_input)

        # 1. Listen - Parse input
        self._listener.listen(user_input)

        # 2. QueryEngine handles the conversation (tools, permissions, hooks, cost)
        if self._engine is None:
            return "Engine not initialized. Please run /connect to set up a provider."

        response_text = ""
        async for event in self._engine.submit_message(user_input):
            # Collect text deltas for the final response
            from niaharness.engine.stream_events import (
                AssistantTextDelta,
                AssistantTurnComplete,
                ToolExecutionStarted,
                ToolExecutionCompleted,
            )
            if isinstance(event, AssistantTextDelta):
                response_text += event.text
            elif isinstance(event, ToolExecutionStarted):
                logger.info(f"Tool starting: {event.tool_name}")
            elif isinstance(event, ToolExecutionCompleted):
                logger.info(f"Tool completed: {event.tool_name} (error={event.is_error})")
            elif isinstance(event, AssistantTurnComplete):
                # Final response from the engine
                if event.message and event.message.text:
                    response_text = event.message.text

        # 3. Speak - Format response with personality
        if response_text:
            spoken = self._speaker.speak(response_text)
            response_text = spoken.text

        # Store in memory
        if response_text:
            self._memory.add_conversation("assistant", response_text)

        return response_text or "No response generated."

    def switch_provider(self, provider_id: str, model: str | None = None) -> bool:
        """Switch the active LLM provider."""
        success = self._provider_registry.set_active(provider_id, model)
        if success:
            provider = self._provider_registry.get_provider(provider_id)
            if provider:
                self._brain.set_provider(provider, model)
                # Rebuild the engine with the new provider
                asyncio.create_task(self._rebuild_engine(provider, model))
        return success

    async def _rebuild_engine(self, provider: Any, model: str | None) -> None:
        """Rebuild QueryEngine after provider switch."""
        try:
            await self._build_engine(provider, model)
            logger.info("QueryEngine rebuilt after provider switch")
        except Exception as e:
            logger.error(f"Failed to rebuild engine: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get N.I.A's current status."""
        provider_info = {}
        active = self._provider_registry.get_active_provider()
        if active:
            config = getattr(active, "config", None)
            if config:
                provider_info = {
                    "id": config.name,
                    "name": config.label,
                    "model": self._provider_registry.get_active_model(),
                    "configured": True,
                }

        engine_info = {}
        if self._engine:
            engine_info = {
                "session_id": self._engine.session_id,
                "total_cost_usd": self._engine.total_cost_usd,
                "messages": len(self._engine.messages),
                "permission_denials": len(self._engine.permission_denials),
            }

        return {
            "state": self._state.system_state.value,
            "provider": provider_info,
            "brain": self._brain.get_stats(),
            "personality": self._personality.get_stats(),
            "memory": self._memory.get_stats(),
            "context": self._context.get_summary(),
            "engine": engine_info,
        }

    def shutdown(self) -> None:
        """Shutdown N.I.A gracefully."""
        self._state.system_state = SystemState.SHUTDOWN
        self._memory.save()
        if self._engine:
            self._engine.interrupt()
        if self._mcp_manager:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._mcp_manager.close())
            except RuntimeError:
                pass
        logger.info("N.I.A shutdown complete")
