"""N.I.A — Neural Intelligence Assistant (unified orchestrator).

NIA is the agent. niaharness is its runtime. NIA owns identity (SOUL.md),
memory, personality, and context — then hands each turn to niaharness's
QueryEngine for execution.

This class was moved from ``agents/nia/nia.py`` to unify the codebase
under ``niaharness/``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from niaharness.identity.memory import Memory
from niaharness.identity.context import Context
from niaharness.identity.personality import Personality, PersonalityConfig

logger = logging.getLogger(__name__)


class NIA:
    """N.I.A — the agent. niaharness is its runtime.

    Use::

        nia = NIA(working_directory="/path/to/project")
        await nia.initialize(api_key="sk-...", model="claude-sonnet-4-6")
        async for event in nia.chat("Read main.py and summarize it"):
            print(event)
        await nia.shutdown()
    """

    def __init__(
        self,
        working_directory: str | None = None,
        personality_config: PersonalityConfig | None = None,
    ) -> None:
        self._working_directory = working_directory or str(Path.cwd())
        self._personality = Personality(personality_config)

        from niaharness.profiles import get_profile_home
        profile_home = get_profile_home()
        self._memory = Memory(storage_path=profile_home / "memory.json")
        self._context = Context()
        self._engine: Any = None
        self._initialized: bool = False

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
    def engine(self) -> Any:
        return self._engine

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _build_system_prompt(self) -> str:
        """Build the NIA system prompt with personality + identity."""
        from niaharness.prompts.soul import load_soul_md

        parts: list[str] = []

        # Load SOUL.md identity.
        soul = load_soul_md()
        if soul:
            parts.append(soul)
        else:
            parts.append(
                "# N.I.A — Neural Intelligence Assistant\n\n"
                "You are NIA, a helpful AI assistant with a calm, professional "
                "demeanor inspired by J.A.R.V.I.S.\n"
            )

        # Add personality guidance.
        parts.append(f"\n## Personality: {self._personality.name}")
        parts.append(f"Base tone: {self._personality._config.base_tone}")
        parts.append(f"Mood: {self._personality.mood.value}")

        # Add memory context.
        memory_summary = self._memory.get_summary_for_prompt()
        if memory_summary:
            parts.append(memory_summary)

        # Add environment context.
        env = self._context.detect_environment(self._working_directory)
        parts.append(f"\n## Environment")
        parts.append(f"- Working directory: {env.working_directory}")
        if env.git_branch:
            parts.append(f"- Git branch: {env.git_branch}")
        if env.project_type:
            parts.append(f"- Project type: {env.project_type}")

        return "\n".join(parts)

    def get_status(self) -> dict[str, Any]:
        """Return current NIA status."""
        return {
            "state": "initialized" if self._initialized else "uninitialized",
            "cwd": self._working_directory,
            "memory": self._memory.get_stats(),
            "tools": len(self._engine._tool_registry.list_tools()) if self._engine else 0,
            "personality": self._personality.get_stats(),
            "context": self._context.get_summary(),
        }

    async def initialize(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        **kwargs: Any,
    ) -> None:
        """Initialize NIA with API credentials and model."""
        from niaharness.api.client import AnthropicApiClient
        from niaharness.config.settings import PermissionSettings
        from niaharness.engine.query_engine import QueryEngine
        from niaharness.permissions.checker import PermissionChecker
        from niaharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        checker = PermissionChecker(PermissionSettings())
        api_client = AnthropicApiClient(api_key=api_key)

        self._engine = QueryEngine(
            api_client=api_client,
            tool_registry=registry,
            permission_checker=checker,
            cwd=self._working_directory,
            model=model,
            system_prompt=self._build_system_prompt(),
            max_tokens=kwargs.get("max_tokens", 4096),
            memory=self._memory,
        )
        self._initialized = True
        logger.info("NIA initialized with model %s", model)

    async def chat(self, message: str) -> AsyncIterator[Any]:
        """Send a message and yield stream events."""
        if not self._initialized or self._engine is None:
            raise RuntimeError("NIA is not initialized. Call initialize() first.")

        self._context.track_activity()
        self._memory.add_conversation("user", message)

        async for event in self._engine.submit_message(message):
            yield event

        # Store response in memory.
        from niaharness.engine.stream_events import AssistantTurnComplete
        # (The last event should be AssistantTurnComplete with the response)
        self._memory.add_conversation("assistant", "(response)")

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._engine = None
        logger.info("NIA shut down")
