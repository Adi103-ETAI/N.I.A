"""N.I.A - Neural Intelligence Assistant.

The main integration module that ties all components together.
N.I.A is the HEAD, OpenHarness is the HANDS.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from agents.nia.core.brain import NIABrain, BrainResponse
from agents.nia.core.personality import Personality, PersonalityConfig
from agents.nia.core.memory import Memory
from agents.nia.core.context import Context
from agents.nia.core.react import ReActLoop
from agents.nia.communication.listener import Listener
from agents.nia.communication.speaker import Speaker
from agents.nia.orchestration.dispatcher import Dispatcher
from agents.nia.orchestration.coordinator import Coordinator
from agents.nia.orchestration.state import StateManager, SystemState
from agents.nia.orchestration.bridge import HarnessExecutorBridge
from agents.nia.config import ConfigManager
from agents.nia.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class NIA:
    """N.I.A - Neural Intelligence Assistant.

    The head that:
    - Listens to user input
    - Thinks about what to do (via LLM)
    - Decides the best approach
    - Delegates execution to OpenHarness (the hands)
    - Speaks the response

    Architecture:
    ┌─────────────────────────────────────────┐
    │                N.I.A (HEAD)              │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
    │  │ Brain   │  │ Memory  │  │ Context │ │
    │  │ (LLM)  │  │         │  │         │ │
    │  └────┬────┘  └────┬────┘  └────┬────┘ │
    │       │            │            │       │
    │  ┌────┴────────────┴────────────┴────┐  │
    │  │          Personality              │  │
    │  └───────────────────────────────────┘  │
    └─────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Provider Registry               │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
    │  │Anthropic│  │ OpenAI  │  │ Ollama  │ │
    │  └─────────┘  └─────────┘  └─────────┘ │
    └─────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │          OpenHarness (HANDS)            │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
    │  │ Tools   │  │ Engine  │  │   UI    │ │
    │  └─────────┘  └─────────┘  └─────────┘ │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        working_directory: str | None = None,
        personality_config: PersonalityConfig | None = None,
    ) -> None:
        # Configuration
        self._config_manager = ConfigManager()
        self._working_directory = working_directory or str(Path.cwd())

        # Provider system
        self._provider_registry = ProviderRegistry(self._config_manager)

        # Core systems
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
        self._dispatcher = Dispatcher()
        self._coordinator = Coordinator()
        self._state = StateManager()

        # Execution bridge (connects Head to Hands)
        self._bridge = HarnessExecutorBridge(self._working_directory)
        self._dispatcher.set_tool_executor(self._bridge.execute_tool_call)

        # State
        self._initialized = False

        logger.info("N.I.A initialized")

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

        # Get greeting
        time_str = self._context.time_of_day.value
        greeting = self._personality.greet(time_str)

        # Add provider info to greeting
        if active_provider:
            provider_name = active_provider.config.label if hasattr(active_provider, 'config') else 'Unknown'
            greeting += f"\nConnected to {provider_name}/{active_model or 'default model'}"
        else:
            greeting += self._get_setup_prompt()

        self._state.system_state = SystemState.READY
        self._initialized = True

        logger.info(f"N.I.A ready. Provider: {active_provider.config.name if active_provider and hasattr(active_provider, 'config') else 'none'}")
        return greeting

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

        This is the main loop:
        1. Listen (parse input)
        2. Think (LLM brain processes)
        3. Decide (what to do - simple or ReAct)
        4. Act (delegate to OpenHarness)
        5. Speak (respond to user)
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
        parsed = self._listener.listen(user_input)

        # 2. Think - Brain processes (via LLM)
        context_data = self._context.get_full_context()
        brain_response = await self._brain.think(user_input, context_data)

        # 3. Check if clarification is needed
        if brain_response.needs_clarification:
            return brain_response.clarification_question or brain_response.response

        # 4. Decide: Simple (one-shot) or ReAct (multi-step)
        if brain_response.use_react and brain_response.tasks:
            # Use ReAct loop for complex multi-step tasks
            response = await self._execute_with_react(user_input, brain_response, context_data)
        elif brain_response.tasks:
            # Simple one-shot execution
            response = await self._execute_simple(brain_response)
        else:
            # No tasks, just conversation
            response = brain_response.response

        # 5. Speak - Format response
        spoken = self._speaker.speak(response)

        # Store in memory
        self._memory.add_conversation("assistant", spoken.text)

        return spoken.text

    async def _execute_simple(self, brain_response: BrainResponse) -> str:
        """Execute tasks in simple one-shot mode."""
        response = brain_response.response
        self._state.start_operation(f"Executing {len(brain_response.tasks)} tasks")

        # Dispatch tasks to OpenHarness
        tool_calls = []
        for task in brain_response.tasks:
            tool_calls.append((task.description, task.tool, task.args))

        if tool_calls:
            tasks = self._dispatcher.dispatch_batch(tool_calls)
            # Execute the tasks
            result = await self._dispatcher.execute_pending()
            if result.tasks_succeeded > 0:
                response += f"\n\nExecuted {result.tasks_succeeded} task(s) successfully."
            if result.tasks_failed > 0:
                response += f"\n\n{result.tasks_failed} task(s) failed."
                for error in result.errors[:3]:  # Show first 3 errors
                    response += f"\n  - {error}"

        self._state.complete_operation("task_execution")
        return response

    async def _execute_with_react(
        self,
        user_input: str,
        brain_response: BrainResponse,
        context_data: dict[str, Any],
    ) -> str:
        """Execute using the ReAct loop for complex multi-step tasks."""
        self._state.start_operation("ReAct loop")

        # Create the ReAct loop
        react = ReActLoop(
            think_fn=self._brain.think_for_react,
            execute_fn=self._bridge.execute_tool,
            max_steps=10,
        )

        # Run the ReAct loop
        final_result = ""
        async for event in react.run(user_input, context_data):
            event_type = event.get("type")

            if event_type == "plan":
                plan = event["plan"]
                final_result = f"Plan: {plan.goal}\n\n"
                for step in plan.steps:
                    final_result += f"  Step {step.step_number}: {step.thought}\n"

            elif event_type == "step_complete":
                step = event["step"]
                status = "✓" if step.status.value == "completed" else "✗"
                final_result += f"\n{status} Step {step.step_number}: {step.action[:50]}"
                if step.result:
                    # Truncate long results
                    result_preview = step.result[:100] + "..." if len(step.result) > 100 else step.result
                    final_result += f"\n  Result: {result_preview}"

            elif event_type == "reflect":
                reflection = event["reflection"]
                if reflection:
                    final_result += f"\n  Reflection: {reflection[:100]}"

            elif event_type == "complete":
                result = event["result"]
                final_result = result

        self._state.complete_operation("react_loop")
        return final_result

    def switch_provider(self, provider_id: str, model: str | None = None) -> bool:
        """Switch the active LLM provider."""
        success = self._provider_registry.set_active(provider_id, model)
        if success:
            provider = self._provider_registry.get_provider(provider_id)
            if provider:
                self._brain.set_provider(provider, model)
        return success

    def get_status(self) -> dict[str, Any]:
        """Get N.I.A's current status."""
        provider_info = {}
        active = self._provider_registry.get_active_provider()
        if active:
            config = getattr(active, 'config', None)
            if config:
                provider_info = {
                    "id": config.name,
                    "name": config.label,
                    "model": self._provider_registry.get_active_model(),
                    "configured": True,
                }

        return {
            "state": self._state.system_state.value,
            "provider": provider_info,
            "brain": self._brain.get_stats(),
            "personality": self._personality.get_stats(),
            "memory": self._memory.get_stats(),
            "context": self._context.get_summary(),
            "dispatcher": self._dispatcher.get_status(),
        }

    def shutdown(self) -> None:
        """Shutdown N.I.A gracefully."""
        self._state.system_state = SystemState.SHUTDOWN
        self._memory.save()
        logger.info("N.I.A shutdown complete")
