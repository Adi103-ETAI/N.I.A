"""N.I.A Brain - LLM-powered decision making.

This is the head that thinks, reasons, and decides what to do.
Unlike the previous hardcoded version, this brain uses an actual LLM
to understand intent, reason about approaches, and formulate decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import LLMRequest

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts"


@dataclass
class Task:
    """A task to delegate to OpenHarness."""
    description: str
    tool: str  # OpenHarness tool name
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainResponse:
    """Structured response from the LLM brain."""
    thinking: str
    intent: str
    tasks: list[Task]
    response: str
    confidence: float = 0.9
    needs_clarification: bool = False
    clarification_question: str | None = None
    # For ReAct mode
    plan: Any | None = None  # ReActPlan if using multi-step reasoning
    use_react: bool = False  # Whether to use ReAct loop

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BrainResponse:
        """Parse LLM JSON response into BrainResponse."""
        tasks = []
        for t in data.get("tasks", []):
            tasks.append(Task(
                description=t.get("description", ""),
                tool=t.get("tool", "bash"),
                args=t.get("args", {}),
            ))

        return cls(
            thinking=data.get("thinking", ""),
            intent=data.get("intent", "conversation"),
            tasks=tasks,
            response=data.get("response", "Processing..."),
            confidence=float(data.get("confidence", 0.9)),
            needs_clarification=data.get("needs_clarification", False),
            clarification_question=data.get("clarification_question"),
            use_react=data.get("use_react", False),
        )


class NIABrain:
    """LLM-powered brain of N.I.A.

    Uses an actual language model to:
    - Understand user intent (not regex)
    - Reason about the best approach
    - Decide what tasks to delegate
    - Formulate natural responses

    The brain reads prompt files from prompts/ directory
    to maintain its personality and behavior.
    """

    def __init__(self, provider: LLMProvider | None = None, model: str | None = None) -> None:
        self._provider = provider
        self._model = model
        self._system_prompt: str = ""
        self._personality_prompt: str = ""
        self._delegation_prompt: str = ""
        self._conversation_history: list[dict[str, str]] = []
        self._decision_count: int = 0

        # Load prompts
        self._load_prompts()

        logger.info("N.I.A Brain initialized (LLM-powered)")

    def set_provider(self, provider: LLMProvider, model: str | None = None) -> None:
        """Set or change the LLM provider."""
        self._provider = provider
        if model:
            self._model = model
        provider_name = getattr(provider, 'config', None)
        name = provider_name.name if provider_name else getattr(provider, 'id', 'unknown')
        logger.info(f"Brain provider set to: {name}/{model or 'default'}")

    def _load_prompts(self) -> None:
        """Load prompt files from prompts/ directory."""
        system_path = PROMPTS_DIR / "system.md"
        personality_path = PROMPTS_DIR / "personality.md"
        delegation_path = PROMPTS_DIR / "delegation.md"

        if system_path.exists():
            self._system_prompt = system_path.read_text(encoding="utf-8")
        if personality_path.exists():
            self._personality_prompt = personality_path.read_text(encoding="utf-8")
        if delegation_path.exists():
            self._delegation_prompt = delegation_path.read_text(encoding="utf-8")

        logger.info(f"Loaded prompts: system={bool(self._system_prompt)}, personality={bool(self._personality_prompt)}")

    async def think(self, user_input: str, context: dict[str, Any] | None = None) -> BrainResponse:
        """Think about user input using the LLM.

        This is the core of N.I.A's intelligence.
        Instead of regex matching, we ask the LLM to understand and decide.
        """
        if self._provider is None:
            return self._fallback_think(user_input)

        # Build context string
        context_str = self._build_context_string(context)

        # Build messages for the LLM
        messages = self._build_messages(user_input, context_str)

        # Create the request
        # Use specified model, or first available model from provider
        model = self._model
        if not model and self._provider:
            models = self._provider.list_models()
            if models:
                model = models[0].id

        request = LLMRequest(
            model=model or self._provider.id if self._provider else "unknown",
            messages=messages,
            system=self._system_prompt,
            max_tokens=2048,
            temperature=0.3,  # Low temp for consistent decisions
        )

        try:
            # Call the LLM
            response = await self._provider.complete(request)

            # Parse the JSON response
            brain_response = self._parse_response(response.content)

            # Track in conversation history
            self._conversation_history.append({"role": "user", "content": user_input})
            self._conversation_history.append({"role": "assistant", "content": response.content})

            # Trim history if too long
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            self._decision_count += 1
            logger.info(f"Brain decision #{self._decision_count}: intent={brain_response.intent}, tasks={len(brain_response.tasks)}")

            return brain_response

        except Exception as e:
            logger.error(f"Brain LLM call failed: {e}")
            return self._fallback_think(user_input)

    def _build_context_string(self, context: dict[str, Any] | None) -> str:
        """Build a context string for the LLM."""
        if not context:
            return ""

        parts = []
        if context.get("time_of_day"):
            parts.append(f"Time: {context['time_of_day']}")
        if context.get("working_directory"):
            parts.append(f"Working directory: {context['working_directory']}")
        if context.get("git_branch"):
            parts.append(f"Git branch: {context['git_branch']}")
        if context.get("project_type"):
            parts.append(f"Project type: {context['project_type']}")
        if context.get("recent_files"):
            parts.append(f"Recent files: {', '.join(context['recent_files'][:5])}")

        return "\n".join(parts) if parts else ""

    def _build_messages(self, user_input: str, context_str: str) -> list[dict[str, str]]:
        """Build message list for the LLM."""
        messages = []

        # Add personality and delegation context as system guidance
        if self._personality_prompt:
            messages.append({"role": "user", "content": f"[Personality Guidelines]\n{self._personality_prompt[:500]}"})
            messages.append({"role": "assistant", "content": "Understood. I'll maintain this personality."})

        if self._delegation_prompt:
            messages.append({"role": "user", "content": f"[Delegation Guide]\n{self._delegation_prompt[:500]}"})
            messages.append({"role": "assistant", "content": "Understood. I'll follow these delegation patterns."})

        # Add conversation history
        messages.extend(self._conversation_history[-10:])  # Last 10 turns

        # Add current input with context
        user_content = user_input
        if context_str:
            user_content = f"[Context]\n{context_str}\n\n[User Input]\n{user_input}"

        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_response(self, content: str) -> BrainResponse:
        """Parse LLM response into structured BrainResponse."""
        # Try to extract JSON from the response
        # The LLM might wrap it in markdown code blocks
        json_str = content.strip()

        # Remove markdown code block markers if present
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        json_str = json_str.strip()

        try:
            data = json.loads(json_str)
            return BrainResponse.from_json(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            # Try to find JSON in the response
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                data = json.loads(content[start:end])
                return BrainResponse.from_json(data)
            except (ValueError, json.JSONDecodeError):
                # Fallback: treat entire response as a conversational response
                return BrainResponse(
                    thinking="LLM response was not valid JSON, treating as conversational",
                    intent="conversation",
                    tasks=[],
                    response=content,
                    confidence=0.5,
                )

    def _fallback_think(self, user_input: str) -> BrainResponse:
        """Fallback thinking when LLM is unavailable."""
        return BrainResponse(
            thinking="LLM unavailable, using fallback logic",
            intent="conversation",
            tasks=[],
            response="I'm currently unable to process that request. The LLM provider may not be configured. Please use /connect to set up a provider.",
            confidence=0.3,
            needs_clarification=True,
            clarification_question="Would you like me to help you configure an LLM provider?",
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return brain statistics."""
        provider_name = "none"
        if self._provider:
            config = getattr(self._provider, 'config', None)
            provider_name = config.name if config else getattr(self._provider, 'id', 'unknown')

        return {
            "total_decisions": self._decision_count,
            "history_length": len(self._conversation_history),
            "provider": provider_name,
            "model": self._model or "default",
            "prompts_loaded": bool(self._system_prompt),
        }

    async def think_for_react(self, prompt: str) -> str:
        """Think about a prompt and return the response as a string.

        Used by the ReAct loop for planning, reflecting, and adjusting.
        """
        if self._provider is None:
            return "LLM unavailable"

        model = self._model
        if not model and self._provider:
            models = self._provider.list_models()
            if models:
                model = models[0].id

        messages = [{"role": "user", "content": prompt}]

        request = LLMRequest(
            model=model or self._provider.id if self._provider else "unknown",
            messages=messages,
            system="You are N.I.A's reasoning engine. Think step by step and provide clear, actionable responses.",
            max_tokens=1024,
            temperature=0.3,
        )

        try:
            response = await self._provider.complete(request)
            return response.content
        except Exception as e:
            logger.error(f"Brain think_for_react failed: {e}")
            return f"Error: {e}"
