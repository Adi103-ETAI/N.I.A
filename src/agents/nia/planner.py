"""N.I.A. Mission Planner — Sprint 2.

Takes raw user intent and uses an LLM to produce a structured MissionManifest
that declares required scopes, execution mode, and step-by-step plan.
Replaces the legacy DecisionCore Router.
"""
from __future__ import annotations

import json
import re
import logging
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.schema.mission import MissionManifest, PlanStep
from src.core.policy.scopes import CapabilityScope
from src.core.config.prompts import load_prompt

logger = logging.getLogger("NIA.Planner")


class MissionPlanner:
    """LLM-driven planner that turns user intent into a MissionManifest."""

    def __init__(self):
        self._llm = None
        self._planning_prompt = None

    @property
    def llm(self):
        if not self._llm:
            from src.models.manager import get_smart_model
            self._llm = get_smart_model(temperature=0.0)
        return self._llm

    @property
    def planning_prompt(self) -> str:
        """Load planning prompt from markdown file (cached)."""
        if self._planning_prompt is None:
            try:
                self._planning_prompt = load_prompt("planner")
                logger.info("✅ Loaded planner prompt from markdown")
            except FileNotFoundError:
                logger.warning("Planner prompt file not found — using fallback")
                self._planning_prompt = self._get_fallback_prompt()
        return self._planning_prompt

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if markdown file not available."""
        return """\
You are N.I.A.'s Strategic Planner. Your job is to read the user's intent and
produce a structured execution plan as a JSON object.

Output ONLY a raw JSON object matching this schema:
{
  "mission_id": "<unique short slug, no spaces>",
  "intent": "<summary of what the user wants>",
  "steps": [
    {"description": "<step description>", "assigned_role": "<planner|researcher|coder|reviewer>", "required_scopes": ["<scope>"]}
  ],
  "required_scopes": ["<all unique scopes across all steps>"],
  "estimated_depth": <int 1-3>,
  "estimated_agents": <int 1-10>,
  "execution_mode": "<fast|standard|deep>"
}

Scope values must be one of: read_only, write, execute, network, agent_spawn, destructive

Rules:
- Only include scopes that are DEFINITELY needed, not "might be needed".
- read_only: For viewing information, analysis, questions.
- write: For creating/modifying files, saving results.
- execute: For running code or scripts.
- network: For internet requests and API calls.
- agent_spawn: For delegating to sub-agents.
- destructive: For delete/remove operations.
- fast = 1-2 steps, simple tasks.
- standard = 3-5 steps, typical tasks.
- deep = 6+ steps, complex multi-agent research and code tasks.
- Do NOT include any text outside the JSON object.
"""

    async def plan(self, user_intent: str) -> MissionManifest:
        """Turn raw user intent into a structured MissionManifest.

        Args:
            user_intent: The raw user input/request string.

        Returns:
            A fully populated MissionManifest (approved=False until pre-flight).
        """
        if not user_intent.strip():
            return MissionManifest(
                mission_id="noop-001",
                intent="(empty)",
                steps=[],
                required_scopes=[CapabilityScope.READ_ONLY],
                execution_mode="fast",
            )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=self.planning_prompt),
                HumanMessage(content=user_intent),
            ])
            raw = response.content.strip()
            # Strip markdown fences if model adds them
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()

            parsed = json.loads(raw)

            # Coerce scope strings to CapabilityScope enum
            def coerce_scopes(scope_list: list) -> List[CapabilityScope]:
                result = []
                for s in scope_list:
                    try:
                        result.append(CapabilityScope(s))
                    except ValueError:
                        logger.warning(f"Unknown scope '{s}' — defaulting to execute")
                        result.append(CapabilityScope.EXECUTE)
                return result

            steps = []
            for raw_step in parsed.get("steps", []):
                steps.append(PlanStep(
                    description=raw_step.get("description", ""),
                    assigned_role=raw_step.get("assigned_role", "coder"),
                    required_scopes=coerce_scopes(raw_step.get("required_scopes", [])),
                ))

            manifest = MissionManifest(
                mission_id=parsed.get("mission_id", "mission-001"),
                intent=parsed.get("intent", user_intent),
                steps=steps,
                required_scopes=coerce_scopes(parsed.get("required_scopes", ["read_only"])),
                estimated_depth=int(parsed.get("estimated_depth", 1)),
                estimated_agents=int(parsed.get("estimated_agents", 1)),
                execution_mode=parsed.get("execution_mode", "standard"),
            )
            logger.info(f"📋 Plan ready: '{manifest.mission_id}' ({len(steps)} steps, mode={manifest.execution_mode})")
            return manifest

        except json.JSONDecodeError as e:
            logger.error(f"Planner JSON parse failed: {e}. Falling back to safe default plan.")
        except Exception as e:
            logger.error(f"Planner LLM call failed: {e}. Falling back to safe default plan.")

        # Safe fallback — treat as simple read-only chat
        return MissionManifest(
            mission_id="fallback-chat",
            intent=user_intent,
            steps=[PlanStep(
                description="Respond conversationally",
                assigned_role="planner",
                required_scopes=[CapabilityScope.READ_ONLY],
            )],
            required_scopes=[CapabilityScope.READ_ONLY],
            execution_mode="fast",
        )
