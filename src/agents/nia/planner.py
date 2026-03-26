"""N.I.A. Mission Planner — Sprint 2.

Takes raw user intent and uses an LLM to produce a structured MissionManifest
that declares required scopes, execution mode, and step-by-step plan.
Replaces the legacy DecisionCore Router.
"""
from __future__ import annotations

import json
import re
import logging
from typing import List, Mapping, Union, Any

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

    def _extract_user_intent(self, payload: Union[str, Mapping[str, Any]]) -> tuple[str, bool]:
        """Normalize planner input and detect legacy state-dict mode."""
        if isinstance(payload, str):
            return payload, False

        if isinstance(payload, Mapping):
            # Backward compatibility: tests/older callers pass full AgentState dict.
            user_input = payload.get("user_input")
            if isinstance(user_input, str) and user_input.strip():
                return user_input, True

            messages = payload.get("messages", [])
            for msg in reversed(messages if isinstance(messages, list) else []):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    return content, True
            return "", True

        return str(payload or ""), False

    def _coerce_scopes(self, scope_list: list) -> List[CapabilityScope]:
        """Coerce mixed legacy/new scope strings to CapabilityScope values."""
        aliases = {
            "single_turn": CapabilityScope.READ_ONLY,
            "conversation": CapabilityScope.READ_ONLY,
            "tool_execution": CapabilityScope.EXECUTE,
        }
        result: List[CapabilityScope] = []
        for scope in scope_list:
            if isinstance(scope, CapabilityScope):
                result.append(scope)
                continue
            try:
                result.append(CapabilityScope(str(scope)))
                continue
            except ValueError:
                alias = aliases.get(str(scope))
                if alias is not None:
                    result.append(alias)
                else:
                    logger.warning("Unknown scope '%s' — defaulting to execute", scope)
                    result.append(CapabilityScope.EXECUTE)
        return result

    def _infer_mission_type(self, manifest: MissionManifest, hint: str | None = None) -> str:
        """Infer legacy mission_type for compatibility consumers."""
        if hint:
            return hint
        if (
            manifest.execution_mode == "deep"
            or len(manifest.steps) > 1
            or CapabilityScope.AGENT_SPAWN in manifest.required_scopes
        ):
            return "agent_spawn"
        if any(
            s in manifest.required_scopes
            for s in (
                CapabilityScope.EXECUTE,
                CapabilityScope.WRITE,
                CapabilityScope.NETWORK,
                CapabilityScope.DESTRUCTIVE,
            )
        ):
            return "tool_execution"
        return "conversation"

    def _to_legacy_manifest_dict(
        self,
        manifest: MissionManifest,
        mission_type_hint: str | None = None,
    ) -> dict:
        """Return legacy dict schema used by pre-Phase-2 tests/callers."""
        mission_type = self._infer_mission_type(manifest, mission_type_hint)
        if CapabilityScope.AGENT_SPAWN in manifest.required_scopes:
            legacy_scope = "agent_spawn"
        elif CapabilityScope.READ_ONLY in manifest.required_scopes and len(manifest.required_scopes) == 1:
            legacy_scope = "single_turn"
        else:
            legacy_scope = "tool_execution"

        legacy_steps = [
            {
                "step_id": f"step_{idx}",
                "role": step.assigned_role,
                "instruction": step.description,
                "dependencies": [],
            }
            for idx, step in enumerate(manifest.steps, start=1)
        ]

        return {
            "mission_type": mission_type,
            "scope": legacy_scope,
            "execution_mode": manifest.execution_mode,
            "steps": legacy_steps,
            "required_scopes": [s.value for s in manifest.required_scopes],
        }

    async def plan(self, user_intent: Union[str, Mapping[str, Any]]) -> Union[MissionManifest, dict]:
        """Turn raw user intent into a structured MissionManifest.

        Args:
            user_intent: The raw user input/request string.

        Returns:
            A fully populated MissionManifest (approved=False until pre-flight).
        """
        normalized_intent, legacy_mode = self._extract_user_intent(user_intent)

        if not normalized_intent.strip():
            manifest = MissionManifest(
                mission_id="noop-001",
                intent="(empty)",
                steps=[],
                required_scopes=[CapabilityScope.READ_ONLY],
                execution_mode="fast",  # type: ignore[arg-type]
            )
            return self._to_legacy_manifest_dict(manifest, mission_type_hint="conversation") if legacy_mode else manifest

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=self.planning_prompt),
                HumanMessage(content=normalized_intent),
            ])
            raw = response.content.strip()
            # Strip markdown fences if model adds them
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()

            parsed = json.loads(raw)

            mission_type = parsed.get("mission_type")
            mode = parsed.get("execution_mode", "standard")
            if mode == "quick":
                mode = "fast"

            steps = []
            for raw_step in parsed.get("steps", []):
                description = raw_step.get("description") or raw_step.get("instruction", "")
                assigned_role = raw_step.get("assigned_role") or raw_step.get("role", "coder")
                steps.append(PlanStep(
                    description=description,
                    assigned_role=assigned_role,
                    required_scopes=self._coerce_scopes(raw_step.get("required_scopes", [])),
                ))

            raw_required_scopes = parsed.get("required_scopes")
            if not raw_required_scopes:
                # Legacy schema: single "scope" key.
                raw_scope = parsed.get("scope")
                raw_required_scopes = [raw_scope] if raw_scope else ["read_only"]

            manifest = MissionManifest(
                mission_id=parsed.get("mission_id", f"{(mission_type or 'mission').replace('_', '-')}-001"),
                intent=parsed.get("intent", normalized_intent),
                steps=steps,
                required_scopes=self._coerce_scopes(raw_required_scopes),
                estimated_depth=int(parsed.get("estimated_depth", 1)),
                estimated_agents=int(parsed.get("estimated_agents", 1)),
                execution_mode=mode,
            )
            logger.info(f"📋 Plan ready: '{manifest.mission_id}' ({len(steps)} steps, mode={manifest.execution_mode})")
            return self._to_legacy_manifest_dict(manifest, mission_type_hint=mission_type) if legacy_mode else manifest

        except json.JSONDecodeError as e:
            logger.error(f"Planner JSON parse failed: {e}. Falling back to safe default plan.")
        except Exception as e:
            logger.error(f"Planner LLM call failed: {e}. Falling back to safe default plan.")

        # Safe fallback — treat as simple read-only chat
        fallback_manifest = MissionManifest(
            mission_id="fallback-chat",
            intent=normalized_intent,
            steps=[PlanStep(
                description="Respond conversationally",
                assigned_role="planner",
                required_scopes=[CapabilityScope.READ_ONLY],
            )],
            required_scopes=[CapabilityScope.READ_ONLY],
            execution_mode="fast",  # type: ignore[arg-type]
        )
        return self._to_legacy_manifest_dict(fallback_manifest, mission_type_hint="conversation") if legacy_mode else fallback_manifest
