"""N.I.A ReAct Loop - Plan → Act → Reflect cycles.

This module implements structured reasoning with self-correction.
Instead of one-shot execution, N.I.A can:
1. Plan multiple steps
2. Execute one step at a time
3. Reflect on results
4. Adjust plan if needed
5. Continue until done
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a reasoning step."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningStep:
    """A single step in the ReAct loop."""
    step_number: int
    thought: str  # What the agent is thinking
    action: str  # What action to take
    tool: str | None = None  # Tool to use
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    reflection: str | None = None  # What the agent learned


@dataclass
class ReActPlan:
    """A complete plan with multiple steps."""
    goal: str  # What the user wants to achieve
    steps: list[ReasoningStep] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    confidence: float = 0.9

    @property
    def is_complete(self) -> bool:
        """Check if all steps are done."""
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)

    @property
    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    @property
    def completed_steps(self) -> int:
        """Count completed steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)


class ReActLoop:
    """Implements the Plan → Act → Reflect cycle.

    This is the core of N.I.A's reasoning capability.
    Instead of one-shot execution, it:
    1. Plans multiple steps
    2. Executes one step at a time
    3. Reflects on results
    4. Adjusts if needed
    """

    def __init__(
        self,
        think_fn: Callable[[str], Any],  # Function to call LLM for thinking
        execute_fn: Callable[[str, dict], Any],  # Function to execute tools
        max_steps: int = 10,
    ) -> None:
        self._think_fn = think_fn
        self._execute_fn = execute_fn
        self._max_steps = max_steps
        self._iteration: int = 0

    async def run(self, user_input: str, context: dict[str, Any] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Execute the ReAct loop.

        Yields events as they happen:
        - {"type": "plan", "plan": ReActPlan}
        - {"type": "step_start", "step": ReasoningStep}
        - {"type": "step_complete", "step": ReasoningStep}
        - {"type": "reflect", "reflection": str}
        - {"type": "complete", "result": str}
        """
        # Phase 1: Plan
        plan = await self._plan(user_input, context)
        yield {"type": "plan", "plan": plan}

        # Phase 2: Execute steps
        while not plan.is_complete and self._iteration < self._max_steps:
            # Find next pending step
            next_step = None
            for step in plan.steps:
                if step.status == StepStatus.PENDING:
                    next_step = step
                    break

            if next_step is None:
                break

            # Execute the step
            yield {"type": "step_start", "step": next_step}
            await self._act(next_step)
            yield {"type": "step_complete", "step": next_step}

            # Phase 3: Reflect
            reflection = await self._reflect(plan, next_step)
            next_step.reflection = reflection
            yield {"type": "reflect", "reflection": reflection}

            # Check if we need to adjust the plan
            if next_step.status == StepStatus.FAILED:
                adjusted = await self._adjust_plan(plan, next_step)
                if adjusted:
                    yield {"type": "plan_adjusted", "plan": plan}

            self._iteration += 1

        # Final result
        result = self._synthesize_result(plan)
        yield {"type": "complete", "result": result, "plan": plan}

    async def _plan(self, user_input: str, context: dict[str, Any] | None = None) -> ReActPlan:
        """Phase 1: Plan - Generate a multi-step plan."""
        # Ask LLM to create a plan
        plan_prompt = f"""Create a plan to accomplish this task:

Task: {user_input}

Context: {json.dumps(context or {}, indent=2)}

Return a JSON plan with these fields:
{{
    "goal": "What we're trying to achieve",
    "steps": [
        {{
            "step_number": 1,
            "thought": "What I'm thinking about this step",
            "action": "What action to take",
            "tool": "tool_name or null",
            "tool_args": {{}}
        }}
    ],
    "confidence": 0.9
}}

Be specific about file paths, commands, and content.
Each step should be atomic and testable."""

        try:
            response = await self._think_fn(plan_prompt)
            plan_data = self._parse_plan_response(response)
            return self._build_plan(plan_data)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Create a simple single-step plan as fallback
            return ReActPlan(
                goal=user_input,
                steps=[
                    ReasoningStep(
                        step_number=1,
                        thought="LLM planning failed, executing directly",
                        action=user_input,
                    )
                ],
            )

    async def _act(self, step: ReasoningStep) -> None:
        """Phase 2: Act - Execute a single step."""
        step.status = StepStatus.EXECUTING

        try:
            if step.tool:
                # Execute via tool
                result = await self._execute_fn(step.tool, step.tool_args)
                if isinstance(result, dict):
                    step.result = result.get("output", str(result))
                    if result.get("is_error"):
                        step.status = StepStatus.FAILED
                        step.error = step.result
                    else:
                        step.status = StepStatus.COMPLETED
                else:
                    step.result = str(result)
                    step.status = StepStatus.COMPLETED
            else:
                # No tool needed, mark as completed
                step.result = step.action
                step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.error(f"Step {step.step_number} failed: {e}")

    async def _reflect(self, plan: ReActPlan, step: ReasoningStep) -> str:
        """Phase 3: Reflect - Analyze what happened."""
        reflect_prompt = f"""Reflect on what just happened:

Goal: {plan.goal}
Step {step.step_number}: {step.action}
Result: {step.result or 'No result'}
Error: {step.error or 'No error'}

What happened? What did we learn? Should we continue or adjust?"""

        try:
            response = await self._think_fn(reflect_prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"Reflection failed: {e}"

    async def _adjust_plan(self, plan: ReActPlan, failed_step: ReasoningStep) -> bool:
        """Adjust the plan based on failures."""
        adjust_prompt = f"""The previous step failed. Adjust the plan:

Goal: {plan.goal}
Failed step: {step.action}
Error: {step.error}

Current plan steps: {[s.action for s in plan.steps]}

Should we:
1. Retry the failed step
2. Skip it and continue
3. Add a new corrective step

Return a JSON with "adjustment": "retry"|"skip"|"add", and optionally "new_step"."""

        try:
            response = await self._think_fn(adjust_prompt)
            data = json.loads(response) if isinstance(response, str) else response

            adjustment = data.get("adjustment", "skip")

            if adjustment == "retry":
                failed_step.status = StepStatus.PENDING
                failed_step.error = None
                return True
            elif adjustment == "skip":
                failed_step.status = StepStatus.SKIPPED
                return True
            elif adjustment == "add" and "new_step" in data:
                new_step = ReasoningStep(
                    step_number=len(plan.steps) + 1,
                    thought=data["new_step"].get("thought", "Corrective action"),
                    action=data["new_step"].get("action", ""),
                    tool=data["new_step"].get("tool"),
                    tool_args=data["new_step"].get("tool_args", {}),
                )
                plan.steps.append(new_step)
                return True

        except Exception as e:
            logger.warning(f"Plan adjustment failed: {e}")

        # Default: skip the failed step
        failed_step.status = StepStatus.SKIPPED
        return True

    def _parse_plan_response(self, response: Any) -> dict[str, Any]:
        """Parse LLM response into plan data."""
        content = response if isinstance(response, str) else str(response)

        # Try to extract JSON
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            # Try to find JSON in response
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                return json.loads(content[start:end])
            except (ValueError, json.JSONDecodeError):
                return {"goal": "Unknown", "steps": [], "confidence": 0.3}

    def _build_plan(self, data: dict[str, Any]) -> ReActPlan:
        """Build a ReActPlan from parsed data."""
        steps = []
        for i, step_data in enumerate(data.get("steps", []), 1):
            steps.append(ReasoningStep(
                step_number=i,
                thought=step_data.get("thought", ""),
                action=step_data.get("action", ""),
                tool=step_data.get("tool"),
                tool_args=step_data.get("tool_args", {}),
            ))

        return ReActPlan(
            goal=data.get("goal", "Unknown goal"),
            steps=steps,
            max_steps=self._max_steps,
            confidence=float(data.get("confidence", 0.9)),
        )

    def _synthesize_result(self, plan: ReActPlan) -> str:
        """Create a final result summary."""
        completed = plan.completed_steps
        total = len(plan.steps)
        failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)

        lines = [f"Goal: {plan.goal}"]
        lines.append(f"Result: {completed}/{total} steps completed")

        if failed > 0:
            lines.append(f"Failed: {failed} step(s)")

        # Add step summaries
        for step in plan.steps:
            status_icon = {
                StepStatus.COMPLETED: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.SKIPPED: "-",
                StepStatus.PENDING: "○",
                StepStatus.EXECUTING: "●",
            }.get(step.status, "?")
            lines.append(f"  {status_icon} Step {step.step_number}: {step.action[:50]}")

        return "\n".join(lines)
