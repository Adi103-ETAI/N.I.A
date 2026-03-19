from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from src.core.policy.scopes import CapabilityScope

class PlanStep(BaseModel):
    description: str = Field(description="Description of the step to perform")
    required_scopes: List[CapabilityScope] = Field(default_factory=list, description="The scopes needed for this step")
    assigned_role: str = Field(description="The role responsible for this step, e.g., 'planner', 'coder', 'researcher'")

class MissionManifest(BaseModel):
    """
    The MissionManifest is the core pre-flight contract produced by the Planner Agent.
    It contains the user's intent, the plan to execute it, and the execution budget.
    """
    mission_id: str = Field(description="Unique identifier for the mission")
    intent: str = Field(description="The original user intent")
    steps: List[PlanStep] = Field(default_factory=list, description="The steps to complete the mission")
    required_scopes: List[CapabilityScope] = Field(default_factory=list, description="All required capability scopes across all steps")
    estimated_depth: int = Field(default=1, description="Estimated depth of the subagent tree needed")
    estimated_agents: int = Field(default=1, description="Estimated total number of subagents needed")
    execution_mode: Literal["fast", "standard", "deep"] = Field(default="standard")
    approved: bool = Field(default=False, description="Whether the user has approved this plan")
    approved_scopes: List[CapabilityScope] = Field(default_factory=list, description="Scopes the user has explicitly approved for this mission")


class SubagentResult(BaseModel):
    """
    Structural contract returned by every subagent invocation (invoke_tara, invoke_iris, etc.).
    The Coordinator acts on the `status` field to decide retry, escalate, or merge.
    """
    agent_id: str = Field(description="Unique identifier for the subagent run")
    status: Literal[
        "success",
        "failed",
        "scope_violation",
        "stuck",
        "needs_clarification",
    ] = Field(description="Outcome status of the subagent execution")
    output: str = Field(default="", description="The textual output/result from the subagent")
    artifacts_created: List[str] = Field(default_factory=list, description="List of file paths or artifact IDs produced")
    scopes_used: List[CapabilityScope] = Field(default_factory=list, description="Capability scopes actually used during execution")
    tokens_used: int = Field(default=0, description="Total LLM tokens consumed by the subagent")
    failure_trace: Optional[str] = Field(default=None, description="Stack trace or error detail on failure")
