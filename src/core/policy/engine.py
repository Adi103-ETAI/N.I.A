from __future__ import annotations

from typing import TYPE_CHECKING, List
from pydantic import BaseModel
from src.core.policy.scopes import CapabilityScope

if TYPE_CHECKING:
    from src.core.schema.mission import MissionManifest

class CapabilityAudit(BaseModel):
    auto_approved: List[CapabilityScope]
    needs_approval: List[CapabilityScope]

class ScopeViolation(Exception):
    """Raised when an agent attempts to execute a tool without the approved scope."""
    pass

def audit_plan(manifest: MissionManifest) -> CapabilityAudit:
    """
    Audits the required scopes in a MissionManifest to split them into auto-approved
    vs manual-approval scopes before prompting the human.
    """
    auto_approved = []
    needs_approval = []
    
    for scope in manifest.required_scopes:
        if scope == CapabilityScope.READ_ONLY:
            auto_approved.append(scope)
        else:
            needs_approval.append(scope)
            
    return CapabilityAudit(
        auto_approved=list(set(auto_approved)),
        needs_approval=list(set(needs_approval))
    )

def enforce_at_runtime(required_scope: CapabilityScope, manifest: MissionManifest) -> bool:
    """
    Silent check — is this tool within approved scope?
    Returns True (proceed) or raises ScopeViolation.
    Never prompts the human at runtime.
    
    Args:
        required_scope: The CapabilityScope demanded by the tool manifest.
        manifest: The current running MissionManifest governing this swarm.
        
    Raises:
        ScopeViolation: if the scope requires approval and was not approved.
    """
    # Read-only operations are always permitted without asking
    if required_scope == CapabilityScope.READ_ONLY:
        return True
        
    if not manifest.approved:
        raise ScopeViolation(f"Mission {manifest.mission_id} has not been approved.")
        
    if required_scope not in manifest.approved_scopes:
        raise ScopeViolation(
            f"Scope {required_scope.value} required but not approved for this mission. "
            f"Approved scopes: {[s.value for s in manifest.approved_scopes]}"
        )
        
    return True

