# N.I.A Framework — Complete Phase 4 Master Plan

---

## Core Philosophy

> The system thinks, plans, and executes autonomously. The human sets intent and approves scope. The system only resurfaces when it's genuinely stuck or about to do something it cannot undo.

---

## The Autonomy Model

```
Human gives intent
        ↓
N.I.A thinks and produces a Mission Plan
        ↓
Single pre-flight approval (scope + plan)
        ↓
Full autonomous swarm execution
        ↓
Human gets result
```

The system resurfaces mid-execution **only for:**
- Discovered ambiguity that changes the mission meaning
- A required capability outside the approved scope
- A destructive/irreversible action not anticipated in the plan
- Genuine deadlock (all retry paths exhausted)

Everything else — tool selection, agent spawning, retries, memory, replanning — happens silently and automatically.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    N.I.A Core                       │
│                                                     │
│   Planner → MissionManifest → Coordinator           │
│                                    ↓                │
│              ┌─────────────────────────────────┐    │
│              │         Swarm                   │    │
│              │  [TARA] [IRIS] [Researcher] ...  │    │
│              └─────────────────────────────────┘    │
│                                                     │
│   Policy Engine   Event Bus   Memory Manager        │
│   Tool Registry   Checkpoint  OTEL Tracer           │
└─────────────────────────────────────────────────────┘
```

---

## Sprint 1 — Foundation Layer

**Goal: Everything the rest of the system depends on.**

### 1.1 Tool Registry Service

Convert `src/capabilities/` from a static package into a queryable registry.

```
src/capabilities/
    registry.py
    manifests/
        read_file.yaml
        write_file.yaml
        invoke_tara.yaml
        invoke_iris.yaml
        run_in_sandbox.yaml
        browser.yaml
        ...
```

Each manifest declares:

```yaml
name: run_in_sandbox
scope: execute
reversible: false
description: Executes code inside Docker sandbox
timeout: 300
```

The registry exposes:
- `get_tool(name)` — returns tool + manifest
- `get_scope(name)` — returns capability scope
- `get_all_by_scope(scope)` — used by the planner to reason about what it needs

---

### 1.2 Capability Scopes

```python
class CapabilityScope(Enum):
    READ_ONLY   = "read_only"    # always auto-approved
    WRITE       = "write"        # requires plan approval
    EXECUTE     = "execute"      # requires plan approval
    NETWORK     = "network"      # requires plan approval
    AGENT_SPAWN = "agent_spawn"  # requires plan approval
    DESTRUCTIVE = "destructive"  # requires explicit flag + approval
```

Scopes live in `src/core/policy/scopes.py`. Every tool in the registry declares exactly one scope. This is the only classification that matters — tools are not "safe" or "dangerous" by name, they're safe or dangerous by scope.

---

### 1.3 Policy Engine

`src/core/policy/engine.py` — operates at plan time, not runtime.

Two functions:

```python
def audit_plan(mission: MissionManifest) -> CapabilityAudit:
    # Collects all scopes needed by the plan
    # Returns what needs human approval vs auto-approved

def enforce_at_runtime(tool_name: str, manifest: MissionManifest) -> bool:
    # Silent check — is this tool within approved scope?
    # Returns True (proceed) or raises ScopeViolation
    # Never prompts the human at runtime
```

TARA's existing `protocols.py` and `security.py` get absorbed here. They don't live in the agent anymore — they live in the framework's policy layer, available to all agents.

---

## Sprint 2 — The Planner & Pre-Flight Gate

**Goal: Replace routing with autonomous planning + single approval.**

### 2.1 Mission Planner

`src/agents/nia/planner.py` — new component, replaces `routing.py` and `decision/router.py`.

The planner takes the user's intent and produces a structured `MissionManifest`:

```python
class MissionManifest(BaseModel):
    mission_id: str
    intent: str
    steps: list[PlanStep]
    required_scopes: list[CapabilityScope]
    estimated_depth: int
    estimated_agents: int
    execution_mode: Literal["fast", "standard", "deep"]
    approved: bool = False
    approved_scopes: list[CapabilityScope] = []
```

The planner is conservative — if there's any reasonable chance a scope will be needed, it declares it. Better to over-declare in the plan than to get a mid-mission scope violation.

---

### 2.2 Pre-Flight Approval Surface

`src/core/approval/preflight.py`

This is a framework primitive — the approval surface is pluggable (CLI now, UI or API later):

```
Mission: Refactor auth module and run tests

Plan:
  1. Read src/auth/*.py                    [read_only — auto]
  2. Search codebase for references        [read_only — auto]
  3. Rewrite modified files                [write — needs approval]
  4. Run test suite in sandbox             [execute — needs approval]

Requesting scopes: write, execute
Estimated agents: 2 (coder, tester)
Estimated depth: 2
Mode: standard

Proceed? [y / n / edit plan]
```

One prompt. User says yes → `MissionManifest.approved = True`, `approved_scopes` populated. Execution begins. No further interruptions unless the system hits a genuine blocker.

---

### 2.3 Remove the Router

With the planner in place:
- Delete `graph/nodes/routing.py`
- Delete `decision/router.py`
- Remove conditional routing edges from `graph/builder.py`
- N.I.A's graph becomes: `input → planner → preflight → coordinator → result`

Clean. Linear. No second-guessing LLM classification.

---

## Sprint 3 — Agent Refactor

**Goal: TARA and IRIS become swarm-compatible tool-callable agents.**

### 3.1 TARA Async Refactor

TARA's ReAct loop in `graph/workflow.py`, `executor.py`, `reasoner.py` needs to become fully async. Every blocking call gets `await`. The lazy initialization pattern stays.

Then wrap as a tool:

```python
# src/capabilities/agents/invoke_tara.py

async def invoke_tara(objective: str, manifest: MissionManifest) -> SubagentResult:
    # Spins up TARA as an async subgraph
    # Passes only the objective + relevant context, not full N.I.A history
    # Returns structured SubagentResult
    # Security constraints enforced via policy engine, not internal to TARA
```

TARA's `protocols.py` and `security.py` don't disappear — they get refactored into policy engine rules. TARA itself becomes a clean execution engine with no embedded policy.

---

### 3.2 IRIS Tool Wrapper

Same pattern:

```python
async def invoke_iris(objective: str, manifest: MissionManifest) -> SubagentResult:
    # Spins up IRIS for vision tasks
```

IRIS is simpler since it has no OS-level security concerns. The wrapper is thin.

---

### 3.3 Subagent Result Contract

```python
class SubagentResult(BaseModel):
    agent_id: str
    status: Literal["success", "failed", "scope_violation",
                    "stuck", "needs_clarification"]
    output: str
    artifacts_created: list[str]
    scopes_used: list[CapabilityScope]
    tokens_used: int
    failure_trace: str | None = None
```

`status` is the key field the Coordinator acts on. `stuck` and `needs_clarification` are the two statuses that bubble up to the human. Everything else is handled internally.

---

## Sprint 4 — The Coordinator

**Goal: The swarm brain that runs everything autonomously.**

### 4.1 Coordinator Architecture

`src/agents/nia/subagents/coordinator.py`

Built as a `StateGraph` backed by `AsyncSqliteSaver` (already in `builder.py` — same pattern, extended schema).

Responsibilities:
- Dispatch subagents via `asyncio.gather()` with role-differentiated timeouts
- Maintain the append-only context log (sole writer)
- Enforce global tree limits
- Handle adaptive retry with reflect/reformulate
- Manage budget extensions
- Wire `ContextObservation` events to `src/core/bus/events.py`
- Merge subagent vector namespaces on success

```python
SWARM_LIMITS = {
    "max_depth": 3,
    "max_total_nodes": 10,
    "max_concurrent_leaves": 4
}

ROLE_TIMEOUTS = {
    "researcher": 30,
    "coder": 300,
    "reviewer": 60,
    "tara": 120,
    "iris": 45
}
```

---

### 4.2 Autonomous Decision Loop

The Coordinator never asks the human anything unless a subagent returns `stuck` or `needs_clarification`. Everything else it resolves:

```
SubagentResult received
        ↓
status == success           → merge namespace, continue
status == failed            → reflect/reformulate, retry (max 3)
status == scope_violation   → hard fail, surface to human with context
status == stuck             → surface to human with full context
status == needs_clarification → surface to human with specific question
3rd retry still fails       → escalate to different agent type, if that fails → human
```

The human surface for `stuck` / `needs_clarification` is minimal and specific — not "something went wrong" but "I've tried 3 approaches to X, here's what I know, I need Y to continue."

---

### 4.3 Reflect/Reformulate Engine

On subagent failure, before retry:

```python
async def reflect_and_reformulate(
    original_objective: str,
    failure_trace: str,
    attempt_number: int
) -> str:
    # Calls LLM with structured reflection prompt
    # Produces a reformulated objective that avoids the failure mode
    # Returns new objective string for retry
```

This runs in the Coordinator, not the subagent. The subagent just receives a better objective and tries again with a clean state.

---

### 4.4 Budget Extension

Subagents can request more steps with a justification. The Coordinator grants based on objective signals — not just free text:

```python
class BudgetExtensionRequest(BaseModel):
    agent_id: str
    current_step: int
    steps_requested: int
    justification: str
    artifacts_produced_so_far: list[str]  # objective progress signal
    tools_called_so_far: list[str]
```

If `artifacts_produced_so_far` is empty and this is already step 4 of 5 — deny. The agent isn't making progress. If meaningful work has been done — grant. Progress is measurable, not just claimed.

---

## Sprint 5 — Memory & State

**Goal: Isolated subagent memory that merges cleanly into shared state.**

### 5.1 ChromaDB Namespace Layer

Extend `memory/manager.py`:

```python
# Global collection — read-only to subagents
global_index = chroma.get_collection("nia_global")

# Per-subagent collection — created on spawn, dropped on failure
agent_index = chroma.create_collection(f"agent_{uuid}")

# Coordinator merge on success
async def merge_namespace(agent_uuid: str):
    # Deduplicate against global index
    # Sequential write to global
    # Drop agent collection
```

On task failure — `chroma.delete_collection(f"agent_{uuid}")`. No pollution.

---

### 5.2 Sandbox Idempotency Keys

`container_engine/sandbox.py` gets idempotency key support:

```python
async def run_in_sandbox(
    command: str,
    idempotency_key: str,  # NEW: uuid per tool call
    manifest_id: str
) -> SandboxResult:
    # Check checkpoint log — has this key already executed?
    # If yes, return cached result
    # If no, execute and log the key + result
```

The `AsyncSqliteSaver` checkpoint schema extends to include executed tool call keys. On Coordinator resume after crash, re-dispatched subagents skip already-executed sandbox calls. No double side effects.

---

### 5.3 Context Wormhole

The shared read-only context all subagents mount:

```python
class ContextObservation(BaseModel):
    agent_id: str
    observation: str
    timestamp: datetime
    relevance_tags: list[str]

# Subagents emit observations onto the event bus
# Coordinator is the sole writer to the consolidated log
# Subagents read a condensed rolling summary — not N.I.A's raw history
```

Wired through `src/core/bus/events.py` — not direct function calls. Coordinator subscribes to `ContextObservation` events and maintains the log. Clean separation.

---

## Sprint 6 — Observability & Cleanup

**Goal: Full visibility into swarm execution. Delete legacy code.**

### 6.1 OTEL Instrumentation

```
src/core/telemetry/
    tracer.py        ← OTEL setup, exports to local Jaeger
    middleware.py    ← token counting per subagent-uuid
    spans.py         ← span helpers for coordinator events
```

Every subagent run:
- Gets a span tagged with `subagent_uuid`, `parent_uuid`, `role`, `mission_id`
- Token usage recorded per span → cost attribution per agent
- Coordinator decisions (retry, escalate, merge) are span events

`maintenance.py` gets trace log rotation added.

---

### 6.2 Correctness Layer

Lightweight output validation beyond Pydantic structure:

```python
class OutputValidator:
    def validate_code_output(self, result: SubagentResult) -> ValidationResult:
        # Did the coder actually write files? Check artifacts_created
        # Do the files exist in the expected namespace?

    def validate_research_output(self, result: SubagentResult) -> ValidationResult:
        # Is the output longer than a sentence?
        # Does it reference the sources it claims to?
```

Not full eval — just sanity checks that catch the "structurally valid but empty" failure mode. Failures here feed back into the reflect/reformulate loop.

---

### 6.3 Delete Legacy Code

Once Coordinator is live and validated:
- Delete `src/agents/soldiers/` entirely
- Delete `graph/nodes/routing.py`
- Delete `decision/router.py`
- Remove routing edges from `graph/builder.py`
- Remove routing-specific tests from test suite

---

### 6.4 Test Suite Expansion

New tests needed:

```
tests/
    swarm/
        test_coordinator.py              ← tree limits, retry logic
        test_parallel_execution.py       ← concurrency, no race conditions
        test_capability_negotiation.py
        test_preflight_approval.py
        test_coordinator_resume.py       ← crash + resume idempotency
    agents/
        test_tara_as_tool.py
        test_iris_as_tool.py
    memory/
        test_namespace_isolation.py
        test_namespace_merge.py
```

---

## Complete Sprint Order

```
Sprint 1 — Foundation
    Tool Registry + manifests
    Capability Scopes
    Policy Engine (audit + runtime enforce)

Sprint 2 — Planner & Pre-Flight
    Mission Planner (replaces router)
    MissionManifest schema
    Pre-flight approval surface (CLI)
    Delete routing layer

Sprint 3 — Agent Refactor
    TARA async refactor
    TARA → invoke_tara tool wrapper
    IRIS → invoke_iris tool wrapper
    SubagentResult contract

Sprint 4 — Coordinator
    Coordinator StateGraph + AsyncSqliteSaver
    asyncio.gather() dispatch
    Append-only context log + events.py wiring
    Global tree limits
    Autonomous decision loop
    Reflect/reformulate engine
    Budget extension with progress signals

Sprint 5 — Memory & State
    ChromaDB namespace isolation + merge
    Sandbox idempotency keys
    Context wormhole (ContextObservation events)

Sprint 6 — Observability & Cleanup
    OTEL + local Jaeger
    Token counting middleware
    Correctness validation layer
    Delete soldiers/
    Test suite expansion
```

---

## What Stays Untouched

| Component | Status |
|---|---|
| `main.py` | Minimal wiring for pre-flight surface only |
| `nola/` | Completely orthogonal — no changes |
| `src/core/bus/events.py` | Used more heavily, not changed |
| `src/capabilities/system/`, `desktop/`, `web/` | Tools unchanged, just registered |
| `memory/manager.py` | Extended, not rewritten |
| `maintenance.py` | Minor trace rotation addition only |

---

## The End State

A framework where:
- Any developer can define a mission and get autonomous multi-agent execution
- The human approves **intent and scope once** — the system handles everything else
- The system resurfaces only when it's genuinely blocked or about to cross a line
- Every execution is fully traceable, cost-attributed, and resumable after failure
- Adding a new agent type means writing a tool wrapper and a manifest — nothing else
