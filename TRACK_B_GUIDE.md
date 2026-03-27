# Track B: PHASE 4 MASTER PLAN - Implementation Guide
**Last Updated**: 2026-03-27  
**Purpose**: File locations and tasks for Sprint 4, 5, 6 completion

---

## 📊 Overall Status

| Sprint | Status | Completion | Files to Work On |
|--------|--------|------------|------------------|
| Sprint 1 | ✅ Done | 100% | N/A |
| Sprint 2 | ✅ Done | 100% | N/A |
| Sprint 3 | ✅ Done | 100% | N/A |
| **Sprint 4** | ⚠️ **In Progress** | **85-90%** | See below ⬇️ |
| **Sprint 5** | ⚠️ **In Progress** | **60%** | See below ⬇️ |
| **Sprint 6** | 🟡 **Infrastructure** | **40%** | See below ⬇️ |

---

## 🔴 SPRINT 4: Coordinator (85-90% Complete)

### ✅ What's Done
- Coordinator StateGraph with 3 nodes (dispatch, evaluate, reflect)
- asyncio.gather() parallel execution
- Append-only context log
- SwarmLimits enforcement
- Autonomous decision loop
- Reflect/reformulate engine
- Budget extension logic

### ❌ CRITICAL GAP: AsyncSqliteSaver Checkpointing

**Problem**: Coordinator doesn't persist state (can't resume after crash)

**Files to Modify**:

#### 1. `/workspaces/N.I.A/src/agents/nia/subagents/coordinator.py`
**Line 501** - Change from:
```python
compiled = graph.compile()
```

To:
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string(
    f"sqlite:///{db_path}/coordinator.db"
) as checkpointer:
    compiled = graph.compile(checkpointer=checkpointer)
```

**Also update**:
- Line 542-590: Modify `run_coordinator()` to accept optional `db_path` parameter
- Add default: `db_path = "data/checkpoints"`

#### 2. `/workspaces/N.I.A/src/agents/nia/graph/nodes/coordinator_node.py`
**Lines 40-60** - Pass db_path when calling coordinator:
```python
coord_result = await run_coordinator(manifest, db_path="data/checkpoints")
```

**Estimated Time**: 2 hours

---

## ⚠️ SPRINT 5: Memory & State (60% Complete)

### ✅ What's Done
- `NamespaceManager` class (404 lines) - Thread-safe, deduplication
- `IdempotentSandbox` class (382 lines) - SQLite checkpoint DB
- `ContextWormhole` class (312 lines) - Event-based observation system
- All unit tests passing

### ❌ INTEGRATION GAPS

#### Gap 1: Namespace Merge Not Called

**File**: `/workspaces/N.I.A/src/agents/nia/subagents/coordinator.py`

**Line 306-313** - In `evaluate_node()`, add after `context_log.append()`:
```python
if status == "success":
    # Archive successful output to context log
    context_log.append({...})
    
    # ✅ ADD THIS:
    from src.core.memory.namespaces import get_namespace_manager
    agent_id = result.get("agent_id", "")
    if agent_id:
        ns = get_namespace_manager()
        await ns.merge_namespace(agent_id)  # Merge to global + cleanup
```

**Line 330-340** - In failed/reflection handling, add cleanup:
```python
elif status == "failed":
    # ... existing retry logic ...
    if retries >= _MAX_RETRIES:
        # ✅ ADD THIS:
        agent_id = result.get("agent_id", "")
        if agent_id:
            ns = get_namespace_manager()
            ns.drop_namespace(agent_id)  # Cleanup failed namespace
```

---

#### Gap 2: Wormhole Not Instantiated

**File**: `/workspaces/N.I.A/src/agents/nia/subagents/coordinator.py`

**Line 542-550** - In `run_coordinator()`, add at start:
```python
async def run_coordinator(manifest: dict) -> dict:
    """Execute a mission via the coordinator sub-graph."""
    
    # ✅ ADD THIS:
    from src.core.bus.context_wormhole import ContextWormhole
    wormhole = ContextWormhole(manifest.get("mission_id", "default"))
    wormhole.subscribe()  # Start listening for observations
    
    try:
        # ... existing code ...
        initial_state = create_coordinator_state(manifest)
        initial_state["_wormhole"] = wormhole  # Pass to nodes
        
        # ... compile and run ...
        
    finally:
        wormhole.unsubscribe()  # Cleanup
```

**Line 150-180** - In `dispatch_node()`, pass wormhole to `_run_step()`:
```python
async def dispatch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    wormhole = state.get("_wormhole")  # Extract from state
    
    # ... existing code ...
    
    raw_results = await asyncio.gather(
        *(_run_step(s, wormhole, manifest) for s in batch),  # ← Pass wormhole
        return_exceptions=True,
    )
```

**Line 166-213** - Update `_run_step()` signature and inject context:
```python
async def _run_step(step: dict, wormhole, manifest: dict) -> SubagentResult:
    """Invoke the right agent wrapper with context injection."""
    role: str = step.get("assigned_role", "tara")
    objective: str = step.get("description", "")
    
    # ✅ ADD THIS: Inject wormhole context into objective
    from src.core.bus.context_wormhole import get_subagent_context
    enriched_objective = get_subagent_context(
        wormhole=wormhole,
        mission_intent=manifest.get("intent", ""),
    ) + "\n\n" + objective
    
    # Use enriched_objective instead of objective
    # ... rest of function ...
```

---

#### Gap 3: Agents Don't Emit Observations

**File**: `/workspaces/N.I.A/src/capabilities/agents/invoke_tara.py`

**Line 99-104** - After success, emit observation:
```python
# Existing return:
return SubagentResult(
    agent_id=agent_id,
    status="success",
    output=final_response,
    scopes_used=[CapabilityScope.EXECUTE],
)

# ✅ ADD BEFORE RETURN:
from src.core.bus.context_wormhole import emit_observation
await emit_observation(
    agent_id=agent_id,
    observation=final_response[:500],  # Truncate for context
    relevance_tags=["execution", "tools"],
)
```

**File**: `/workspaces/N.I.A/src/capabilities/agents/invoke_iris.py`

**Line 90-95** - Same pattern, emit observation:
```python
# ✅ ADD BEFORE RETURN:
from src.core.bus.context_wormhole import emit_observation
await emit_observation(
    agent_id=agent_id,
    observation=final_response[:500],
    relevance_tags=["vision", "analysis"],
)
```

**Estimated Time**: 4-6 hours

---

## 🟡 SPRINT 6: Observability & Cleanup (40% Complete)

### ✅ What's Done (Infrastructure)
- OTEL tracer (`src/core/telemetry/tracer.py`, 78 lines)
- Token counter (`src/core/telemetry/middleware.py`, 60 lines)
- Validation layer (`src/core/validation/__init__.py`, 467 lines, 9 tests)
- Span helpers (`src/core/telemetry/spans.py`, 62 lines)
- Soldiers directory deleted ✅

### ❌ INTEGRATION GAPS

#### Gap 1: Add OTEL Dependencies

**File**: `/workspaces/N.I.A/pyproject.toml`

**Line 7-47** - Add to dependencies array:
```toml
dependencies = [
    # ... existing deps ...
    "opentelemetry-api>=1.40.0",
    "opentelemetry-sdk>=1.40.0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.40.0",
]
```

---

#### Gap 2: Add Jaeger Docker Setup

**File**: `/workspaces/N.I.A/docker/docker-compose.yml`

Add new service:
```yaml
services:
  # ... existing services ...
  
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: nia-jaeger
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC (what tracer expects)
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped
```

---

#### Gap 3: Initialize Tracer

**File**: `/workspaces/N.I.A/main.py`

**After imports, before main()** - Add initialization:
```python
from src.core.telemetry.tracer import init_tracer
import os

# Initialize telemetry if OTEL endpoint is set
otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
init_tracer(service_name="nia-core", endpoint=otel_endpoint)
```

---

#### Gap 4: Wire Token Counter

**File**: `/workspaces/N.I.A/src/agents/nia/subagents/coordinator.py`

**Line 210-250** - In `evaluate_node()`, record tokens:
```python
from src.core.telemetry.middleware import get_token_counter

for result in completed:
    # ... existing code ...
    
    # ✅ ADD THIS:
    token_counter = get_token_counter()
    token_counter.record(
        agent_id=result.get("agent_id", ""),
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
    )
```

**Note**: Also update `SubagentResult` schema to include `prompt_tokens` and `completion_tokens` fields.

---

#### Gap 5: Wire Validation Layer

**File**: `/workspaces/N.I.A/src/agents/nia/subagents/reflect.py`

**Line 84-171** - In `reflect_and_reformulate()`, add validation check:
```python
from src.core.validation import apply_validation

async def reflect_and_reformulate(
    original_objective: str,
    failure_trace: str,
    attempt_number: int,
    role: str = "coder",  # ← ADD role parameter
    result_dict: dict = None,  # ← ADD result dict
) -> str:
    """Analyse failure and reformulate."""
    
    # ✅ ADD THIS: Validate before reflection
    if result_dict:
        validation_result = apply_validation(result_dict, role)
        
        if validation_result.verdict == ValidationVerdict.FAIL:
            # Use validation reasons in reflection prompt
            failure_trace += f"\n\nValidation Issues:\n"
            failure_trace += "\n".join(f"- {r}" for r in validation_result.reasons)
    
    # ... rest of function ...
```

---

#### Gap 6: Wire Coordinator Spans

**File**: `/workspaces/N.I.A/src/agents/nia/subagents/coordinator.py`

**Line 92-150** - Wrap `dispatch_node()` with span:
```python
from src.core.telemetry.spans import coordinator_span, record_dispatch_event

async def dispatch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    mission_id = state.get("mission", {}).get("mission_id", "unknown")
    
    with coordinator_span(mission_id, "dispatch") as span:
        # ... existing dispatch logic ...
        
        # Before returning, record event:
        record_dispatch_event(
            span=span,
            batch_size=len(batch),
            total_spawned=new_total_spawned,
            mission_id=mission_id,
        )
        
        return {...}
```

**Repeat for** `evaluate_node()` and `reflect_node()` with their respective event recorders.

---

#### Gap 7: Expand Test Suite

**Create 9 new test files** (estimated 50-100 tests total):

1. `/workspaces/N.I.A/tests/integration/test_coordinator_resume.py`
   - Test crash recovery with AsyncSqliteSaver
   - Test idempotency of tool calls

2. `/workspaces/N.I.A/tests/integration/test_parallel_execution.py`
   - Test concurrent subagent dispatch
   - Test race conditions

3. `/workspaces/N.I.A/tests/integration/test_capability_negotiation.py`
   - Test scope approval flow
   - Test scope violations

4. `/workspaces/N.I.A/tests/integration/test_preflight_approval.py`
   - Test CLI approval prompts
   - Test auto-approval for read-only

5. `/workspaces/N.I.A/tests/unit/agents/test_tara_as_tool.py`
   - Test invoke_tara wrapper
   - Test result formatting

6. `/workspaces/N.I.A/tests/unit/agents/test_iris_as_tool.py`
   - Test invoke_iris wrapper
   - Test vision result handling

7. `/workspaces/N.I.A/tests/integration/test_namespace_isolation.py`
   - Test per-agent ChromaDB isolation
   - Test memory leakage prevention

8. `/workspaces/N.I.A/tests/integration/test_namespace_merge.py`
   - Test merge deduplication
   - Test global collection updates

9. `/workspaces/N.I.A/tests/integration/test_context_propagation.py`
   - Test wormhole observation flow
   - Test context injection into prompts

**Estimated Time**: 3-4 hours for all tests

---

## 📊 Track B Summary

| Task | Files | Time | Priority |
|------|-------|------|----------|
| **Sprint 4: AsyncSqliteSaver** | 2 files | 2 hrs | 🔴 P0 |
| **Sprint 5: Namespace merge** | 1 file | 1 hr | 🔴 P0 |
| **Sprint 5: Wormhole integration** | 3 files | 3 hrs | 🔴 P0 |
| **Sprint 6: OTEL setup** | 2 files | 30 min | 🟡 P1 |
| **Sprint 6: Token counter** | 2 files | 1 hr | 🟡 P1 |
| **Sprint 6: Validation** | 1 file | 1 hr | 🟡 P1 |
| **Sprint 6: Spans** | 1 file | 1 hr | 🟡 P1 |
| **Sprint 6: Tests** | 9 files | 3-4 hrs | 🟡 P1 |
| **TOTAL** | ~20 files | **~13 hours** | |

---

## 🎯 Recommended Order

1. **Sprint 4** (2 hrs) - Critical for persistence
2. **Sprint 5** (4 hrs) - Critical for memory/context
3. **Sprint 6 Setup** (30 min) - Add deps + Docker
4. **Sprint 6 Integration** (3 hrs) - Wire all middleware
5. **Sprint 6 Tests** (4 hrs) - Comprehensive coverage

**Total**: ~13.5 hours to 100% completion

---

## 📝 Notes

- All files are well-structured and tested
- Integration points are clearly defined
- No major refactoring needed, just wiring
- Backward compatibility maintained throughout
- Can be done incrementally (commit per sprint)

---

**Ready to start?** This guide has everything needed to complete Track B!
