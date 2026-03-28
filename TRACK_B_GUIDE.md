# Track B: PHASE 4 MASTER PLAN - Implementation Guide
**Last Updated**: 2026-03-28  
**Purpose**: File locations and tasks for Sprint 4, 5, 6 completion

---

## 📊 Overall Status

| Sprint | Status | Completion | Files to Work On |
|--------|--------|------------|------------------|
| Sprint 1 | ✅ Done | 100% | N/A |
| Sprint 2 | ✅ Done | 100% | N/A |
| Sprint 3 | ✅ Done | 100% | N/A |
| **Sprint 4** | ✅ **Done** | **100%** | N/A |
| **Sprint 5** | ✅ **Done** | **100%** | N/A |
| **Sprint 6** | ✅ **Done** | **100%** | N/A |

---

## 🟢 SPRINT 4: Coordinator (100% COMPLETE) ✅

### ✅ Implementation Done (2026-03-27)
- AsyncSqliteSaver checkpointing integrated
- Crash recovery functional
- Thread ID isolation implemented
- Tests created and passing

**Files Modified:**
- `src/agents/nia/subagents/coordinator.py` - Added checkpointing
- `src/agents/nia/graph/nodes/coordinator_node.py` - Wired db_path

**Tests Added:**
- `tests/integration/test_coordinator_resume.py`
- `tests/integration/test_coordinator_idempotency.py`

**Status: PRODUCTION READY**

---

## 🟢 SPRINT 5: Memory & State (100% COMPLETE) ✅

### ✅ Implementation Done (2026-03-28)
- NamespaceManager integrated into coordinator evaluate_node
- ContextWormhole lifecycle managed in run_coordinator
- Context injection into agent prompts (_run_step)
- Observation emission from TARA and IRIS agents
- Integration tests created and passing

**Key Features:**
- Namespace merge on success, cleanup on failure
- Real-time observation sharing between agents
- Context-aware agent prompts with team observations
- Isolated memory per agent, merged to global on success

**Files Modified:**
- `src/agents/nia/subagents/coordinator.py` - Wormhole + Namespace integration
- `src/capabilities/agents/invoke_tara.py` - Observation emission
- `src/capabilities/agents/invoke_iris.py` - Observation emission

**Tests Added:**
- `tests/integration/test_sprint5_memory.py`

**Status: PRODUCTION READY**

---

## 🟢 SPRINT 6: Observability & Cleanup (100% COMPLETE) ✅

### ✅ Implementation Done (2026-03-28)
- OpenTelemetry dependencies added to pyproject.toml
- Jaeger service added to docker-compose.yml
- Tracer initialization in main.py
- Token counter wired in coordinator evaluate_node
- Coordinator telemetry spans added
- Integration tests created and passing

**Key Features:**
- OTEL tracing (optional, enabled via OTEL_EXPORTER_OTLP_ENDPOINT)
- Jaeger UI for trace visualization (port 16686)
- Token usage tracking for cost monitoring
- Graceful degradation if OTEL not installed

**Files Modified:**
- `pyproject.toml` - OTEL dependencies
- `docker/docker-compose.yml` - Jaeger service
- `main.py` - Tracer initialization
- `src/agents/nia/subagents/coordinator.py` - Token counter + spans

**Tests Added:**
- `tests/integration/test_sprint6_observability.py`

**Status: PRODUCTION READY**

---

## 📊 Track B Summary

| Task | Files | Status |
|------|-------|--------|
| **Sprint 4: AsyncSqliteSaver** | 2 files | ✅ Done |
| **Sprint 5: Namespace merge** | 1 file | ✅ Done |
| **Sprint 5: Wormhole integration** | 3 files | ✅ Done |
| **Sprint 6: OTEL setup** | 2 files | ✅ Done |
| **Sprint 6: Token counter** | 2 files | ✅ Done |
| **Sprint 6: Spans** | 1 file | ✅ Done |
| **Sprint 6: Tests** | 1 file | ✅ Done |

**🎉 PHASE 4 MASTER PLAN: 100% COMPLETE!**

---

## 📝 Notes

- All files are well-structured and tested
- Integration points are clearly defined
- No major refactoring needed, just wiring
- Backward compatibility maintained throughout
- Can be done incrementally (commit per sprint)

---

**Ready to start?** This guide has everything needed to complete Track B!
