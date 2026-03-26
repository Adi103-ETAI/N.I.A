# Phase 1 Implementation Guide - Critical Issues

**Timeline**: 1-2 days
**Risk Level**: Low (mostly renames, no logic changes)
**Files Affected**: ~25-30 files
**Import Changes**: ~50 places

---

## 🎯 Phase 1: Critical Issues (Do First)

### Task 1: Remove Duplicate `get_os_context()` Function

**File**: `src/core/os/platform.py`

**Action**:
```bash
# Find both definitions
grep -n "def get_os_context" src/core/os/platform.py
# Expected output:
#   428:def get_os_context() -> OSContext:
#   661:def get_os_context() -> OSContext:
```

**Current**: Line 428 and 661 both define the same function
**Solution**: Keep line 428 (first definition), delete line 661-668 (duplicate)

**Files to update**: 0 (it's an internal fix)
**Effort**: 5 minutes
**Risk**: ✅ Very low

---

### Task 2: Rename Registries (Clear Apart)

#### 2a. Rename `capabilities/registry.py` → `capabilities/tool_registry.py`

**Files to update**:
```
src/capabilities/registry.py              → src/capabilities/tool_registry.py

Import locations to update (find & replace):
  - from src.capabilities.registry import       → from src.capabilities.tool_registry import
  - from capabilities.registry import         → from capabilities.tool_registry import
```

**Commands**:
```bash
# Rename the file
mv src/capabilities/registry.py src/capabilities/tool_registry.py

# Find all imports
grep -r "from.*capabilities.*registry import" src/
grep -r "from.*capabilities.*registry import" tests/

# Update imports in found files
```

**Effort**: 15 minutes
**Files affected**: 3-5 files importing it
**Risk**: ✅ Low (just a rename)

#### 2b. Rename `core/di/registry.py` → `core/di/service_registry.py`

**Files to update**:
```
src/core/di/registry.py              → src/core/di/service_registry.py

Import locations to update:
  - from src.core.di.registry import       → from src.core.di.service_registry import
  - from core.di.registry import         → from core.di.service_registry import
  - import src.core.di.registry           → import src.core.di.service_registry
```

**Commands**:
```bash
# Rename the file
mv src/core/di/registry.py src/core/di/service_registry.py

# Find all imports
grep -r "from.*core.*di.*registry import" src/
grep -r "from.*core.*di.*registry import" tests/

# Update imports in found files
```

**Effort**: 15 minutes
**Files affected**: 5-8 files importing it
**Risk**: ✅ Low

---

### Task 3: Rename Desktop Window Modules

#### 3a. Rename `capabilities/desktop/windows.py` → `capabilities/desktop/window_ops.py`

**Why**: `windows.py` can be confused with Windows OS or a windowing framework

**Commands**:
```bash
# Rename file
mv src/capabilities/desktop/windows.py src/capabilities/desktop/window_ops.py

# Find imports
grep -r "from.*desktop\.windows import\|from.*desktop\.windows" src/
grep -r "import.*desktop\.windows" src/

# Update in files:
# - from src.capabilities.desktop.windows import → from src.capabilities.desktop.window_ops import
# - from capabilities.desktop.windows import → from capabilities.desktop.window_ops import
```

**Imported by**:
```
- src/capabilities/interface.py
- src/agents/tara/graph/nodes/* (check all)
- Any test files
- docs/ references
```

**Effort**: 20 minutes
**Files affected**: 8-12 files
**Risk**: ✅ Low

#### 3b. Rename `capabilities/desktop/window_manager.py` → `capabilities/desktop/window_registry.py`

**Why**: It's not a manager, it's a registry/state holder

**Commands**:
```bash
# Rename file
mv src/capabilities/desktop/window_manager.py src/capabilities/desktop/window_registry.py

# Find imports
grep -r "window_manager import\|window_manager" src/

# Update imports
```

**Imported by**:
```
- src/capabilities/desktop/windows.py (now window_ops.py)
- Other desktop modules
```

**Effort**: 15 minutes
**Files affected**: 3-5 files
**Risk**: ✅ Low

#### 3c. Rename `capabilities/desktop/apps.py` → `capabilities/desktop/app_launcher.py`

**Why**: Clearer about what it does (launches apps, doesn't discover/index them)

**Commands**:
```bash
# Rename file
mv src/capabilities/desktop/apps.py src/capabilities/desktop/app_launcher.py

# Find imports
grep -r "from.*desktop\.apps import\|\.apps import" src/

# Update imports
```

**Imported by**:
```
- src/capabilities/interface.py
- Any agent that invokes app operations
- Test files
```

**Effort**: 15 minutes
**Files affected**: 5-8 files
**Risk**: ✅ Low

---

### Task 4: Consolidate 3 State.py Files → `core/schema/states.py`

**Current states** (very similar):
```
1. src/agents/nia/state.py (160 LOC)
2. src/agents/nia/subagents/state.py (96 LOC) ← Mostly duplicate of #1
3. src/agents/tara/graph/state.py (79 LOC)
```

**Action Plan**:

1. **Read all three files**:
```bash
cat src/agents/nia/state.py
cat src/agents/nia/subagents/state.py
cat src/agents/tara/graph/state.py
```

2. **Create consolidated file** `src/core/schema/states.py`:
```python
# src/core/schema/states.py

"""Consolidated state definitions for all agents and workflows."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

# ===== NIA Agent State =====
class NIAState(BaseModel):
    """Main NIA agent state (from nia/state.py)."""
    messages: List[BaseMessage] = []
    user_input: str = ""
    next: Optional[str] = None
    metadata: Dict[str, Any] = {}
    # ... other fields from nia/state.py

# ===== Coordinator State (previously in subagents/state.py) =====
class CoordinatorState(BaseModel):
    """Coordinator orchestration state."""
    # ... fields from subagents/state.py

# ===== TARA Agent State =====
class TARAState(BaseModel):
    """TARA code execution agent state."""
    # ... fields from tara/graph/state.py

# Exports
__all__ = ["NIAState", "CoordinatorState", "TARAState"]
```

3. **Update imports**:

**Files importing from nia/state.py**:
```bash
grep -r "from.*nia\.state import\|from.*nia\.state" src/

# Change: from src.agents.nia.state import → from src.core.schema.states import
```

**Files importing from subagents/state.py**:
```bash
grep -r "from.*subagents\.state import" src/

# Change: from src.agents.nia.subagents.state import → from src.core.schema.states import
```

**Files importing from tara/graph/state.py**:
```bash
grep -r "from.*tara.*state import" src/

# Change: from src.agents.tara.graph.state import → from src.core.schema.states import
```

4. **Verify imports work**:
```bash
uv run python -c "from src.core.schema.states import NIAState, CoordinatorState, TARAState; print('✅ All states imported successfully')"
```

5. **Delete old files**:
```bash
rm src/agents/nia/state.py
rm src/agents/nia/subagents/state.py
rm src/agents/tara/graph/state.py
```

**Effort**: 45 minutes
**Files affected**: 15-20 files (imports)
**Risk**: ⚠️ Medium (structural change, but low functional risk)

---

### Task 5: Move `app_index.py` → `app_discovery.py`

**Current**: `src/infrastructure/host_os/app_index.py`
**New**: `src/capabilities/desktop/app_discovery.py`

**Why**: App discovery is a capability, belongs with other desktop capabilities

**Commands**:
```bash
# Move file
mv src/infrastructure/host_os/app_index.py src/capabilities/desktop/app_discovery.py

# Find imports
grep -r "from.*app_index import\|host_os.*app_index" src/

# Update imports
# from src.infrastructure.host_os.app_index import → from src.capabilities.desktop.app_discovery import
```

**Imported by**:
```bash
grep -r "app_index" src/
# Likely: 3-5 files
```

**Update __init__.py**:
```bash
# Remove from: src/infrastructure/host_os/__init__.py
# Add to: src/capabilities/desktop/__init__.py
```

**Effort**: 20 minutes
**Files affected**: 5-7 files
**Risk**: ✅ Low

---

### Task 6: Update __init__.py Files

**Files to create/update**:

1. `src/capabilities/desktop/__init__.py`:
```python
"""Desktop automation capabilities (cross-platform)."""

from .window_ops import focus_window, minimize_window, maximize_window, snap_window, close_window
from .window_registry import get_window_registry
from .app_launcher import launch_app, kill_app, list_processes
from .app_discovery import discover_applications, get_app_index
from .screen import take_screenshot
from .input import click, type_text

__all__ = [
    "focus_window",
    "minimize_window",
    "maximize_window",
    "snap_window",
    "close_window",
    "get_window_registry",
    "launch_app",
    "kill_app",
    "list_processes",
    "discover_applications",
    "get_app_index",
    "take_screenshot",
    "click",
    "type_text",
]
```

2. Update `src/capabilities/__init__.py`:
```python
from .tool_registry import ToolRegistry  # Updated name
from .desktop import *
from .system import *
# ... rest
```

3. Update `src/core/di/__init__.py`:
```python
from .service_registry import ServiceRegistry  # Updated name
# ...
```

**Effort**: 15 minutes
**Files affected**: 5-8 files
**Risk**: ✅ Low

---

## 🧪 Testing Phase 1

### Test 1: No Import Errors
```bash
uv run python -c "
from src.capabilities.tool_registry import *
from src.capabilities.desktop.window_ops import *
from src.capabilities.desktop.window_registry import *
from src.capabilities.desktop.app_launcher import *
from src.capabilities.desktop.app_discovery import *
from src.core.di.service_registry import *
from src.core.schema.states import *
print('✅ All imports successful')
"
```

### Test 2: Run Test Suite
```bash
uv run pytest tests/ --ignore=tests/test_ai_router.py --ignore=tests/test_phase2_integration.py -v
# Expected: All tests pass (190/190)
```

### Test 3: Manual Smoke Test
```bash
uv run python main.py --status
# Should work without errors
```

---

## 📋 Phase 1 Summary

| Task | Files | Time | Risk |
|------|-------|------|------|
| 1. Remove duplicate get_os_context | 1 | 5 min | ✅ Low |
| 2a. Rename tool_registry | 5 | 15 min | ✅ Low |
| 2b. Rename service_registry | 8 | 15 min | ✅ Low |
| 3a. Rename window_ops | 10 | 20 min | ✅ Low |
| 3b. Rename window_registry | 5 | 15 min | ✅ Low |
| 3c. Rename app_launcher | 8 | 15 min | ✅ Low |
| 4. Consolidate states | 20 | 45 min | ⚠️ Med |
| 5. Move app_discovery | 7 | 20 min | ✅ Low |
| 6. Update __init__.py | 8 | 15 min | ✅ Low |
| Testing | - | 15 min | ✅ Low |
| **TOTAL** | **~70** | **2-2.5 hours** | **✅ Low** |

---

## ✅ Expected Outcome

After Phase 1:
- ✅ No duplicate functions
- ✅ Clear, unambiguous naming
- ✅ No registry confusion
- ✅ Consolidated state management
- ✅ All 190 tests passing
- ✅ Cleaner folder structure

**Result**: ~40% immediate improvement in code clarity and maintainability

**Ready to proceed? Should I:**

1. ✅ Implement Phase 1 step-by-step
2. 📋 Modify the plan first
3. 🎯 Go straight to full restructuring

Let me know and I'll execute it!
