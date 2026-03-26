# Changelog

All notable project updates are tracked here.

## 2026-03-26

### Completed
- Phase 1 restructuring completed:
  - Removed duplicate `get_os_context()` definition.
  - Renamed/moved registries and desktop modules with compatibility shims.
  - Added centralized state facade: `src/core/schema/states.py`.
  - Fixed root config/defaults path usage to centralized `src/core/config/defaults/`.

- Phase 2 restructuring completed:
  - Added `src/core/security/` (`warden.py`, `validation.py`, `policies.py`) with backward compatibility from legacy paths.
  - Added `src/core/utils/` (`file_utils.py`, `text_utils.py`, `graph_utils.py`) and redirected legacy helper modules through wrappers/shims.
  - Split `src/models/manager.py` into:
    - `src/models/config.py`
    - `src/models/factory.py`
    - `src/models/presets.py`
    - `src/models/embeddings.py`
    while preserving `manager.py` compatibility surface.

### Compatibility & Stability
- Updated imports and compatibility aliases across `src/` and tests.
- Added compatibility shims for `src/core/platform.py` and `src/core/context.py`.
- Planner integration compatibility fixes applied for legacy call/patch paths.

### Decisions
- Filesystem consolidation was intentionally deferred and remains unchanged.
