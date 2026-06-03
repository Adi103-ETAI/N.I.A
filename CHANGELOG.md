# Changelog

All notable changes to N.I.A will be documented in this file.

## [Unreleased]

### Phase 2 - NIA Tools Exposed (2026-06-03)

NIA's unique features (memory and context awareness) are now available as tools that the brain can use during conversations.

#### Added
- `src/niaharness/tools/nia_memory_tool.py` - `nia_memory` tool: search, add_fact, add_preference, list_preferences, recent, stats
- `src/niaharness/tools/nia_context_tool.py` - `nia_context` tool: full, environment, time, session, set_user_name
- `register_nia_tools()` function in `tools/__init__.py` to wire memory/context instances into tools
- `tests/test_nia/test_tools.py` - 9 tests for both NIA tools

#### Changed
- `src/niaharness/tools/__init__.py` - NiaMemoryTool and NiaContextTool registered in default tool registry
- `src/agents/nia/nia.py` - Calls `register_nia_tools()` after engine build to wire instances
- `prompts/system.md` - Added nia_memory and nia_context to tool list

### Phase 1 - Unified Architecture (2026-06-03)

**Core Change**: NIA now delegates to niaharness's QueryEngine for the conversation loop, tool execution, permissions, hooks, cost tracking, and file state management. Previously, NIA reimplemented all of this from scratch, missing ~80% of niaharness's production features.

#### Added
- `src/agents/nia/providers/adapter.py` - Adapter bridging NIA's LLMProvider to niaharness's `SupportsStreamingMessages` protocol. Converts request/response formats between the two interfaces.
- `tests/test_nia/test_adapter.py` - 4 tests for the provider adapter (text response, tool calls, error handling, empty response)
- `tests/test_nia/test_nia.py` - 4 tests for the unified NIA class (instantiation, merged prompt, status, shutdown)

#### Changed
- **`src/agents/nia/nia.py`** - Rewritten to use `QueryEngine` instead of `Dispatcher` + `HarnessExecutorBridge`. The `process()` method now delegates to `QueryEngine.submit_message()` which handles the full conversation loop with permissions, hooks, cost tracking, and abort control. Removed `_execute_simple()` and `_execute_with_react()` methods (QueryEngine handles this now).
- **`prompts/system.md`** - Expanded tool list from 8 to 38+ tools organized by category (File Ops, Shell, Web, Code Intelligence, Agents, Teams, Scheduling, Workspace, MCP, Utility) with correct arg signatures.
- **`src/agents/nia/orchestration/dispatcher.py`** - Fixed `"file_write"` → `"write_file"` (tool name was wrong, causing "create" intent to fail)
- **`src/agents/nia/orchestration/bridge.py`** - Fixed docstring references from `"file_write"` to `"write_file"`
- **`src/niaharness/coordinator/agent_definitions.py`** - Fixed `disallowed_tools` lists: `"file_write"` → `"write_file"` (3 occurrences)
- **`src/niaharness/coordinator/coordinator_mode.py`** - Fixed worker tools list: `"file_write"` → `"write_file"`
- **`src/niaharness/ui/output.py`** - Added `"write_file"` to UI output formatting aliases

#### Fixed
- **`src/niaharness/engine/query.py:507`** - Fixed `auto_compact_if_needed()` call: removed invalid `api_client` and `system_prompt` kwargs, added missing `compact_state` return value unpacking (3-tuple instead of 2)

#### Architecture (before → after)
```
BEFORE: NIA Brain → Dispatcher → HarnessExecutorBridge → ToolRegistry (no permissions, no hooks, no cost)
AFTER:  NIA Brain → QueryEngine → ToolRegistry + Permissions + Hooks + Cost + FileState + Abort
```

NIA now gets for free:
- Permission checking (PermissionChecker)
- Pre/post tool hooks (HookExecutor)
- Cost tracking (CostTracker)
- File state cache (FileStateCache)
- Abort controller (AbortController)
- Auto-compaction
- Budget enforcement
- Tool failure loop detection
- Continuation nudge detection
