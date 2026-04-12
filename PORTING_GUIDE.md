# N.I.A Tools Porting Guide

## Reference Files in OpenClaude

### Priority 1: Core Tools (Port These First)

#### FileEditTool
- Reference: `/workspaces/openclaude/src/tools/FileEditTool/FileEditTool.ts`
- Read: FileEditTool.ts, types.ts, prompt.ts, utils.js
- Port to: `src/openharness/tools/FileEditTool/`
- Focus: String matching, diff generation, formatting preservation

#### BashTool
- Reference: `/workspaces/openclaude/src/tools/BashTool/BashTool.ts`
- Read: BashTool.ts, constants.ts
- Port to: `src/openharness/tools/BashTool/`
- Focus: Streaming output, timeout, signal handling

#### FileReadTool
- Reference: `/workspaces/openclaude/src/tools/FileReadTool/FileReadTool.ts`
- Read: FileReadTool.ts, types.ts
- Port to: `src/openharness/tools/FileReadTool/`
- Focus: Line number support, encoding, content display

#### FileWriteTool
- Reference: `/workspaces/openclaude/src/tools/FileWriteTool/FileWriteTool.ts`
- Port to: `src/openharness/tools/FileWriteTool/`
- Focus: Create vs update, directory creation, validation

### Priority 2: Agent Tools (Complex but Critical)

#### AgentTool
- Reference: `/workspaces/openclaude/src/tools/AgentTool/AgentTool.ts`
- Also read: `/workspaces/openclaude/src/tools/AgentTool/runAgent.ts`
- Port to: `src/openharness/tools/AgentTool/`
- Focus: Subagent spawning, context isolation, forking

#### SendMessageTool
- Reference: `/workspaces/openclaude/src/tools/SendMessageTool/SendMessageTool.ts`
- Port to: `src/openharness/tools/SendMessageTool/`
- Focus: Inter-agent messaging pattern

#### TeamCreateTool
- Reference: `/workspaces/openclaude/src/tools/TeamCreateTool/TeamCreateTool.ts`
- Port to: `src/openharness/tools/TeamCreateTool/`
- Focus: Team management, member registry

### Priority 3: All Other Tools

For each tool (40+ more):
1. Find it in OpenClaude: `/workspaces/openclaude/src/tools/{ToolName}/`
2. Understand the logic
3. Port to N.I.A folder structure
4. Keep current folder structure from OpenHarness as base

## File Structure Template

For each tool, create this structure:

```
src/openharness/tools/{ToolName}/
├── __init__.py          # from .{ToolName} import {ToolName}
├── {ToolName}.py        # Main tool class
├── types.py             # Pydantic input/output models
├── prompt.py            # Tool description and instructions
├── constants.py         # Constants specific to tool
├── utils.py             # Helper functions
└── ui.py                # Error messages, user-facing formatting
```

## Query Engine Port

Read in OpenClaude: `/workspaces/openclaude/src/query.ts`

Add to `src/openharness/engine/query.py`:
- Query tracking (chainId, depth)
- Streaming tool executor
- Tool result budgeting
- Error recovery
- Abort signal handling

## Import Updates After Restructuring

File: `src/openharness/tools/__init__.py`

Update all imports from:
```python
from openharness.tools.file_edit_tool import FileEditTool
```

To:
```python
from openharness.tools.FileEditTool import FileEditTool
```

## Testing

After each tool:
```bash
pytest tests/test_tools/ -k tool_name
# Verify imports work
# Run full suite: pytest -q
```

---

## Porting Pattern

When you look at OpenClaude's FileEditTool.ts, you'll see:

```typescript
// What you see in OpenClaude (TypeScript)
class FileEditTool {
  name = "edit_file"
  description = "..."
  
  async execute(args: FileEditToolInput, context) {
    // logic here
  }
}
```

What you write in N.I.A (Python, same pattern):
```python
class FileEditTool(BaseTool):
    name = "edit_file"
    description = "..."
    
    async def execute(self, args: FileEditToolInput, context):
        # Same logic, Python syntax
```

## Checklist for Each Tool

```
Tool Name: ________________

□ Read OpenClaude reference files
□ Understand the logic and patterns
□ Create folder structure in N.I.A
□ Create __init__.py
□ Create {ToolName}.py (main class)
□ Create types.py (Pydantic models)
□ Create prompt.py (descriptions)
□ Create constants.py (if needed)
□ Create utils.py (helpers)
□ Create ui.py (messages)
□ Test imports work
□ Update src/openharness/tools/__init__.py
□ Run tests
□ Move to next tool
```
