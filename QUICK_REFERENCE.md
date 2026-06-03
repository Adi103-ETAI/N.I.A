# Quick Reference: FileEditTool Port

## Files Created (7 total)

### 1. `/src/niaharness/tools/FileEditTool/__init__.py`
```python
from .FileEditTool import FileEditTool
__all__ = ["FileEditTool"]
```

### 2. `/src/niaharness/tools/FileEditTool/constants.py`
- `FILE_EDIT_TOOL_NAME = "edit_file"`
- `MAX_EDIT_FILE_SIZE = 1 GiB`
- Error message constants

### 3. `/src/niaharness/tools/FileEditTool/types.py`
**Input Model:**
- `file_path: str` - Absolute path to file
- `old_string: str` - Text to replace
- `new_string: str` - Replacement text  
- `replace_all: bool` - Replace all occurrences (default: False)

**Output Models:**
- `FileEditToolOutput` - Success result with patch
- `HunkInfo` - Diff hunk information
- `GitDiffInfo` - Git diff details

### 4. `/src/niaharness/tools/FileEditTool/prompt.py`
```python
def get_edit_tool_description() -> str:
    """Returns the tool's usage instructions"""
```

### 5. `/src/niaharness/tools/FileEditTool/utils.py`
**Key Functions:**
- `normalize_quotes(text)` - Convert curly → straight quotes
- `find_actual_string(content, search)` - Find with normalization
- `preserve_quote_style(old, actual, new)` - Keep file's quote style
- `apply_edit_to_file(content, old, new, replace_all)` - Apply edit
- `get_patch_for_edit(...)` - Generate unified diff
- `count_matches(content, search)` - Count occurrences

### 6. `/src/niaharness/tools/FileEditTool/ui.py`
**Formatting Functions:**
- `format_error_message(error_type, details)` 
- `format_success_message(file_path, changes_made)`

### 7. `/src/niaharness/tools/FileEditTool/FileEditTool.py`
**Main Class:**
```python
class FileEditTool(BaseTool):
    name = "edit_file"
    description = get_edit_tool_description()
    input_model = FileEditToolInput
    
    async def execute(self, arguments, context) -> ToolResult:
        # Main implementation
```

## Integration

Updated: `/src/niaharness/tools/__init__.py`
```python
from niaharness.tools.FileEditTool import FileEditTool  # NEW
```

## Testing Commands

```bash
# Syntax validation
cd /workspaces/N.I.A
python3 -m py_compile src/niaharness/tools/FileEditTool/*.py

# Check structure
tree src/niaharness/tools/FileEditTool/

# Count lines
wc -l src/niaharness/tools/FileEditTool/*.py
```

## Pattern for Next Tools

Use this as template for BashTool, FileReadTool, FileWriteTool, etc:

1. Create folder: `mkdir src/niaharness/tools/{ToolName}/`
2. Create 7 files following the same structure
3. Port logic from OpenClaude TypeScript to Python
4. Update `__init__.py` import
5. Validate syntax
6. Update PORTING_PROGRESS.md

## References

- OpenClaude source: `/workspaces/openclaude/src/tools/FileEditTool/`
- PORTING_GUIDE.md - Full methodology  
- PORTING_PROGRESS.md - Status tracker
