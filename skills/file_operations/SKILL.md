---
name: File Operations
description: Read, write, create, move, copy, and delete files and directories
tools: ["read_file", "write_file", "append_file", "list_dir", "create_dir", "move_file", "copy_file", "delete_file", "file_exists", "get_file_info"]
platforms: ["windows", "linux", "darwin"]
---
To manage files and directories:

**Reading Files:**
1. Use `file_exists(path)` to check if file exists
2. Use `read_file(path)` to read text content
3. Use `get_file_info(path)` for size, dates, permissions

**Writing Files:**
- `write_file(path, content)` - Create new file (fails if exists)
- `write_file(path, content, overwrite=True)` - Overwrite existing
- `append_file(path, content)` - Add to end of file

**Directory Operations:**
- `list_dir(path)` - List files and folders
- `create_dir(path)` - Create directory (with parents)

**File Management:**
- `copy_file(src, dst)` - Copy file (preserves metadata)
- `move_file(src, dst)` - Move or rename file
- `delete_file(path, confirm=True)` - Delete (requires confirm=True)

> [!CAUTION]
> `delete_file` requires `confirm=True` as a safety lock.
> Always double-check paths before deletion.

**Path Tips:**
- Use absolute paths: `C:/Users/user/file.txt` or `/home/user/file.txt`
- Forward slashes work on all platforms
- Use `~` for home directory on Linux/macOS
