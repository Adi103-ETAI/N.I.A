"""Prompt."""
def get_bash_description() -> str:
    return """Execute a shell command in the local environment.

Usage:
- Commands run in bash with full shell features (pipes, redirects, etc)
- Default timeout is 120 seconds (configurable up to 600)
- Output is captured from both stdout and stderr
- Use for file operations, git commands, build tools, testing
- Always provide clear descriptions of what commands do"""
