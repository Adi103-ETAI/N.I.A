"""BashTool constants."""

BASH_TOOL_NAME = "bash"
DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_OUTPUT_LENGTH = 100_000

# Progress display constants
PROGRESS_THRESHOLD_MS = 2000

# Commands that are semantic-neutral (pure output/status)
BASH_SEMANTIC_NEUTRAL_COMMANDS = frozenset({"echo", "printf", "true", "false", ":"})

# Commands that typically produce no stdout on success
BASH_SILENT_COMMANDS = frozenset({
    "mv", "cp", "rm", "mkdir", "rmdir", "chmod", "chown", "chgrp",
    "touch", "ln", "cd", "export", "unset", "wait",
})

# Commands that should not be auto-backgrounded
DISALLOWED_AUTO_BACKGROUND_COMMANDS = frozenset({"sleep"})

# Common background commands
COMMON_BACKGROUND_COMMANDS = frozenset({
    "npm", "yarn", "pnpm", "node", "python", "python3", "go", "cargo",
    "make", "docker", "terraform", "webpack", "vite", "jest", "pytest",
    "curl", "wget", "build", "test", "serve", "watch", "dev",
})

# Search commands for collapsible display
BASH_SEARCH_COMMANDS = frozenset({
    "find", "grep", "rg", "ag", "ack", "locate", "which", "whereis",
})

# Read/view commands for collapsible display
BASH_READ_COMMANDS = frozenset({
    "cat", "head", "tail", "less", "more",
    "wc", "stat", "file", "strings",
    "jq", "awk", "cut", "sort", "uniq", "tr",
})

# Directory-listing commands
BASH_LIST_COMMANDS = frozenset({"ls", "tree", "du"})
