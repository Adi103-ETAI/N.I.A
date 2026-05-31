"""Shell quoting utilities.

Provides safe shell argument quoting and command construction.
"""

from __future__ import annotations

import shlex
import sys
from typing import List


def shell_quote(arg: str) -> str:
    """Quote a string for safe use in a shell command.

    Uses shlex.quote on Unix-like systems. On Windows, wraps in double
    quotes with appropriate escaping.
    """
    if sys.platform == "win32":
        return _win32_quote(arg)
    return shlex.quote(arg)


def _win32_quote(arg: str) -> str:
    """Quote an argument for Windows cmd.exe.

    Wraps in double quotes and escapes special characters.
    """
    if not arg:
        return '""'

    # If the argument already contains spaces or special chars, wrap in quotes
    needs_quoting = any(c in arg for c in " \t\n\"&|<>^%")
    if not needs_quoting:
        return arg

    # Escape double quotes within the argument
    escaped = arg.replace('"', '\\"')
    return f'"{escaped}"'


def shell_join(args: List[str]) -> str:
    """Join a list of arguments into a shell command string.

    Each argument is properly quoted for safe shell execution.
    """
    return " ".join(shell_quote(arg) for arg in args)


def shell_split(command: str) -> List[str]:
    """Split a shell command string into arguments.

    Handles quoted strings and escapes properly.
    """
    try:
        return shlex.split(command, posix=(sys.platform != "win32"))
    except ValueError:
        # If shlex fails, fall back to simple splitting
        return command.split()


def escape_shell_special(arg: str) -> str:
    """Escape shell special characters in a string.

    Useful for embedding user input in shell commands safely.
    """
    if sys.platform == "win32":
        # Windows: escape the special characters
        special_chars = "&|<>()^%\""
        result = []
        for char in arg:
            if char in special_chars:
                result.append("^" + char)
            else:
                result.append(char)
        return "".join(result)
    else:
        # Unix: use shlex.quote
        return shlex.quote(arg)
