"""Shell command hardening — deobfuscation + hardline blocklist + dangerous patterns.

Ported from the reference project's tools/approval.py (2,985 lines),
focused on the critical security primitives NIA's permission system lacks:

  - **Deobfuscation** — strip ANSI escapes, null bytes, Unicode fullwidth,
    backslash-escapes, empty-string token splits, $IFS word-separator
    expansions, line continuations, and resolved home prefixes so that
    obfuscation techniques cannot bypass pattern detection
  - **Hardline blocklist** — unconditional blocks (rm -rf /, mkfs, dd to
    raw device, fork bomb, kill -1, shutdown/reboot) that fire even under
    FULL_AUTO / yolo. These have no recovery path.
  - **Dangerous patterns** — recoverable-but-costly operations (rm -rf
    /tmp, chmod -R 777, curl|sh, sensitive file writes) that require
    confirmation even in auto mode
  - **Sudo stdin guard** — block ``sudo -S`` (password guessing via stdin)
    when SUDO_PASSWORD is not configured
  - **Quote-aware command-start detection** — so ``echo "rm -rf /"`` and
    ``--title "block (reboot)"`` don't false-positive

Why this matters
----------------
NIA's current permission system is a 99-line stub that does fnmatch on
the raw command string. This means:

  - ``r\\m -rf /`` (backslash-escape) bypasses ``rm -rf /`` patterns
  - ``r''m -rf /`` (empty-string split) bypasses ``rm`` patterns
  - ``rm${IFS}-rf${IFS}/`` ($IFS expansion) bypasses ALL patterns
  - ``rm -rf /\\n`` (line continuation) bypasses root-delete patterns
  - ``shutdown`` under yolo/FULL_AUTO is allowed (no hardline floor)
  - ``sudo -S`` (password guessing) is allowed

This module closes those gaps by normalizing the command before
pattern matching, then applying hardline + dangerous + user-deny
patterns in priority order.

Usage::

    from niaharness.permissions.shell_hardening import check_command

    decision = check_command("rm -rf /")
    if not decision.allowed:
        raise PermissionError(decision.reason)
"""

from __future__ import annotations

import fnmatch
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI escape stripping (ECMA-48 full coverage)
# ---------------------------------------------------------------------------

# Matches: CSI (ESC [ ...), OSC (ESC ] ... BEL/ST), DCS (ESC P ... ST),
# 8-bit C1 (0x80-0x9F ...), and other ANSI sequences.
_ANSI_ESCAPE_RE = re.compile(
    r"""
    (?:                              # Non-capturing group for alternation
        \x1B(?:                      # ESC sequences
            [@-Z\\-_]                # 7-bit C1 Fe (except CSI)
            | \[ [0-?]* [ -/]* [@-~] # CSI: ESC [ ... <final>
            | \] [^\x07\x1B]* (?:\x07 | \x1B\\)  # OSC: ESC ] ... BEL or ST
            | P [^\x1B]* \x1B\\      # DCS: ESC P ... ST
            | X [^\x1B]* \x1B\\      # SOS: ESC X ... ST
            | \^ [^\x1B]* \x1B\\     # PM:  ESC ^ ... ST
            | _ [^\x1B]* \x1B\\      # APC: ESC _ ... ST
        )
        | [\x80-\x9F]                # 8-bit C1 control codes
    )
    """,
    re.VERBOSE,
)


def strip_ansi(text: str) -> str:
    """Strip all ANSI escape sequences (CSI, OSC, DCS, 8-bit C1)."""
    return _ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Command normalization (deobfuscation)
# ---------------------------------------------------------------------------


def _rewrite_resolved_home(command: str, home: Optional[Path] = None) -> str:
    """Fold absolute home prefixes into ``~/`` so static patterns catch them.

    E.g. ``/home/alice/.bashrc`` → ``~/.bashrc``, ``C:\\Users\\alice\\.bashrc``
    → ``~/.bashrc``. Resolves at detection time so it tracks ``$HOME`` even
    when set after this module is imported.
    """
    if home is None:
        try:
            home = Path.home()
        except Exception:
            return command
    home_str = str(home)
    if not home_str or home_str == "/":
        return command
    # Match home prefix with either separator (/ or \\).
    home_escaped = re.escape(home_str).replace(r"\\", r"[\\/]")
    return re.sub(
        rf"{home_escaped}([\\/])",
        "~/",
        command,
    )


def _rewrite_resolved_nia_home(command: str) -> str:
    """Fold NIA home prefixes into ``~/.nia/`` so static patterns catch them."""
    try:
        from niaharness.prompts.soul import get_nia_home

        nia_home = get_nia_home()
    except Exception:
        return command
    nia_home_str = str(nia_home)
    if not nia_home_str or nia_home_str == "/":
        return command
    nia_home_escaped = re.escape(nia_home_str).replace(r"\\", r"[\\/]")
    return re.sub(
        rf"{nia_home_escaped}([\\/])",
        "~/.nia/",
        command,
    )


def normalize_command_for_detection(command: str) -> str:
    """Normalize a command string before dangerous-pattern matching.

    Strips ANSI escape sequences, null bytes, and normalizes Unicode
    fullwidth characters so that obfuscation techniques cannot bypass
    the pattern-based detection.

    Adapted from reference ``_normalize_command_for_detection``.
    """
    # Strip all ANSI escape sequences (CSI, OSC, DCS, 8-bit C1, etc.)
    command = strip_ansi(command)
    # Strip null bytes.
    command = command.replace("\x00", "")
    # Normalize Unicode (fullwidth Latin, halfwidth Katakana, etc.)
    command = unicodedata.normalize("NFKC", command)
    # Collapse shell line continuations (backslash-newline). The shell
    # removes BOTH characters and joins the tokens, so `rm -rf \<newline>/`
    # executes as `rm -rf /`.
    command = re.sub(r"\\\r?\n", "", command)
    # Fold absolute home / NIA-home prefixes into canonical ~/ forms.
    command = _rewrite_resolved_nia_home(command)
    command = _rewrite_resolved_home(command)
    # Strip shell backslash-escapes: r\m → rm. Prevents \-injection bypass.
    command = re.sub(r"\\([^\n])", r"\1", command)
    # Strip empty-string literals that split tokens: r''m → rm, r"\"m → rm.
    command = re.sub(r"''|\"\"", "", command)
    # Collapse $IFS / ${IFS} word-separator expansions to a literal space.
    # In any POSIX shell IFS defaults to <space><tab><newline>, so
    # `rm${IFS}-rf${IFS}/` is executed as `rm -rf /`.
    command = re.sub(r"\$\{IFS\b[^}]*\}|\$IFS\b", " ", command)
    return command


# ---------------------------------------------------------------------------
# Quote-aware command-start detection
# ---------------------------------------------------------------------------


def _skip_shell_whitespace(command: str, pos: int) -> int:
    while pos < len(command) and command[pos] in " \t":
        pos += 1
    return pos


def _iter_shell_command_starts(command: str):
    """Yield offsets where a new shell command begins (quote-aware).

    Adapted from reference ``_iter_shell_command_starts``. Handles
    single/double quotes, backslash escapes, ``$(`` substitutions, bare
    subshell ``(`` and brace-group ``{`` openers, and command separators
    (``;``, ``&&``, ``||``, ``|``, ``&``, newline).
    """
    starts = [0]
    quote: Optional[str] = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < len(command):
                i += 2
                continue
            if ch == '"':
                quote = None
                i += 1
                continue
            if command.startswith("$(", i):
                starts.append(i + 2)
                i += 2
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command.startswith("$(", i):
            starts.append(i + 2)
            i += 2
            continue
        if ch in ("(", "{"):
            starts.append(i + 1)
            i += 1
            continue
        if ch == ";":
            starts.append(i + 1)
            i += 1
            continue
        if ch == "&":
            if i + 1 < len(command) and command[i + 1] == "&":
                starts.append(i + 2)
                i += 2
            else:
                starts.append(i + 1)
                i += 1
            continue
        if ch == "|":
            if i + 1 < len(command) and command[i + 1] == "|":
                starts.append(i + 2)
                i += 2
            else:
                starts.append(i + 1)
                i += 1
            continue
        if ch == "\n":
            starts.append(i + 1)
        i += 1

    seen: set[int] = set()
    for start in starts:
        start = _skip_shell_whitespace(command, start)
        if start < len(command) and start not in seen:
            seen.add(start)
            yield start


def _mark_command_starts(command: str) -> str:
    """Insert a newline before each real (quote-aware) command start.

    ``\\n`` is already a command-position separator, so this rewrites
    subshell ``(cmd)`` and brace-group ``{ cmd; }`` openers into a form
    the anchored patterns recognize, WITHOUT the quoted-prose false
    positives that adding ``(`` / ``{`` to ``_CMDPOS`` would cause.
    """
    offsets = sorted(o for o in _iter_shell_command_starts(command) if o > 0)
    if not offsets:
        return command
    out = command
    for offset in reversed(offsets):
        out = out[:offset] + "\n" + out[offset:]
    return out


def _command_detection_variants(command: str) -> List[str]:
    """Return the normalized + start-marked + per-word-deobfuscated variants.

    The dangerous-pattern detector runs over ALL variants so that
    quoted-prose false positives (``echo "rm -rf /"``), quote-aware
    command-start detection (``{ reboot; }``), and obfuscation tricks
    (``r\\m``, ``$(echo rm)``, ``r''m``) are all handled.
    """
    normalized = normalize_command_for_detection(command)
    marked = _mark_command_starts(normalized)

    # Per-word deobfuscation: walk each command-position word and collapse
    # shell quoting/escaping + simple literal command substitutions.
    deobfuscated_parts: List[str] = []
    for variant in (normalized, marked):
        words: List[str] = []
        for start in _iter_shell_command_word_spans(variant):
            word_start, word_end = start
            raw_word = variant[word_start:word_end]
            deobfuscated = _deobfuscate_shell_word_for_detection(raw_word)
            if deobfuscated and deobfuscated != raw_word:
                words.append(variant[:word_start] + deobfuscated + variant[word_end:])
        if words:
            deobfuscated_parts.extend(words)

    variants = [normalized, marked]
    # Add deobfuscated variants (deduped).
    for dp in deobfuscated_parts:
        if dp not in variants:
            variants.append(dp)
    return variants


# ---------------------------------------------------------------------------
# Per-word shell deobfuscation (prevents r\m, $(echo rm), r''m evasion)
# ---------------------------------------------------------------------------

# Regex for safe shell literal (no special chars that need expansion).
_SIMPLE_SHELL_LITERAL_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")

# ${var/pat/repl} substitution.
_PARAM_REPLACEMENT_RE = re.compile(r"\$\{(\w+)/[^}]*?/([^}]*)\}")

# ${var:-default} substitution.
_PARAM_DEFAULT_RE = re.compile(r"\$\{(\w+):-(\w+)\}")


def _scan_dollar_paren_end(command: str, start: int) -> Optional[int]:
    """Find the closing ``)`` for a ``$(`` starting at *start*. None if unbalanced."""
    depth = 1
    i = start + 2
    quote: Optional[str] = None
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
            i += 1
            continue
        i += 1
    return None


def _scan_backtick_end(command: str, start: int) -> Optional[int]:
    """Find the closing backtick for a backtick starting at *start*. None if unbalanced."""
    i = start + 1
    while i < len(command):
        if command[i] == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command[i] == "`":
            return i + 1
        i += 1
    return None


def _read_shell_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read one shell word without executing expansions."""
    start = _skip_shell_whitespace(command, pos)
    i = start
    quote: Optional[str] = None
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command.startswith("$(", i):
            end = _scan_dollar_paren_end(command, i)
            if end is None:
                i += 2
            else:
                i = end
            continue
        if command.startswith("${", i):
            end = command.find("}", i + 2)
            if end == -1:
                i += 2
            else:
                i = end + 1
            continue
        if ch == "`":
            end = _scan_backtick_end(command, i)
            if end is None:
                i += 1
            else:
                i = end
            continue
        if ch.isspace() or ch in ";&|":
            break
        i += 1
    return (start, i, command[start:i])


def _strip_optional_shell_quotes(word: str) -> str:
    """Strip matching outer quotes from a word."""
    if len(word) >= 2 and word[0] == word[-1] and word[0] in ("'", '"'):
        return word[1:-1]
    return word


def _is_simple_shell_literal(value: str) -> bool:
    """True if value contains only safe literal characters (no expansions)."""
    return bool(value and _SIMPLE_SHELL_LITERAL_RE.fullmatch(value))


def _literal_command_substitution_output(script: str) -> Optional[str]:
    """Resolve tiny literal command substitutions (echo/printf) without a shell."""
    import shlex
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    command = tokens[0].lower()
    args = tokens[1:]
    if command == "echo":
        while args and re.fullmatch(r"-[nEe]+", args[0]):
            args = args[1:]
        if len(args) == 1 and _is_simple_shell_literal(args[0]):
            return args[0]
        return None
    if command == "printf":
        if len(args) == 1 and _is_simple_shell_literal(args[0]):
            return args[0]
        if len(args) == 2 and args[0] == "%s" and _is_simple_shell_literal(args[1]):
            return args[1]
    return None


def _replace_simple_command_substitutions(word: str) -> str:
    """Replace $(...) and `...` with their literal output when safe."""
    chars: list[str] = []
    i = 0
    while i < len(word):
        if word.startswith("$(", i):
            end = _scan_dollar_paren_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(word[i + 2:end - 1])
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        if word[i] == "`":
            end = _scan_backtick_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(word[i + 1:end - 1])
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        chars.append(word[i])
        i += 1
    return "".join(chars)


def _replace_simple_shell_expansions(word: str) -> str:
    """Collapse ${var/pat/repl} and ${var:-default} substitutions."""
    word = _replace_simple_command_substitutions(word)
    word = _PARAM_REPLACEMENT_RE.sub(lambda m: m.group(2), word)
    return _PARAM_DEFAULT_RE.sub(lambda m: m.group(2), word)


def _strip_shell_word_syntax(word: str) -> str:
    """Collapse shell quoting/escaping: r\m → rm, r''m → rm, "rm" → rm."""
    chars: list[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(word):
        ch = word[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(word):
                chars.append(word[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
                i += 1
                continue
            chars.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(word):
            chars.append(word[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Collapse shell quoting/escaping + simple command substitutions.

    Two iterations catch nested cases like ``$(echo $(echo rm))``.
    """
    deobfuscated = word
    for _ in range(2):
        previous = deobfuscated
        deobfuscated = _replace_simple_shell_expansions(deobfuscated)
        deobfuscated = _strip_shell_word_syntax(deobfuscated)
        if deobfuscated == previous:
            break
    return deobfuscated


def _iter_shell_command_word_spans(command: str):
    """Yield (start, end) spans for each command-position word in *command*.

    A command-position word is the first word after a command separator
    (start of string, after ``;``, ``&&``, ``||``, ``|``, newline).
    """
    for start_offset in _iter_shell_command_starts(command):
        _s, word_end, _w = _read_shell_word(command, start_offset)
        if word_end > _s:
            yield (_s, word_end)


# ---------------------------------------------------------------------------
# Shell comment stripping (prevents # injection from hiding commands)
# ---------------------------------------------------------------------------


def _strip_line_comment(line: str) -> str:
    """Strip a ``#`` comment from a single line (quote-aware)."""
    quote: Optional[str] = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(line):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(line):
            i += 2
            continue
        if ch == "#":
            # Only strip if # is at start of line or preceded by whitespace.
            if i == 0 or line[i - 1] in " \t":
                return line[:i].rstrip()
        i += 1
    return line


def _strip_shell_comments(command: str) -> str:
    """Strip ``#`` comments from a multi-line command (outside quotes).

    A command like ``rm -rf / # cleanup`` must still be detected as
    dangerous. Comment stripping happens before pattern detection so
    the detector sees the actual command, not the comment-injected
    version.
    """
    if not command:
        return command
    lines = command.split("\n")
    stripped = [_strip_line_comment(line) for line in lines]
    return "\n".join(stripped)


# ---------------------------------------------------------------------------
# Command-position anchor (for hardline + dangerous patterns)
# ---------------------------------------------------------------------------

# Matches: start of string, after command separators (; && || | newline),
# after subshell openers ($(` or backtick), optionally consuming leading
# wrapper commands (sudo, env VAR=VAL, exec, nohup, setsid).
_CMDPOS = (
    r"(?:^|[;&|\n`]|\$\()"
    r"\s*"
    r"(?:sudo\s+(?:-[^\s]+\s+)*)?"
    r"(?:env\s+(?:\w+=\S*\s+)*)?"
    r"(?:(?:exec|nohup|setsid|time)\s+)*"
    r"\s*"
)


# ---------------------------------------------------------------------------
# Hardline (unconditional) blocklist
# ---------------------------------------------------------------------------

# Commands so catastrophic they should NEVER run via the agent, regardless
# of FULL_AUTO / yolo. This is a floor below yolo: opting into yolo is the
# user trusting the agent with their files and services, not trusting it
# to wipe the disk or power the box off.
#
# The list is deliberately tiny — only things with no recovery path.


def _hardline_rm_path(path_alt: str, tail: str = r"(?:\s|$|[)`;|&])") -> str:
    """Build a regex fragment matching a destructive path argument to rm."""
    return rf'(?:["\'](?:{path_alt})["\']|(?:{path_alt}){tail})'


_HARDLINE_SYSTEM_DIRS = (
    r"/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|"
    r"/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*"
)

_RM_FLAG_PREFIX = _CMDPOS + r"rm\s+(-[^\s]*\s+)*"

HARDLINE_PATTERNS: List[Tuple[str, str]] = [
    # rm recursive targeting the root filesystem.
    (
        _RM_FLAG_PREFIX
        + _hardline_rm_path(r"/(?:(?:\.\.?)?/)*(?:\.\.?)?\**|/ \*"),
        "recursive delete of root filesystem",
    ),
    # rm recursive targeting system directories.
    (
        _RM_FLAG_PREFIX + _hardline_rm_path(_HARDLINE_SYSTEM_DIRS),
        "recursive delete of system directory",
    ),
    # rm recursive targeting the home directory.
    (
        _RM_FLAG_PREFIX + _hardline_rm_path(r"(?:~|\$\{?HOME\}?)(?:/?|/\*)?"),
        "recursive delete of home directory",
    ),
    # Filesystem format.
    (r"\bmkfs(\.[a-z0-9]+)?\b", "format filesystem (mkfs)"),
    # Raw block device overwrites (dd + redirection).
    (
        r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*",
        "dd to raw block device",
    ),
    (
        r">\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b",
        "redirect to raw block device",
    ),
    # Fork bomb (classic shell form).
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    # Kill every process on the system.
    (r"\bkill\s+(-[^\s]+\s+)*-1\b", "kill all processes"),
    # System shutdown / reboot (anchored to command position).
    (_CMDPOS + r"(shutdown|reboot|halt|poweroff)\b", "system shutdown/reboot"),
    (_CMDPOS + r"init\s+[06]\b", "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r"systemctl\s+(poweroff|reboot|halt|kexec)\b", "systemctl poweroff/reboot"),
    (_CMDPOS + r"telinit\s+[06]\b", "telinit 0/6 (shutdown/reboot)"),
]

_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in HARDLINE_PATTERNS
]


# ---------------------------------------------------------------------------
# Sudo stdin guard
# ---------------------------------------------------------------------------

# When SUDO_PASSWORD is not configured, any explicit "sudo -S" in the
# command is the LLM piping a guessed password via stdin. This is a
# brute-force attack vector: the model iterates through candidate
# passwords, inspects sudo's "Sorry, try again" output, and refines.
_SUDO_STDIN_RE = re.compile(
    r"(?:^|[;&|`\n]|\$\()\s*sudo\s+-S\b",
    re.IGNORECASE,
)


def check_sudo_stdin_guard(command: str) -> Tuple[bool, Optional[str]]:
    """Detect ``sudo -S`` (stdin password) without configured SUDO_PASSWORD.

    Returns ``(is_violation, description)``. When SUDO_PASSWORD is set,
    this guard is a no-op (the agent's own sudo transformation injects
    ``-S`` internally — that path is legitimate).
    """
    import os

    if os.environ.get("SUDO_PASSWORD"):
        return (False, None)
    for variant in _command_detection_variants(command):
        if _SUDO_STDIN_RE.search(variant):
            return (
                True,
                "sudo -S (stdin password) without SUDO_PASSWORD configured — "
                "this looks like password guessing and is blocked",
            )
    return (False, None)


# ---------------------------------------------------------------------------
# Dangerous (recoverable-but-costly) patterns
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    # rm -rf on /tmp or other world-writable dirs (recoverable but costly).
    (_CMDPOS + r"rm\s+-[^\s]*r[^\s]*\s+/tmp\b", "recursive delete in /tmp"),
    # chmod -R 777 (security weakening).
    (r"\bchmod\s+-R\s+777\b", "chmod -R 777 (security weakening)"),
    # curl|sh / wget|sh (remote code execution).
    (
        r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash|zsh)\b",
        "pipe remote content to shell (curl|sh)",
    ),
    # Write to sensitive files via redirection.
    (
        r">\s*(?:~/\.ssh/|~/\.bashrc|~/\.zshrc|~/\.profile|/etc/)",
        "redirect to sensitive file",
    ),
    # Sensitive file writes via tee.
    (
        r"\btee\b[^\n]*(?:~/\.ssh/|~/\.bashrc|/etc/)",
        "tee to sensitive file",
    ),
    # Git force push to main/master.
    (
        r"\bgit\s+push\s+(?:--force|-f)\s+(?:origin\s+)?(?:main|master)\b",
        "force push to main/master",
    ),
    # Git reset --hard (data loss).
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard (data loss)"),
    # Docker run with --privileged (security weakening).
    (
        r"\bdocker\s+run\s+(?:-[^\s]*\s+)*--privileged\b",
        "docker run --privileged (security weakening)",
    ),
]

DANGEROUS_PATTERNS_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in DANGEROUS_PATTERNS
]


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def detect_hardline_command(command: str) -> Tuple[bool, Optional[str]]:
    """Check if a command matches the unconditional hardline blocklist.

    Returns ``(is_hardline, description)``. Hardline commands are blocked
    unconditionally — even under FULL_AUTO / yolo. They have no recovery
    path (filesystem destruction, raw block device overwrites, system
    shutdown, fork bomb, kill all processes).
    """
    for command_variant in _command_detection_variants(command):
        normalized = command_variant.lower()
        for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
            if pattern_re.search(normalized):
                return (True, description)
    return (False, None)


def detect_dangerous_command(command: str) -> Tuple[bool, Optional[str]]:
    """Check if a command matches a dangerous pattern.

    Returns ``(is_dangerous, description)``. Dangerous commands are
    recoverable but costly — they require confirmation even in auto mode.
    """
    for command_variant in _command_detection_variants(command):
        normalized = command_variant.lower()
        for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
            if pattern_re.search(normalized):
                return (True, description)
    return (False, None)


# ---------------------------------------------------------------------------
# Permission decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellHardeningDecision:
    """Result of checking a shell command against the hardening rules.

    Priority order (first match wins):
      1. ``hardline`` — unconditional block (rm -rf /, mkfs, etc.)
      2. ``sudo_stdin`` — sudo -S without SUDO_PASSWORD
      3. ``user_deny`` — matches a user-configured deny glob
      4. ``dangerous`` — recoverable but costly (requires confirmation)
      5. ``ok`` — no pattern matched
    """

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""
    category: str = "ok"  # ok | hardline | sudo_stdin | user_deny | dangerous
    description: Optional[str] = None


def check_command(
    command: str,
    *,
    user_deny_patterns: Optional[List[str]] = None,
    full_auto: bool = False,
) -> ShellHardeningDecision:
    """Check a shell command against the hardening rules.

    Parameters
    ----------
    command : str
        The shell command to check.
    user_deny_patterns : list of str, optional
        User-configured fnmatch globs that block a command unconditionally
        (like the hardline floor, but user-editable). Empty/None = no-op.
    full_auto : bool
        If True, dangerous commands are allowed without confirmation
        (but hardline + sudo_stdin + user_deny still block).

    Returns
    -------
    ShellHardeningDecision
        The decision. ``allowed=False`` means block. ``requires_confirmation=True``
        means the command is allowed but should prompt the user first.
    """
    if not command or not command.strip():
        return ShellHardeningDecision(allowed=True, category="ok")

    # 0. Strip shell comments before detection — a command like
    # ``rm -rf / # cleanup`` must still be detected as dangerous.
    command = _strip_shell_comments(command)

    # 1. Hardline blocklist (unconditional).
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        return ShellHardeningDecision(
            allowed=False,
            reason=(
                f"BLOCKED: {hardline_desc}. This command is on the unconditional "
                "hardline blocklist and cannot be executed via the agent — not "
                "even with FULL_AUTO or yolo. Do NOT retry or rephrase this "
                "command; the block is permanent."
            ),
            category="hardline",
            description=hardline_desc,
        )

    # 2. Sudo stdin guard.
    is_sudo_stdin, sudo_desc = check_sudo_stdin_guard(command)
    if is_sudo_stdin:
        return ShellHardeningDecision(
            allowed=False,
            reason=f"BLOCKED: {sudo_desc}",
            category="sudo_stdin",
            description=sudo_desc,
        )

    # 3. User deny patterns (unconditional, like hardline but user-editable).
    if user_deny_patterns:
        for variant in _command_detection_variants(command):
            candidate = variant.lower().strip()
            for pattern in user_deny_patterns:
                if isinstance(pattern, str) and pattern.strip():
                    if fnmatch.fnmatchcase(candidate, pattern.lower().strip()):
                        return ShellHardeningDecision(
                            allowed=False,
                            reason=(
                                f"BLOCKED: this command matches the user-defined "
                                f"deny rule '{pattern}'. It cannot be executed via "
                                "the agent — not even with FULL_AUTO or yolo."
                            ),
                            category="user_deny",
                            description=pattern,
                        )

    # 4. Dangerous patterns (require confirmation unless full_auto).
    is_dangerous, dangerous_desc = detect_dangerous_command(command)
    if is_dangerous:
        if full_auto:
            # Allow but log a warning.
            logger.warning(
                "Allowing dangerous command under FULL_AUTO: %s — %s",
                command[:80], dangerous_desc,
            )
            return ShellHardeningDecision(
                allowed=True,
                reason=f"Dangerous command allowed under FULL_AUTO: {dangerous_desc}",
                category="dangerous",
                description=dangerous_desc,
            )
        return ShellHardeningDecision(
            allowed=False,
            requires_confirmation=True,
            reason=(
                f"This command is flagged as dangerous: {dangerous_desc}. "
                "Confirmation is required before execution."
            ),
            category="dangerous",
            description=dangerous_desc,
        )

    # 5. OK — no pattern matched.
    return ShellHardeningDecision(allowed=True, category="ok")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def append_permission_audit_log(
    *,
    command: str,
    decision: ShellHardeningDecision,
    tool_name: str = "bash",
    session_id: Optional[str] = None,
) -> None:
    """Append a structured entry to the permission audit log.

    The audit log lives at ``~/.nia/permissions/audit.log`` and records
    every blocked or confirmed command for forensic review. Successful
    ``ok`` commands are not logged (would be too noisy).

    Format (one JSON line per event):
        2026-07-08T10:30:00Z BLOCK hardline "rm -rf /" tool=bash session=abc123
        description="recursive delete of root filesystem"
    """
    if decision.category == "ok" and decision.allowed and not decision.requires_confirmation:
        return  # Don't log successful non-dangerous commands.

    try:
        from datetime import datetime, timezone

        from niaharness.prompts.soul import get_nia_home

        audit_dir = get_nia_home() / "permissions"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / "audit.log"

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        action = "BLOCK" if not decision.allowed else "CONFIRM"
        # Truncate command to 200 chars for log readability.
        cmd_preview = command[:200].replace("\n", "\\n")
        parts = [
            timestamp,
            action,
            decision.category,
            f'tool={tool_name}',
        ]
        if session_id:
            parts.append(f"session={session_id}")
        parts.append(f'cmd="{cmd_preview}"')
        if decision.description:
            parts.append(f'description="{decision.description}"')

        line = " ".join(parts) + "\n"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception as exc:
        logger.debug("Failed to write permission audit log: %s", exc)


__all__ = [
    "DANGEROUS_PATTERNS",
    "HARDLINE_PATTERNS",
    "ShellHardeningDecision",
    "append_permission_audit_log",
    "check_command",
    "check_sudo_stdin_guard",
    "detect_dangerous_command",
    "detect_hardline_command",
    "normalize_command_for_detection",
    "strip_ansi",
]
