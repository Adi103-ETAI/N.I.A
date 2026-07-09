"""Threat pattern scanner for memory content + context files.

Ported from Hermes Agent's ``tools/threat_patterns.py``. Provides
:func:`scan_for_threats` and :func:`first_threat_message` for detecting
prompt-injection, exfiltration, C2, and persistence patterns in
user-authored content that will be injected into the system prompt.

Three scopes (additive):
  - ``"all"`` — classic injection + exfil (narrowest, applied everywhere)
  - ``"context"`` — adds role-hijack + C2 + promptware (context files, memory, tool results)
  - ``"strict"`` — adds persistence + SSH backdoor + hardcoded secrets (memory writes, skill installs)

Memory uses ``scope="strict"`` everywhere.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

MAX_SCAN_CHARS = 65_536

# Bounded filler between key attack tokens.
_FILLER = r"(?:\w+\s+){0,8}"

# Invisible unicode codepoints (zero-width, BOM, bidi, etc.).
INVISIBLE_CHARS = frozenset({
    "\u200b", "\u200c", "\u200d",  # ZWSP, ZWNJ, ZWJ
    "\u200e", "\u200f",  # LRM, RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embedding/override
    "\u2060", "\u2061", "\u2062", "\u2063", "\u2064",  # word joiner, invisible math
    "\ufeff",  # BOM / ZWNBSP
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolate
    "\u00ad",  # soft hyphen
})

# Pattern table: (regex, pattern_id, scope).
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Scope "all" (classic injection + exfil) ────────────────────────
    (rf'ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (rf'disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}(have\s+no|don\'t\s+have)\s+{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][^>]{0,2048}display\s*:\s*none', "hidden_div", "all"),
    (rf'translate\s+[^\n]{{0,512}}\s+into\s+[^\n]{{0,512}}\s+and\s+(execute|run|eval)', "translate_execute", "all"),
    (rf'do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user', "deception_hide", "all"),

    # ── Scope "context" (role-hijack + C2 + promptware) ────────────────
    (rf'you\s+are\s+{_FILLER}now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (rf'pretend\s+{_FILLER}(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (rf'output\s+{_FILLER}(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s+without\s+{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s+have\s+been\s+{_FILLER}(updated|upgraded|patched)\s+to', "fake_update", "context"),
    (r'\bname\s+yourself\s+\w+', "identity_override", "context"),
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context"),
    (rf'you\s+must\s+(?:\w+\s+){{0,3}}(register|connect|report|beacon)\b', "forced_action", "context"),
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s+{_FILLER}(?:create|write)\s+{_FILLER}(?:script|file)\s+{_FILLER}disk', "anti_forensic_disk", "context"),
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC|NIA)\w*', "env_var_unset_agent", "context"),
    (r'\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),
    (rf'curl\s+[^\n]{{0,2048}}\$\{{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "context"),
    (rf'wget\s+[^\n]{{0,2048}}\$\{{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "context"),
    (rf'cat\s+[^\n]{{0,2048}}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "context"),

    # ── Scope "strict" (persistence + SSH backdoor + hardcoded secrets) ─
    (rf'(send|post|upload|transmit)\s+[^\n]{{0,2048}}\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s+{_FILLER}(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),
    (r'authorized_keys', "ssh_backdoor", "strict"),
    (r'\$HOME/\.ssh|~/\.ssh', "ssh_access", "strict"),
    (r'\$HOME/\.nia/\.env|~/\.nia/\.env', "nia_env", "strict"),
    (rf'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{{0,2048}}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (rf'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{{0,2048}}\.nia/(config\.yaml|SOUL\.md)', "nia_config_mod", "strict"),
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
]

# Compile + organize by scope (additive: strict includes context includes all).
_COMPILED: dict[str, List[Tuple[re.Pattern, str]]] = {}
_SCOPE_HIERARCHY = {"all": ["all"], "context": ["all", "context"], "strict": ["all", "context", "strict"]}

for _scope_name, _scope_levels in _SCOPE_HIERARCHY.items():
    _compiled_list: List[Tuple[re.Pattern, str]] = []
    for _regex, _pid, _scope in _PATTERNS:
        if _scope in _scope_levels:
            _compiled_list.append((re.compile(_regex, re.IGNORECASE), _pid))
    _COMPILED[_scope_name] = _compiled_list


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """Return a list of matched pattern IDs in *content* at the given *scope*.

    Also checks for invisible unicode characters (returned as
    ``"invisible_unicode_U+XXXX"``).
    """
    if not content:
        return []

    findings: List[str] = []
    content = content[:MAX_SCAN_CHARS]

    # Invisible unicode check (on raw content before NFKC).
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # NFKC normalization (folds full-width / compatibility variants).
    normalised = unicodedata.normalize("NFKC", content)

    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid in patterns:
        if compiled.search(normalised):
            findings.append(pid)

    return findings


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """Return a human-readable error string for the first threat found, or None."""
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pid}'. "
        f"Content is injected into the system prompt and must not contain "
        f"injection or exfiltration payloads."
    )


__all__ = [
    "INVISIBLE_CHARS",
    "MAX_SCAN_CHARS",
    "first_threat_message",
    "scan_for_threats",
]
