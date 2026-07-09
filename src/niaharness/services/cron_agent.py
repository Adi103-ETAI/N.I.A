"""Cron agent execution — LLM path for scheduled agent tasks.

Ported from Hermes Agent's ``cron/scheduler.py`` (3,637 LOC), scoped to
the LLM/agent execution path (not the shell/subprocess path, which
already exists in ``cron_scheduler.py``). Provides:

  - :func:`run_job` — execute a cron job that has a ``prompt`` field by
    constructing a NIA agent, building the job prompt, running the agent,
    and returning the result for delivery.
  - :func:`build_job_prompt` — assemble the full prompt from the job's
    ``prompt`` + optional ``script`` output + optional ``context_from``
    upstream-job outputs + skill loading + the ``cron_hint`` delivery
    guidance.
  - :func:`scan_assembled_cron_prompt` — two-tier injection scanner
    (strict on bare user prompts, loose on skill/data-injected prompts
    with defense-in-depth on the raw user prompt). Raises
    :class:`CronPromptInjectionBlocked` on match.
  - :func:`resolve_cron_disabled_toolsets` — always disables
    ``cronjob`` / ``messaging`` / ``clarify`` in cron context.
  - :func:`deliver_cron_result` — route the agent's output to delivery
    targets via the :class:`DeliveryRouter` from Task 9.

The cron hint tells the agent: (1) it's running as a scheduled job,
(2) its final response will be auto-delivered, (3) it should NOT call
``send_message`` itself, (4) it can respond with ``[SILENT]`` to
suppress delivery when there's nothing to report.

Usage::

    from niaharness.services.cron_agent import run_job

    result = await run_job(job)
    if result.success and result.response:
        await deliver_cron_result(job, result.response)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILENT_MARKER = "[SILENT]"

# Silence tokens the model might emit (with/without brackets).
_CRON_SILENCE_TOKENS = frozenset({"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"})

# Cron hint prepended to every job prompt.
_CRON_HINT = (
    "[IMPORTANT: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered "
    "to the user — do NOT use send_message or try to deliver "
    "the output yourself. Just produce your report/output as your "
    "final response and the system handles the rest. "
    "SILENT: If there is genuinely nothing new to report, respond "
    'with exactly "[SILENT]" (nothing else) to suppress delivery. '
    "Never combine [SILENT] with content — either report your "
    "findings normally, or say [SILENT] and nothing more.]\n\n"
)

# Toolsets always disabled in cron context.
_CRON_DISABLED_TOOLSETS = ["cronjob", "messaging", "clarify"]

# Context-from size cap.
_MAX_CONTEXT_CHARS = 8000

# Default agent inactivity timeout (seconds).
_DEFAULT_CRON_TIMEOUT = 600.0

# Default max iterations for the cron agent.
_DEFAULT_MAX_TURNS = 90


# ---------------------------------------------------------------------------
# Prompt injection scanner
# ---------------------------------------------------------------------------

# Strict patterns (applied to bare user prompts — no skills, no injected data).
_CRON_THREAT_PATTERNS: List[Tuple[str, str]] = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"),
    (r'/etc/sudoers|visudo', "sudoers_mod"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

# Loose patterns (applied when skills or injected data are present — only
# unambiguous injection directives, NOT command-shape patterns).
_CRON_SKILL_ASSEMBLED_PATTERNS: List[Tuple[str, str]] = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
]

# Exfiltration patterns (strict set only).
_CRON_SECRET_VAR_RE = r'\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)\w*\}?'
_CRON_EXFIL_PATTERNS: List[Tuple[str, str]] = [
    (rf'curl\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_curl_url"),
    (rf'wget\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_wget_url"),
    (rf'curl\s+[^\n]*(?:--data(?:-raw|-binary|-urlencode)?|-d|--form|-F)\s+[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_curl_data"),
    (rf'wget\s+[^\n]*--post-(?:data|file)=[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_wget_post"),
    (rf'curl\s+[^\n]*(?:-H|--header)\s+["\']Authorization:\s*(?:Bearer|token)\s+{_CRON_SECRET_VAR_RE}["\']', "exfil_curl_auth_header"),
]

# Pre-compile all patterns.
_CRON_THREAT_PATTERNS_COMPILED = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _CRON_THREAT_PATTERNS]
_CRON_SKILL_ASSEMBLED_PATTERNS_COMPILED = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _CRON_SKILL_ASSEMBLED_PATTERNS]
_CRON_EXFIL_PATTERNS_COMPILED = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _CRON_EXFIL_PATTERNS]


class CronPromptInjectionBlocked(Exception):
    """Raised when the assembled cron prompt trips the injection scanner."""


def _scan_strict(text: str) -> Optional[str]:
    """Scan with strict patterns (threats + exfil). Returns error string or None."""
    for pattern, desc in _CRON_THREAT_PATTERNS_COMPILED:
        if pattern.search(text):
            return f"Blocked: prompt matches threat pattern '{desc}'"
    for pattern, desc in _CRON_EXFIL_PATTERNS_COMPILED:
        if pattern.search(text):
            return f"Blocked: prompt matches exfiltration pattern '{desc}'"
    return None


def _scan_loose(text: str) -> Tuple[str, Optional[str]]:
    """Scan with loose patterns (injection directives only). Returns (cleaned, error)."""
    for pattern, desc in _CRON_SKILL_ASSEMBLED_PATTERNS_COMPILED:
        if pattern.search(text):
            return text, f"Blocked: prompt matches injection pattern '{desc}'"
    return text, None


def scan_assembled_cron_prompt(
    assembled: str,
    job: dict,
    *,
    has_skills: bool = False,
    has_injected_data: bool = False,
    user_prompt: Optional[str] = None,
) -> str:
    """Scan the fully-assembled cron prompt for injection patterns.

    Two-tier dispatch:
      - **Strict** (no skills, no injected data): applies all threat +
        exfil patterns. A bare ``rm -rf /`` in a small prompt is a smoking gun.
      - **Loose** (skills or injected data present): applies only the 4
        unambiguous injection-directive patterns. Command-shape patterns
        are dropped because vetted skill markdown / data feeds legitimately
        quote dangerous commands.

    Defense-in-depth: on the data-only loose path, the raw ``user_prompt``
    is additionally scanned with the strict set so legacy jobs keep their
    create-time guarantee at runtime.

    Raises :class:`CronPromptInjectionBlocked` on any match.
    """
    if has_skills or has_injected_data:
        cleaned, scan_error = _scan_loose(assembled)
        assembled = cleaned
        if not scan_error and not has_skills and user_prompt:
            # Data-injection path: keep the strict guarantee on the
            # user-authored prompt itself.
            scan_error = _scan_strict(user_prompt)
    else:
        scan_error = _scan_strict(assembled)

    if scan_error:
        job_label = job.get("name") or job.get("id") or "<unknown>"
        logger.warning(
            "Cron job '%s': assembled prompt blocked by injection scanner — %s",
            job_label, scan_error,
        )
        raise CronPromptInjectionBlocked(scan_error)
    return assembled


# ---------------------------------------------------------------------------
# Disabled toolsets resolver
# ---------------------------------------------------------------------------


def resolve_cron_disabled_toolsets(config: Optional[dict] = None) -> List[str]:
    """Return toolsets that must always be disabled in cron context.

    Always includes ``cronjob`` (would let a cron agent schedule more
    jobs), ``messaging`` (interactive, needs a live gateway), and
    ``clarify`` (interactive, blocks on user input). Layers on the
    operator's ``agent.disabled_toolsets`` from config.
    """
    disabled = list(_CRON_DISABLED_TOOLSETS)
    if config:
        agent_cfg = config.get("agent") or {}
        user_disabled = agent_cfg.get("disabled_toolsets") or []
        for name in user_disabled:
            name = str(name).strip()
            if name and name not in disabled:
                disabled.append(name)
    return disabled


# ---------------------------------------------------------------------------
# Job prompt builder
# ---------------------------------------------------------------------------


def build_job_prompt(job: dict, *, prerun_script: Optional[Tuple[bool, str]] = None) -> Optional[str]:
    """Build the effective prompt for a cron job.

    Assembles (in order):
      1. Optional ``## Script Output`` block (from ``job["script"]``).
      2. Optional ``## Output from job '<id>'`` blocks (from ``job["context_from"]``).
      3. The ``cron_hint`` (delivery + [SILENT] guidance).
      4. The user prompt (``job["prompt"]``).
      5. Optional skill content (from ``job["skills"]``).

    Returns ``None`` when the script produced no output (skip the LLM call).
    Raises :class:`CronPromptInjectionBlocked` when the scanner trips.
    """
    user_prompt = str(job.get("prompt") or "")
    prompt = user_prompt
    has_injected_data = False

    # 1. Run data-collection script if configured.
    script_path = job.get("script")
    if script_path:
        if prerun_script is not None:
            success, script_output = prerun_script
        else:
            success, script_output = _run_job_script(script_path)

        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
                has_injected_data = True
            else:
                return None  # Script produced nothing — skip LLM.
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )
            has_injected_data = True

    # 2. Inject context_from upstream-job outputs.
    context_from = job.get("context_from")
    if context_from:
        if isinstance(context_from, str):
            context_from = [context_from]
        for source_job_id in context_from:
            if not source_job_id or not all(c in "0123456789abcdef" for c in str(source_job_id)):
                logger.warning("context_from: skipping invalid job_id %r", source_job_id)
                continue
            try:
                output = _load_upstream_output(str(source_job_id))
                if output:
                    if len(output) > _MAX_CONTEXT_CHARS:
                        output = output[:_MAX_CONTEXT_CHARS] + "\n\n[... output truncated ...]"
                    prompt = (
                        f"## Output from job '{source_job_id}'\n"
                        "The following is the most recent output from a preceding "
                        "cron job. Use it as context for your analysis.\n\n"
                        f"```\n{output}\n```\n\n"
                        f"{prompt}"
                    )
                    has_injected_data = True
            except Exception as exc:
                logger.warning("context_from: failed to read output for job %r: %s", source_job_id, exc)

    # 3. Prepend cron hint.
    prompt = _CRON_HINT + prompt

    # 4. Load skills if configured.
    skills = job.get("skills")
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return scan_assembled_cron_prompt(
            prompt, job,
            has_skills=False,
            has_injected_data=has_injected_data,
            user_prompt=user_prompt,
        )

    # Load each skill's content.
    parts: List[str] = []
    skipped: List[str] = []
    for skill_name in skill_names:
        content = _load_skill_content(skill_name)
        if content is None:
            skipped.append(skill_name)
            continue
        if parts:
            parts.append("")
        parts.extend([
            f'[IMPORTANT: The user has invoked the "{skill_name}" skill, '
            "indicating they want you to follow its instructions. "
            "The full skill content is loaded below.]",
            "",
            content,
        ])

    if skipped:
        parts.insert(0, (
            f"[IMPORTANT: The following skill(s) were listed for this job but "
            f"could not be found and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware.]"
        ))

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])

    return scan_assembled_cron_prompt("\n".join(parts), job, has_skills=True)


def _run_job_script(script_path: str) -> Tuple[bool, str]:
    """Run a pre-job data-collection script. Returns (success, stdout)."""
    try:
        import subprocess
        result = subprocess.run(
            ["bash", "-c", script_path],
            capture_output=True, text=True, timeout=3600,
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Script timed out"
    except Exception as exc:
        return False, f"Script error: {exc}"


def _load_upstream_output(job_id: str) -> Optional[str]:
    """Load the latest output from an upstream cron job."""
    try:
        from niaharness.prompts.soul import get_nia_home
        output_dir = get_nia_home() / "cron" / "outputs" / job_id
        if not output_dir.exists():
            return None
        output_files = sorted(
            output_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not output_files:
            return None
        return output_files[0].read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _load_skill_content(skill_name: str) -> Optional[str]:
    """Load a skill's markdown content by name. Returns None if not found."""
    try:
        from niaharness.tools.skills_loader import load_skill
        skill = load_skill(skill_name)
        if skill and hasattr(skill, "content"):
            return skill.content
        if skill and isinstance(skill, dict):
            return skill.get("content", "")
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Job result
# ---------------------------------------------------------------------------


@dataclass
class CronJobResult:
    """Result of executing a cron job via the LLM agent path."""
    success: bool = False
    response: str = ""
    output_doc: str = ""
    error: Optional[str] = None
    silent: bool = False  # True when the agent responded with [SILENT].


# ---------------------------------------------------------------------------
# Main entry point — run_job
# ---------------------------------------------------------------------------


async def run_job(job: dict) -> CronJobResult:
    """Execute a cron job via the LLM agent path.

    Constructs a fresh NIA agent, builds the job prompt (with skills,
    script output, context_from), runs the agent with FULL_AUTO
    permissions (cron is non-interactive), and returns the result.

    Args:
        job: The cron job dict. Must have a ``prompt`` field. Optional
            fields: ``model``, ``script``, ``context_from``, ``skills``,
            ``workdir``, ``delivery_targets``.

    Returns:
        :class:`CronJobResult` with the agent's response.
    """
    job_id = job.get("id", "unknown")
    job_name = str(job.get("name") or job.get("prompt") or job_id)

    # Build the prompt (may raise CronPromptInjectionBlocked).
    try:
        prompt = build_job_prompt(job)
    except CronPromptInjectionBlocked as exc:
        logger.warning("Cron job '%s' blocked: %s", job_name, exc)
        return CronJobResult(
            success=False,
            error=f"Prompt blocked by injection scanner: {exc}",
            output_doc=f"# Cron Job: {job_name}\n\n**Status:** BLOCKED (injection scanner)\n\n{exc}",
        )

    if prompt is None:
        # Script produced no output — silent success.
        return CronJobResult(success=True, silent=True)

    # Set cron session env (triggers auto-approve in the approval system).
    os.environ["NIA_CRON_SESSION"] = "1"

    try:
        # Construct and run the agent.
        response = await _run_agent(job, prompt)

        # Check for [SILENT] marker.
        is_silent = _is_silence_response(response)
        if is_silent:
            return CronJobResult(success=True, silent=True, response=SILENT_MARKER)

        # Build the output doc.
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        output_doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {now_iso}\n\n"
            f"## Prompt\n\n{prompt}\n\n"
            f"## Response\n\n{response}\n"
        )
        return CronJobResult(success=True, response=response, output_doc=output_doc)

    except Exception as exc:
        logger.error("Cron job '%s' failed: %s", job_name, exc)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        output_doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {now_iso}\n"
            f"**Status:** FAILED\n\n"
            f"**Error:** {exc}\n"
        )
        return CronJobResult(success=False, error=str(exc), output_doc=output_doc)
    finally:
        os.environ.pop("NIA_CRON_SESSION", None)


async def _run_agent(job: dict, prompt: str) -> str:
    """Construct a NIA agent and run it with the cron prompt.

    Uses FULL_AUTO permission mode (cron is non-interactive — the approval
    system auto-approves all tool calls). The agent gets a restricted tool
    registry with cronjob/messaging/clarify disabled.
    """
    # Lazy imports to avoid circular deps.
    from niaharness.api.client import AnthropicApiClient
    from niaharness.config.settings import PermissionSettings, PermissionMode
    from niaharness.engine.query_engine import QueryEngine
    from niaharness.permissions.checker import PermissionChecker
    from niaharness.tools import create_default_tool_registry

    # Resolve model.
    model = job.get("model") or os.environ.get("NIAHARNESS_MODEL") or "claude-sonnet-4-6"

    # Resolve API key + base URL.
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None

    if not api_key:
        raise RuntimeError("No API key configured for cron job — set ANTHROPIC_API_KEY or OPENAI_API_KEY")

    # Build tool registry.
    registry = create_default_tool_registry()

    # Build the query engine.
    api_client = AnthropicApiClient(api_key=api_key, base_url=base_url)
    checker = PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO))

    engine = QueryEngine(
        api_client=api_client,
        tool_registry=registry,
        permission_checker=checker,
        cwd=job.get("workdir") or os.getcwd(),
        model=model,
        system_prompt="You are NIA, a helpful AI assistant running as a scheduled cron job.",
        max_tokens=4096,
        max_turns=int(job.get("max_turns", _DEFAULT_MAX_TURNS)),
    )

    # Run the agent.
    events = []
    full_text = ""
    async for event in engine.submit_message(prompt):
        from niaharness.engine.stream_events import (
            AssistantTextDelta,
            AssistantTurnComplete,
            QueryResult,
        )
        if isinstance(event, AssistantTextDelta):
            full_text += event.text
        elif isinstance(event, AssistantTurnComplete):
            if event.message.text:
                full_text = event.message.text
        # QueryResult is consumed internally.

    return full_text.strip()


def _is_silence_response(response: str) -> bool:
    """Check if the response is a silence marker."""
    if not response or not response.strip():
        return True
    stripped = response.strip()
    if stripped in _CRON_SILENCE_TOKENS:
        return True
    # Check first/last line.
    lines = stripped.split("\n")
    if lines[0].strip() in _CRON_SILENCE_TOKENS:
        return True
    if lines[-1].strip() in _CRON_SILENCE_TOKENS:
        return True
    return False


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


async def deliver_cron_result(
    job: dict,
    response: str,
    *,
    adapters: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Deliver a cron job's response to its configured targets.

    Uses the :class:`DeliveryRouter` from Task 9. Targets are resolved
    from ``job["delivery_targets"]`` (defaults to ``["local"]``).

    Returns ``None`` on success, or an error string on failure.
    """
    from niaharness.gateway.delivery import DeliveryRouter, DeliveryTarget
    from niaharness.gateway.session import SessionSource

    # Resolve delivery targets.
    target_strs = job.get("delivery_targets", ["local"])
    if isinstance(target_strs, str):
        target_strs = [t.strip() for t in target_strs.split(",") if t.strip()]

    # Build a SessionSource for "origin" resolution (if needed).
    origin = None
    if "origin" in target_strs:
        origin = SessionSource(
            platform=job.get("origin_platform", "local"),
            chat_id=job.get("origin_chat_id", ""),
        )

    targets = [DeliveryTarget.parse(t, origin=origin) for t in target_strs]

    # Build the delivery router.
    router = DeliveryRouter(adapters=adapters or {})

    # Wrap the content with job metadata.
    job_name = job.get("name", job.get("id", "cron job"))
    job_id = job.get("id", "")
    wrapped = (
        f"Cronjob Response: {job_name}\n"
        f"(job_id: {job_id})\n"
        f"-------------\n\n"
        f"{response}\n\n"
        f'To stop or manage this job, send me a new message (e.g. "stop reminder {job_name}").'
    )

    # Deliver.
    result = await router.deliver(
        wrapped,
        targets,
        job_id=str(job_id),
        job_name=job_name,
        metadata={"job_id": str(job_id), "job_name": job_name},
    )

    # Check for failures.
    errors = []
    for target_str, target_result in result.items():
        if not target_result.get("success"):
            errors.append(f"{target_str}: {target_result.get('error', 'unknown')}")

    return "; ".join(errors) if errors else None


# ---------------------------------------------------------------------------
# Per-profile isolation
# ---------------------------------------------------------------------------


def get_cron_dir() -> Path:
    """Return the cron directory for the active profile.

    Per-profile isolation: each NIA profile gets its own cron jobs,
    scripts, and output history. Anchored on ``get_nia_home()`` resolved
    at call time (not frozen at import).
    """
    try:
        from niaharness.prompts.soul import get_nia_home
        return get_nia_home() / "cron"
    except Exception:
        return Path(os.path.expanduser("~/.nia/cron"))


def get_cron_output_dir(job_id: str) -> Path:
    """Return the output directory for a specific cron job."""
    return get_cron_dir() / "outputs" / str(job_id)


def save_job_output(job_id: str, content: str) -> Path:
    """Save a cron job's output to disk for later retrieval (context_from)."""
    output_dir = get_cron_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


__all__ = [
    "CronJobResult",
    "CronPromptInjectionBlocked",
    "build_job_prompt",
    "deliver_cron_result",
    "get_cron_dir",
    "get_cron_output_dir",
    "resolve_cron_disabled_toolsets",
    "run_job",
    "save_job_output",
    "scan_assembled_cron_prompt",
    "SILENT_MARKER",
]
