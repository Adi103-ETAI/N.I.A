"""Persistent session goals — the Ralph loop for NIA.

Ported from hermes-agent/hermes_cli/goals.py (1,765 LOC).

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, NIA feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message.

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_TOKENS = 4096
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3

CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Continue working toward the outcome above. Take the next concrete step. "
    "Stay within the stated boundaries and do not violate the constraints. "
    "Before claiming the goal is done, satisfy the Verification criterion and "
    "show the concrete evidence (command output, file contents, test result). "
    "If you hit the stated stop condition or are otherwise blocked and need "
    "user input, say so clearly and stop."
)

CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly "
    "and stop."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of three verdicts.\n\n"
    "DONE — the goal is fully satisfied:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The response explains the goal is unachievable / blocked / needs "
    "user input (treat this as DONE with reason describing the block).\n\n"
    "WAIT — the goal is NOT done, but the next step is to wait for async "
    "work to finish rather than act again.\n\n"
    "CONTINUE — not done, and there is a concrete next step the agent can "
    "take right now. This is the default when in doubt.\n\n"
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied — done, continue, or wait?"
)

JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Additional criteria the user added mid-loop (all must also be "
    "satisfied for the goal to be DONE):\n{subgoals_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal AND every additional criterion satisfied?"
)

JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Completion contract (the authoritative definition of done):\n"
    "{contract_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied per its completion contract — done, continue, or wait?"
)

JUDGE_BACKGROUND_BLOCK_TEMPLATE = (
    "Background processes the agent currently has running (it may be waiting "
    "on one of these):\n{background_lines}\n\n"
)


# ──────────────────────────────────────────────────────────────────────
# Completion contract
# ──────────────────────────────────────────────────────────────────────

_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

_CONTRACT_LABELS = {
    "outcome": "Outcome",
    "verification": "Verification",
    "constraints": "Constraints",
    "boundaries": "Boundaries",
    "stop_when": "Stop when blocked",
}

_CONTRACT_ALIASES = {
    "outcome": "outcome", "goal": "outcome", "done": "outcome", "done when": "outcome",
    "verification": "verification", "verify": "verification", "verified by": "verification",
    "evidence": "verification", "proof": "verification",
    "constraints": "constraints", "constraint": "constraints", "preserve": "constraints",
    "must not": "constraints", "do not change": "constraints",
    "boundaries": "boundaries", "boundary": "boundaries", "scope": "boundaries",
    "allowed": "boundaries", "files": "boundaries",
    "stop when": "stop_when", "stop_when": "stop_when", "blocked": "stop_when",
    "stop if blocked": "stop_when", "give up when": "stop_when",
}


@dataclass
class GoalContract:
    """Optional structured completion contract for a goal.

    Ported from hermes-agent/hermes_cli/goals.py line 294.
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> Dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})

    def render_block(self) -> str:
        lines = []
        for f in _CONTRACT_FIELDS:
            val = getattr(self, f).strip()
            if val:
                lines.append(f"- {_CONTRACT_LABELS[f]}: {val}")
        return "\n".join(lines)


def parse_contract(text: str) -> Tuple[str, GoalContract]:
    """Split user-typed goal text into a headline + structured contract.

    Ported from hermes-agent/hermes_cli/goals.py line 334.
    """
    if not text:
        return "", GoalContract()

    headline_parts: List[str] = []
    fields: Dict[str, List[str]] = {f: [] for f in _CONTRACT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        if ":" in line:
            prefix, _, value = line.partition(":")
            key = _CONTRACT_ALIASES.get(prefix.strip().lower())
            if key is not None and value.strip():
                fields[key].append(value.strip())
                matched = True
        if not matched:
            headline_parts.append(line)

    headline = " ".join(headline_parts).strip()
    contract = GoalContract(**{f: " ".join(v).strip() for f, v in fields.items()})
    return headline, contract


# ──────────────────────────────────────────────────────────────────────
# GoalState dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session.

    Ported from hermes-agent/hermes_cli/goals.py line 389.
    """

    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None
    consecutive_parse_failures: int = 0
    subgoals: List[str] = field(default_factory=list)
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    contract: GoalContract = field(default_factory=GoalContract)

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            consecutive_parse_failures=int(data.get("consecutive_parse_failures", 0) or 0),
            subgoals=subgoals,
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_until=float(data.get("waiting_until", 0.0) or 0.0),
            waiting_reason=data.get("waiting_reason"),
            waiting_since=float(data.get("waiting_since", 0.0) or 0.0),
            contract=GoalContract.from_dict(data.get("contract")),
        )

    def has_contract(self) -> bool:
        return self.contract is not None and not self.contract.is_empty()

    def render_subgoals_block(self) -> str:
        if not self.subgoals:
            return ""
        return "\n".join(f"- {i}. {text}" for i, text in enumerate(self.subgoals, start=1))


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current NIA_HOME.

    Ported from hermes-agent/hermes_cli/goals.py line 496.
    """
    try:
        from niaharness.prompts.soul import get_nia_home
        from niaharness.services.session_db import SessionDB

        home = str(get_nia_home())
    except Exception as exc:
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


def migrate_goal_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /goal from a parent session to its continuation.

    Ported from hermes-agent/hermes_cli/goals.py line 569.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_goal(old_session_id)
        if state is None or getattr(state, "status", None) == "cleared":
            return False
        save_goal(new_session_id, state)
        state.status = "cleared"
        save_goal(old_session_id, state)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        pass
    # Last-resort: os.kill(pid, 0) — POSIX only, not safe on Windows.
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False


def _session_waiting(session_id: str) -> bool:
    """Whether a goal parked on a process_registry session should stay parked."""
    if not session_id:
        return False
    try:
        from niaharness.tools.process_registry import process_registry  # type: ignore

        return bool(process_registry.is_session_waiting(session_id))
    except Exception:
        return False


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judge_response(raw: str) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Parse the judge's reply. Fail-open on unusable output.

    Ported from hermes-agent/hermes_cli/goals.py line 693.
    """
    if not raw:
        return "continue", "judge returned empty response", True, None

    text = raw.strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return "continue", f"judge reply was not JSON: {_truncate(raw, 200)!r}", True, None

    reason = str(data.get("reason") or "").strip() or "no reason provided"

    verdict_raw = data.get("verdict")
    if isinstance(verdict_raw, str):
        verdict = verdict_raw.strip().lower()
    else:
        done_val = data.get("done")
        if isinstance(done_val, str):
            done = done_val.strip().lower() in {"true", "yes", "1", "done"}
        else:
            done = bool(done_val)
        verdict = "done" if done else "continue"

    if verdict not in {"done", "continue", "wait"}:
        verdict = "continue"

    if verdict != "wait":
        return verdict, reason, False, None

    def _first_int(*keys: str) -> Optional[int]:
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                continue
        return None

    sess = data.get("wait_on_session") or data.get("session_id") or data.get("wait_session")
    if isinstance(sess, str) and sess.strip():
        return "wait", reason, False, {"session_id": sess.strip()}
    pid = _first_int("wait_on_pid", "pid", "wait_pid")
    if pid is not None:
        return "wait", reason, False, {"pid": pid}
    seconds = _first_int("wait_for_seconds", "seconds", "wait_seconds")
    if seconds is not None:
        return "wait", reason, False, {"seconds": seconds}
    return "continue", f"{reason} (wait verdict had no target — continuing)", False, None


def _render_background_block(background_processes: Optional[List[Dict[str, Any]]]) -> str:
    """Render the live background-process list for the judge prompt."""
    if not background_processes:
        return ""
    lines = []
    for p in background_processes:
        if not isinstance(p, dict):
            continue
        pid = p.get("pid", "?")
        cmd = p.get("command") or p.get("cmd") or p.get("name") or "(unknown)"
        status = p.get("status", "running")
        lines.append(f"  - pid={pid} status={status} cmd={_truncate(str(cmd), 80)}")
    if not lines:
        return ""
    return JUDGE_BACKGROUND_BLOCK_TEMPLATE.format(background_lines="\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    subgoals: Optional[List[str]] = None,
    background_processes: Optional[List[Dict[str, Any]]] = None,
    contract: Optional[GoalContract] = None,
) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Ask the auxiliary model whether the goal is satisfied.

    Ported from hermes-agent/hermes_cli/goals.py line 836.

    Returns ``(verdict, reason, parse_failed, wait_directive)`` where verdict
    is ``"done"``, ``"continue"``, ``"wait"``, or ``"skipped"``.

    This is deliberately fail-open: any error returns ``("continue", ..., False, None)``
    so a broken judge doesn't wedge progress.
    """
    if not goal.strip():
        return "skipped", "empty goal", False, None
    if not last_response.strip():
        return "continue", "empty response (nothing to evaluate)", False, None

    try:
        from niaharness.auxiliary import get_auxiliary_extra_body, get_text_auxiliary_client  # type: ignore
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None

    if client is None or not model:
        return "continue", "no auxiliary client configured", False, None

    clean_subgoals = [s.strip() for s in (subgoals or []) if s and s.strip()]
    background_block = _render_background_block(background_processes)
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    if contract is not None and not contract.is_empty():
        contract_block = contract.render_block()
        if clean_subgoals:
            extra = "\n".join(
                f"- Extra criterion {i}: {text}"
                for i, text in enumerate(clean_subgoals, start=1)
            )
            contract_block = f"{contract_block}\n{extra}"
        prompt = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            contract_block=_truncate(contract_block, 2500),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    elif clean_subgoals:
        subgoals_block = "\n".join(
            f"- {i}. {text}" for i, text in enumerate(clean_subgoals, start=1)
        )
        prompt = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    else:
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:
        logger.info("goal judge: API call failed (%s) — falling through to continue", exc)
        return "continue", f"judge error: {type(exc).__name__}", False, None

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    verdict, reason, parse_failed, wait_directive = _parse_judge_response(raw)
    logger.info(
        "goal judge: verdict=%s reason=%s%s",
        verdict, _truncate(reason, 120),
        f" wait={wait_directive}" if wait_directive else "",
    )
    return verdict, reason, parse_failed, wait_directive


def gather_background_processes(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the live background-process snapshot for the goal judge.

    Ported from hermes-agent/hermes_cli/goals.py line 967.
    """
    try:
        from niaharness.tools.process_registry import process_registry  # type: ignore

        sessions = process_registry.list_sessions(task_id=task_id) or []
    except Exception as exc:
        logger.debug("gather_background_processes failed: %s", exc)
        return []
    return [s for s in sessions if isinstance(s, dict) and s.get("status") != "exited"]


# ──────────────────────────────────────────────────────────────────────
# GoalManager
# ──────────────────────────────────────────────────────────────────────


class GoalManager:
    """Per-session goal state + continuation decisions.

    Ported from hermes-agent/hermes_cli/goals.py line 1077.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:
    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[GoalState] = load_goal(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def has_contract(self) -> bool:
        return self._state is not None and self._state.has_contract()

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared"}:
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        con = ", contract" if self.has_contract() else ""
        meta = f"{turns}{sub}{con}"
        if s.status == "active":
            if s.waiting_on_session and _session_waiting(s.waiting_on_session):
                wr = s.waiting_reason or f"session {s.waiting_on_session}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_on_pid and _pid_alive(s.waiting_on_pid):
                wr = s.waiting_reason or f"pid {s.waiting_on_pid}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_until and time.time() < s.waiting_until:
                remaining = int(s.waiting_until - time.time())
                wr = s.waiting_reason or f"{remaining}s"
                return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
            return f"⊙ Goal (active, {meta}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {meta}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({meta}): {s.goal}"
        return f"Goal ({s.status}, {meta}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None, contract: Optional[GoalContract] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            contract=contract if contract is not None else GoalContract(),
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        if self._state is None:
            return None
        self._state.contract = contract or GoalContract()
        save_goal(self.session_id, self._state)
        return self._state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> str:
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        self._state.subgoals.append(text)
        save_goal(self.session_id, self._state)
        return text

    def remove_subgoal(self, index_1based: int) -> str:
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.subgoals):
            raise IndexError(f"index out of range (1..{len(self._state.subgoals)})")
        removed = self._state.subgoals.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_subgoals(self) -> int:
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.subgoals)
        self._state.subgoals = []
        save_goal(self.session_id, self._state)
        return prev

    def render_subgoals(self) -> str:
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- /goal wait barrier -------------------------------------------

    def wait_on(self, pid: int, reason: str = "") -> GoalState:
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")
        self._state.waiting_on_pid = pid
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_on_session(self, session_id: str, reason: str = "") -> GoalState:
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._state.waiting_on_session = session_id
        self._state.waiting_on_pid = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_for_seconds(self, seconds: int, reason: str = "") -> GoalState:
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = time.time() + seconds
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def stop_waiting(self) -> bool:
        if self._state is None:
            return False
        if (
            self._state.waiting_on_pid is None
            and self._state.waiting_on_session is None
            and not self._state.waiting_until
        ):
            return False
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return True

    def is_waiting(self) -> bool:
        s = self._state
        if s is None:
            return False
        if s.waiting_on_session is not None:
            if _session_waiting(s.waiting_on_session):
                return True
            self.stop_waiting()
            return False
        if s.waiting_on_pid is not None:
            if _pid_alive(s.waiting_on_pid):
                return True
            self.stop_waiting()
            return False
        if s.waiting_until:
            if time.time() < s.waiting_until:
                return True
            self.stop_waiting()
            return False
        return False

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        Ported from hermes-agent/hermes_cli/goals.py line 1382.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "wait" | "skipped" | "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        if self.is_waiting():
            if state.waiting_on_session is not None:
                tgt = f"session {state.waiting_on_session}"
            elif state.waiting_on_pid is not None:
                tgt = f"pid {state.waiting_on_pid}"
            else:
                remaining = max(0, int(state.waiting_until - time.time()))
                tgt = f"{remaining}s remaining"
            reason = state.waiting_reason or tgt
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "waiting",
                "reason": reason,
                "message": f"⏳ Goal parked — waiting on {tgt}: {reason}",
            }

        state.turns_used += 1
        state.last_turn_at = time.time()

        verdict, reason, parse_failed, wait_directive = judge_goal(
            state.goal,
            last_response,
            subgoals=state.subgoals or None,
            background_processes=background_processes,
            contract=state.contract if state.has_contract() else None,
        )
        state.last_verdict = verdict
        state.last_reason = reason

        if parse_failed:
            state.consecutive_parse_failures += 1
        else:
            state.consecutive_parse_failures = 0

        if verdict == "wait" and wait_directive:
            if wait_directive.get("session_id"):
                self.wait_on_session(str(wait_directive["session_id"]), reason=reason)
                tgt = f"session {wait_directive['session_id']}"
            elif wait_directive.get("pid"):
                self.wait_on(int(wait_directive["pid"]), reason=reason)
                tgt = f"pid {wait_directive['pid']}"
            else:
                self.wait_for_seconds(int(wait_directive["seconds"]), reason=reason)
                tgt = f"{wait_directive['seconds']}s"
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "wait",
                "reason": reason,
                "message": f"⏳ Goal parked (judge) — waiting on {tgt}: {reason}",
            }

        if verdict == "done":
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        if state.consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge model returned unparseable output {state.consecutive_parse_failures} turns in a row"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — the judge model ({state.consecutive_parse_failures} turns) "
                    "isn't returning the required JSON verdict. Route the judge to a stricter "
                    "model in ~/.nia/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: openrouter\n"
                    "      model: google/gemini-3-flash-preview\n"
                    "Then /goal resume to continue."
                ),
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        if self._state.subgoals:
            return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)

    def render_contract(self) -> str:
        if self._state is None:
            return "(no active goal)"
        if not self._state.has_contract():
            return "(no completion contract — set one with /goal draft <objective> or inline field: value lines)"
        return self._state.contract.render_block()


__all__ = [
    "GoalContract",
    "GoalState",
    "GoalManager",
    "parse_contract",
    "load_goal",
    "save_goal",
    "clear_goal",
    "migrate_goal_to_session",
    "judge_goal",
    "gather_background_processes",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_JUDGE_TIMEOUT",
    "DEFAULT_JUDGE_MAX_TOKENS",
]
