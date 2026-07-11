"""P1 Cron blueprint catalog — parameterized automation templates.

Ported from Hermes Agent's ``cron/blueprint_catalog.py`` (714 LOC).

A blueprint is a parameterized automation template: a cron schedule
with fillable slots that the user fills in to create a concrete cron
job. Blueprints make it easy to set up common automations (morning
briefing, weekly review, bill reminders) without writing a cron
expression from scratch.

Each blueprint has:
  - A ``key`` (unique identifier).
  - A ``title`` + ``description`` (human-readable).
  - A ``schedule_template`` (cron expression with {slot} placeholders).
  - A ``prompt_template`` (the agent prompt, also with {slot} placeholders).
  - A list of ``BlueprintSlot`` objects (fillable fields with types,
    defaults, options, validation).

Slot types:
  - ``time`` — HH:MM (24h), expands to minute + hour cron fields.
  - ``enum`` — one of a fixed set of options.
  - ``text`` — free text.
  - ``weekdays`` — named weekday recurrence (everyday/weekdays/weekends).

Usage::

    from niaharness.cron.blueprint_catalog import (
        get_blueprint, fill_blueprint, list_blueprints,
    )

    bp = get_blueprint("morning-brief")
    spec = fill_blueprint(bp, {"time": "07:30", "deliver": "telegram"})
    # spec = {"prompt": "...", "schedule": "30 7 * * *", "name": "...", "deliver": "telegram"}
    # Pass spec to cron.upsert_cron_job(spec).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode


class BlueprintFillError(ValueError):
    """Raised when supplied slot values fail validation."""


# Slot types the renderers understand.
_SLOT_TYPES = frozenset({"time", "enum", "text", "weekdays"})

# Named weekday recurrences → cron day-of-week field.
WEEKDAY_PRESETS: Dict[str, str] = {
    "everyday": "*",
    "weekdays": "1-5",
    "weekends": "0,6",
}

# Day name → cron day-of-week (0=Sunday).
_DAY_TO_DOW: Dict[str, str] = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
}

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


@dataclass(frozen=True)
class BlueprintSlot:
    """A single fillable field on a blueprint.

    Attributes:
        name: The slot's identifier (used in {name} placeholders).
        type: One of "time", "enum", "text", "weekdays".
        label: Human-readable label for the form.
        default: Default value (used when the slot is omitted).
        options: For type="enum": allowed values. For type="weekdays":
            the WEEKDAY_PRESETS keys.
        optional: True if the slot can be omitted.
        help: Help text for the form.
        strict: When False, ``options`` are suggestions rather than a
            closed set — any value is accepted (e.g. the deliver slot).
    """

    name: str
    type: str
    label: str
    default: Any = None
    options: tuple = ()
    optional: bool = False
    help: str = ""
    strict: bool = True

    def __post_init__(self) -> None:
        if self.type not in _SLOT_TYPES:
            raise ValueError(f"unknown slot type {self.type!r} (slot {self.name})")


@dataclass(frozen=True)
class AutomationBlueprint:
    """A parameterized automation blueprint.

    Attributes:
        key: Unique identifier.
        title: Human-readable title.
        description: What the blueprint does.
        category: Category for grouping (daily, weekly, general, etc.).
        schedule_template: Cron expression with {slot} placeholders.
        prompt_template: Agent prompt with {slot} placeholders.
        slots: List of BlueprintSlot objects.
        deliver_default: Default delivery target ("origin", "local", platform).
        skills: Skills the job loads before running.
        tags: Tags for filtering.
    """

    key: str
    title: str
    description: str
    category: str
    schedule_template: str
    prompt_template: str
    slots: List[BlueprintSlot] = field(default_factory=list)
    deliver_default: str = "origin"
    skills: tuple = ()
    tags: tuple = ()


# ---------------------------------------------------------------------------
# Slot factories
# ---------------------------------------------------------------------------


def _time_slot(default: str = "08:00") -> BlueprintSlot:
    return BlueprintSlot(
        name="time", type="time", label="What time?", default=default,
        help="24h local time, e.g. 08:00",
    )


_DELIVER_SLOT = BlueprintSlot(
    name="deliver", type="enum", label="Where to deliver?",
    default="origin", options=("origin", "local", "telegram", "discord", "email"),
    optional=False, strict=False,
    help="origin = the chat you set this up from; local = save only, no message; "
    "or any connected platform name",
)


# ---------------------------------------------------------------------------
# Curated in-repo catalog
# ---------------------------------------------------------------------------

CATALOG: List[AutomationBlueprint] = [
    AutomationBlueprint(
        key="morning-brief",
        title="Morning briefing",
        description="A short daily briefing: today's calendar, weather, and "
        "anything urgent waiting on you.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Produce a concise morning briefing for the user: today's calendar "
            "events, the local weather, and any urgent items. Keep it short and "
            "scannable. If no data sources are connected, give a brief "
            "good-morning with the date and offer to connect calendar/email."
        ),
        slots=[_time_slot("08:00"), _DELIVER_SLOT],
        tags=("daily", "briefing"),
    ),
    AutomationBlueprint(
        key="important-mail",
        title="Important-mail monitor",
        description="Check your inbox periodically and ping you ONLY about mail "
        "that actually needs attention.",
        category="email",
        schedule_template="*/{interval_min} * * * *",
        prompt_template=(
            "Check the user's inbox for new messages since the last run. Surface "
            "ONLY mail matching: {criteria}. Score candidates with the urgency "
            "classifier and deliver only what clears the bar; if nothing does, "
            "respond with [SILENT]. Requires a connected mail source; if none is "
            "configured, explain how to connect one and stop."
        ),
        slots=[
            BlueprintSlot(
                name="interval_min", type="enum", label="How often?",
                default="30", options=("15", "30", "60"),
                help="minutes between checks",
            ),
            BlueprintSlot(
                name="criteria", type="text",
                label="Only notify me if the mail…",
                default="needs a reply today, is from my manager or family, "
                "or mentions a deadline",
            ),
            _DELIVER_SLOT,
        ],
        tags=("email", "monitor"),
    ),
    AutomationBlueprint(
        key="weekly-review",
        title="Weekly review",
        description="A weekly recap: what got done, what's still open, and "
        "what's coming up.",
        category="weekly",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Produce a weekly review for the user: what was accomplished this "
            "week, still-open items, and next week's calendar. Pull from "
            "connected sources. Keep it tight."
        ),
        slots=[
            _time_slot("18:00"),
            BlueprintSlot(
                name="day", type="enum", label="Which day?",
                default="sunday",
                options=("sunday", "monday", "friday", "saturday"),
            ),
            _DELIVER_SLOT,
        ],
        tags=("weekly", "review"),
    ),
    AutomationBlueprint(
        key="workday-start",
        title="Workday start reminder",
        description="A weekday nudge with your agenda and top priorities.",
        category="daily",
        schedule_template="{minute} {hour} * * 1-5",
        prompt_template=(
            "Give the user a brief weekday start-of-day nudge: today's calendar "
            "and the 1-3 highest-priority things to focus on, inferred from "
            "recent context and any task tools. Encouraging, short, one message."
        ),
        slots=[_time_slot("09:00"), _DELIVER_SLOT],
        tags=("daily", "focus"),
    ),
    AutomationBlueprint(
        key="custom-reminder",
        title="Custom reminder",
        description="A recurring reminder in your own words, on your schedule.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template="Remind the user: {what}",
        slots=[
            BlueprintSlot(name="what", type="text", label="Remind me to…",
                       default="take a break and stretch"),
            _time_slot("14:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER_SLOT,
        ],
        tags=("reminder",),
    ),
    AutomationBlueprint(
        key="evening-winddown",
        title="Evening wind-down",
        description="An end-of-day check-in: tomorrow's calendar at a glance "
        "and anything you should prep tonight.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Give the user a short evening wind-down: tomorrow's calendar, any "
            "early commitments to prep for, and one gentle nudge to wrap up "
            "loose ends from today. Keep it calm and brief — one message."
        ),
        slots=[_time_slot("21:00"), _DELIVER_SLOT],
        tags=("daily", "evening"),
    ),
    AutomationBlueprint(
        key="news-digest",
        title="Topic news digest",
        description="A recurring digest on a topic you care about — deduped "
        "against what was already sent, so only genuinely new items land.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Search the web for new and noteworthy items about: {topic}. "
            "Dedupe against what you sent in previous runs — only include "
            "genuinely new developments. Deliver a tight digest of at most "
            "{count} bullets, each one line with a link. If nothing new since "
            "last run, respond with [SILENT]."
        ),
        slots=[
            BlueprintSlot(
                name="topic", type="text", label="What topic?",
                default="AI and technology",
                help="a subject, product, person, or search phrase",
            ),
            _time_slot("18:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            BlueprintSlot(
                name="count", type="enum", label="How many bullets?",
                default="5", options=("3", "5", "8"),
            ),
            _DELIVER_SLOT,
        ],
        tags=("digest", "research"),
    ),
    AutomationBlueprint(
        key="bill-renewal-watch",
        title="Bills & renewals reminder",
        description="A heads-up before a recurring payment, subscription "
        "renewal, or due date — so nothing auto-charges by surprise.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Remind the user about an upcoming payment or renewal: {what}. "
            "Phrase it as an actionable heads-up (e.g. 'review or cancel before "
            "it renews'), not just a notification. One short message."
        ),
        slots=[
            BlueprintSlot(
                name="what", type="text", label="What's due?",
                default="my streaming subscription renews soon",
            ),
            _time_slot("10:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER_SLOT,
        ],
        tags=("reminder", "finance"),
    ),
    AutomationBlueprint(
        key="habit-checkin",
        title="Habit check-in",
        description="A recurring nudge to keep a habit on track and reflect "
        "on whether you did it.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Nudge the user about their habit: {habit}. Ask whether they did it "
            "today, keep it warm and non-judgmental, and offer a one-line word "
            "of encouragement. One short message."
        ),
        slots=[
            BlueprintSlot(
                name="habit", type="text", label="Which habit?",
                default="20 minutes of reading",
            ),
            _time_slot("20:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER_SLOT,
        ],
        tags=("habit", "wellbeing"),
    ),
    AutomationBlueprint(
        key="learn-daily",
        title="Daily learning drip",
        description="One bite-sized lesson a day on a topic you want to learn, "
        "building progressively over time.",
        category="daily",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Teach the user one bite-sized lesson about: {topic}. Build on "
            "earlier lessons so it progresses rather than repeating. Keep it to "
            "a couple of short paragraphs with one concrete example, and end "
            "with a single question to check understanding."
        ),
        slots=[
            BlueprintSlot(
                name="topic", type="text", label="Learn about…",
                default="Spanish vocabulary",
            ),
            _time_slot("08:30"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER_SLOT,
        ],
        tags=("learning", "daily"),
    ),
]

# Index by key for O(1) lookup.
_CATALOG_BY_KEY: Dict[str, AutomationBlueprint] = {bp.key: bp for bp in CATALOG}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_blueprint(key: str) -> Optional[AutomationBlueprint]:
    """Look up a blueprint by key. Returns None if not found."""
    return _CATALOG_BY_KEY.get(key)


def list_blueprints(
    *, category: Optional[str] = None, tag: Optional[str] = None,
) -> List[AutomationBlueprint]:
    """List blueprints, optionally filtered by category or tag."""
    result = list(CATALOG)
    if category:
        result = [bp for bp in result if bp.category == category]
    if tag:
        result = [bp for bp in result if tag in bp.tags]
    return result


def blueprint_form_schema(blueprint: AutomationBlueprint) -> Dict[str, Any]:
    """Emit the JSON a form renderer needs for this blueprint."""
    return {
        "key": blueprint.key,
        "title": blueprint.title,
        "description": blueprint.description,
        "category": blueprint.category,
        "tags": list(blueprint.tags),
        "fields": [
            {
                "name": s.name,
                "type": s.type,
                "label": s.label,
                "default": s.default,
                "options": list(s.options),
                "optional": s.optional,
                "strict": s.strict,
                "help": s.help,
            }
            for s in blueprint.slots
        ],
    }


def blueprint_slash_command(
    blueprint: AutomationBlueprint,
    values: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the flattened ``/blueprint <key> slot=val …`` command string."""
    values = values or {}
    parts = [f"/blueprint {blueprint.key}"]
    for s in blueprint.slots:
        val = values.get(s.name, s.default)
        if val is None or val == "":
            if s.optional:
                continue
            val = ""
        sval = str(val)
        if s.type == "text" or " " in sval:
            sval = '"' + sval.replace('"', '\\"') + '"'
        parts.append(f"{s.name}={sval}")
    return " ".join(parts)


def blueprint_deeplink(
    blueprint: AutomationBlueprint,
    values: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the ``nia://blueprint/<key>?slot=val`` deep-link URL."""
    values = values or {}
    query = {}
    for s in blueprint.slots:
        val = values.get(s.name, s.default)
        if val not in (None, ""):
            query[s.name] = str(val)
    qs = ("?" + urlencode(query)) if query else ""
    return f"nia://blueprint/{quote(blueprint.key)}{qs}"


def _humanize_schedule(blueprint: AutomationBlueprint) -> str:
    """A short human-readable description of when a blueprint runs (defaults)."""
    sched = blueprint.schedule_template
    if sched.startswith("*/"):
        iv = next((s for s in blueprint.slots if s.name == "interval_min"), None)
        every = (iv.default if iv else None) or sched.split("/")[1].split()[0]
        return f"every {every} minutes"
    time_slot = next((s for s in blueprint.slots if s.type == "time"), None)
    when = time_slot.default if time_slot else None
    if "* * 1-5" in sched:
        return f"weekdays at {when}" if when else "every weekday"
    if "{dow}" in sched:
        day_slot = next(
            (s for s in blueprint.slots if s.name in ("day", "recurrence")), None
        )
        scope = (day_slot.default if day_slot else "") or ""
        if scope and when:
            return f"{scope} at {when}"
        return f"at {when}" if when else "on a schedule"
    if when:
        return f"daily at {when}"
    return "on a schedule"


def blueprint_catalog_entry(blueprint: AutomationBlueprint) -> Dict[str, Any]:
    """Unified serializable shape for a blueprint.

    Combines the form schema, the ready-to-paste slash command, the
    deep-link URL, and a human-readable schedule.
    """
    return {
        **blueprint_form_schema(blueprint),
        "slash_command": blueprint_slash_command(blueprint),
        "deeplink": blueprint_deeplink(blueprint),
        "humanized_schedule": _humanize_schedule(blueprint),
    }


# ---------------------------------------------------------------------------
# Schedule resolution
# ---------------------------------------------------------------------------


def _resolve_schedule(
    blueprint: AutomationBlueprint,
    values: Dict[str, Any],
) -> str:
    """Fill the schedule_template placeholders from resolved slot values."""
    sched = blueprint.schedule_template

    # A free-text `schedule` slot passes through verbatim.
    if "schedule" in values and values["schedule"]:
        return str(values["schedule"])

    repl: Dict[str, str] = {}

    # time → minute/hour
    time_val = values.get("time")
    if "{minute}" in sched or "{hour}" in sched:
        if not time_val:
            raise BlueprintFillError("a time is required")
        m = _TIME_RE.match(str(time_val).strip())
        if not m:
            raise BlueprintFillError(
                f"invalid time {time_val!r} — use HH:MM (24h)"
            )
        repl["hour"] = str(int(m.group(1)))
        repl["minute"] = str(int(m.group(2)))

    # weekday set → dow
    if "{dow}" in sched:
        if "recurrence" in values:
            preset = str(values.get("recurrence", "everyday")).lower()
            if preset not in WEEKDAY_PRESETS:
                raise BlueprintFillError(
                    f"unknown recurrence {preset!r} — one of {', '.join(WEEKDAY_PRESETS)}"
                )
            repl["dow"] = WEEKDAY_PRESETS[preset]
        elif "day" in values:
            day = str(values.get("day", "")).lower()
            if day not in _DAY_TO_DOW:
                raise BlueprintFillError(f"unknown day {day!r}")
            repl["dow"] = _DAY_TO_DOW[day]
        else:
            repl["dow"] = "*"

    # interval (minutes) for */N schedules
    if "{interval_min}" in sched:
        iv = str(values.get("interval_min", "")).strip()
        if not iv.isdigit() or int(iv) <= 0:
            raise BlueprintFillError(
                f"invalid interval {iv!r} — minutes as a positive integer"
            )
        repl["interval_min"] = iv

    # Any remaining {slot} placeholders are filled verbatim.
    for name in re.findall(r"\{(\w+)\}", sched):
        if name not in repl and name in values:
            repl[name] = str(values[name])

    try:
        return sched.format(**repl)
    except KeyError as e:
        raise BlueprintFillError(
            f"schedule template missing value for {e}"
        ) from e


# ---------------------------------------------------------------------------
# Blueprint fill
# ---------------------------------------------------------------------------


def fill_blueprint(
    blueprint: AutomationBlueprint,
    values: Dict[str, Any],
    *,
    origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate ``values`` and return cron job spec kwargs.

    Missing required (non-optional) slots raise BlueprintFillError naming
    the slot. Unknown slot names are rejected. Enum values are checked
    against their options. The result is passed straight to
    ``upsert_cron_job``.

    Args:
        blueprint: The blueprint to fill.
        values: Slot values from the user.
        origin: Optional origin dict (platform + chat_id) to stamp on
            the job so ``deliver=origin`` resolves correctly.

    Returns:
        A dict with keys: prompt, schedule, name, deliver, (optional)
        skills, (optional) origin.
    """
    known = {s.name for s in blueprint.slots}
    unknown = sorted(set(values) - known)
    if unknown:
        raise BlueprintFillError(
            f"unknown slot{'s' if len(unknown) > 1 else ''}: "
            f"{', '.join(unknown)} — valid: {', '.join(s.name for s in blueprint.slots)}"
        )

    resolved: Dict[str, Any] = {}
    for s in blueprint.slots:
        raw = values.get(s.name, s.default)
        if raw in (None, ""):
            if s.optional:
                continue
            raise BlueprintFillError(
                f"missing required value: {s.name} ({s.label})"
            )
        if (
            s.type == "enum"
            and s.strict
            and s.options
            and str(raw) not in {str(o) for o in s.options}
        ):
            raise BlueprintFillError(
                f"{s.name}={raw!r} not allowed — one of {', '.join(map(str, s.options))}"
            )
        resolved[s.name] = raw

    schedule = _resolve_schedule(blueprint, resolved)

    # Render the prompt with whatever slots it references.
    try:
        prompt = blueprint.prompt_template.format(**resolved)
    except KeyError as e:
        raise BlueprintFillError(
            f"blueprint prompt missing value for {e}"
        ) from e

    spec: Dict[str, Any] = {
        "prompt": prompt,
        "schedule": schedule,
        "name": blueprint.title,
        "deliver": resolved.get("deliver", blueprint.deliver_default),
    }
    if blueprint.skills:
        spec["skills"] = list(blueprint.skills)
    if origin is not None:
        spec["origin"] = origin
    return spec


__all__ = [
    "AutomationBlueprint",
    "BlueprintFillError",
    "BlueprintSlot",
    "WEEKDAY_PRESETS",
    "blueprint_catalog_entry",
    "blueprint_deeplink",
    "blueprint_form_schema",
    "blueprint_slash_command",
    "fill_blueprint",
    "get_blueprint",
    "list_blueprints",
]
