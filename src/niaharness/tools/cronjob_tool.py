"""Cronjob multi-op tool — manage cron jobs with a single tool.

Adapted from Hermes Agent's tools/cronjob_tools.py.

Collapses NIA's 4 cron tools (cron_create, cron_list, cron_delete,
cron_toggle) into a single tool with 7 operations:

- ``create``  — create or replace a cron job
- ``list``    — list all cron jobs
- ``update``  — update an existing job's schedule/command/delivery
- ``pause``   — disable a job (set enabled=False)
- ``resume``  — enable a job (set enabled=True)
- ``remove``  — delete a job
- ``run``     — trigger a job immediately

The old 4 tools remain registered for backward compatibility, but the
``cronjob`` tool is the recommended interface — it matches Hermes's
architecture and reduces the tool-schema token cost per API call.

Reference: Hermes Agent's tools/cronjob_tools.py (action enum: create,
list, update, pause, resume, remove, run).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from niaharness.services.cron import (
    delete_cron_job,
    get_cron_job,
    load_cron_jobs,
    set_job_enabled,
    upsert_cron_job,
    validate_cron_expression,
)
from niaharness.services.cron_delivery import validate_delivery_config
from niaharness.services.cron_scheduler import execute_job, append_history
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class CronjobToolInput(BaseModel):
    """Arguments for the cronjob tool."""

    action: Literal["create", "list", "update", "pause", "resume", "remove", "run"] = Field(
        description=(
            "The cron management operation. create: create/replace a job. "
            "list: list all jobs. update: update a job. pause: disable. "
            "resume: enable. remove: delete. run: trigger immediately."
        )
    )
    name: str | None = Field(
        default=None,
        description="Unique cron job name (required for all actions except 'list').",
    )
    schedule: str | None = Field(
        default=None,
        description="5-field cron expression (e.g. '0 9 * * 1-5' for weekdays at 9am). Required for 'create' and 'update'.",
    )
    command: str | None = Field(
        default=None,
        description="Shell command to run (for 'create' and 'update').",
    )
    cwd: str | None = Field(default=None, description="Working directory override.")
    enabled: bool = Field(default=True, description="Whether the job is active (for 'create').")
    delivery: dict[str, Any] | None = Field(
        default=None,
        description="Delivery config (email/webhook). Same shape as cron_create_tool.",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CronjobTool(BaseTool):
    """Manage cron jobs with a single multi-operation tool."""

    name = "cronjob"
    description = (
        "Manage scheduled cron jobs: create, list, update, pause, resume, "
        "remove, or run immediately. Replaces the 4 separate cron_* tools "
        "with a unified interface. Supports email/webhook delivery."
    )
    input_model = CronjobToolInput

    def is_read_only(self, arguments: CronjobToolInput) -> bool:
        return arguments.action == "list"

    async def execute(self, arguments: CronjobToolInput, context: ToolExecutionContext) -> ToolResult:
        action = arguments.action

        if action == "list":
            return self._list(context)
        if action == "create":
            return self._create(arguments, context)
        if action == "update":
            return self._update(arguments, context)
        if action == "pause":
            return self._toggle(arguments, enabled=False)
        if action == "resume":
            return self._toggle(arguments, enabled=True)
        if action == "remove":
            return self._remove(arguments)
        if action == "run":
            return await self._run(arguments)

        return ToolResult(output=f"Unknown action: {action}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _list(self, context: ToolExecutionContext) -> ToolResult:
        """List all cron jobs."""
        jobs = load_cron_jobs()
        if not jobs:
            return ToolResult(output="No cron jobs configured.")
        lines = [f"Cron jobs ({len(jobs)}):"]
        for job in jobs:
            enabled = "on" if job.get("enabled", True) else "off"
            schedule = job.get("schedule", "?")
            name = job.get("name", "?")
            cmd_preview = (job.get("command") or "")[:60]
            lines.append(f"  [{enabled}] {name}  {schedule}  cmd: {cmd_preview}")
        return ToolResult(output="\n".join(lines), metadata={"count": len(jobs)})

    def _create(self, args: CronjobToolInput, context: ToolExecutionContext) -> ToolResult:
        """Create or replace a cron job."""
        if not args.name:
            return ToolResult(output="create requires 'name'", is_error=True)
        if not args.schedule:
            return ToolResult(output="create requires 'schedule'", is_error=True)
        if not args.command:
            return ToolResult(output="create requires 'command'", is_error=True)
        if not validate_cron_expression(args.schedule):
            return ToolResult(
                output=f"Invalid cron expression: {args.schedule!r}. Use 5-field format: minute hour day month weekday",
                is_error=True,
            )
        if args.delivery:
            errors = validate_delivery_config(args.delivery)
            if errors:
                return ToolResult(output="Delivery config errors:\n  - " + "\n  - ".join(errors), is_error=True)

        job_dict: dict[str, Any] = {
            "name": args.name,
            "schedule": args.schedule,
            "command": args.command,
            "cwd": args.cwd or str(context.cwd),
            "enabled": args.enabled,
        }
        if args.delivery:
            job_dict["delivery"] = args.delivery

        upsert_cron_job(job_dict)
        status = "enabled" if args.enabled else "disabled"
        delivery_note = ""
        if args.delivery:
            channels = [k for k in args.delivery if k in ("email", "webhook")]
            delivery_note = f" + delivery via {', '.join(channels)}"
        return ToolResult(output=f"Created cron job '{args.name}' [{args.schedule}] ({status}){delivery_note}")

    def _update(self, args: CronjobToolInput, context: ToolExecutionContext) -> ToolResult:
        """Update an existing cron job."""
        if not args.name:
            return ToolResult(output="update requires 'name'", is_error=True)
        existing = get_cron_job(args.name)
        if existing is None:
            return ToolResult(output=f"Job not found: {args.name}. Use 'create' to make a new one.", is_error=True)

        # Merge updates into existing job.
        job_dict = dict(existing)
        if args.schedule:
            if not validate_cron_expression(args.schedule):
                return ToolResult(output=f"Invalid cron expression: {args.schedule!r}", is_error=True)
            job_dict["schedule"] = args.schedule
        if args.command:
            job_dict["command"] = args.command
        if args.cwd:
            job_dict["cwd"] = args.cwd
        if args.delivery:
            errors = validate_delivery_config(args.delivery)
            if errors:
                return ToolResult(output="Delivery config errors:\n  - " + "\n  - ".join(errors), is_error=True)
            job_dict["delivery"] = args.delivery

        upsert_cron_job(job_dict)
        return ToolResult(output=f"Updated cron job '{args.name}'")

    def _toggle(self, args: CronjobToolInput, *, enabled: bool) -> ToolResult:
        """Pause or resume a cron job."""
        if not args.name:
            return ToolResult(output=f"{('pause' if not enabled else 'resume')} requires 'name'", is_error=True)
        if not set_job_enabled(args.name, enabled):
            return ToolResult(output=f"Job not found: {args.name}", is_error=True)
        action = "paused" if not enabled else "resumed"
        return ToolResult(output=f"Job '{args.name}' {action}")

    def _remove(self, args: CronjobToolInput) -> ToolResult:
        """Delete a cron job."""
        if not args.name:
            return ToolResult(output="remove requires 'name'", is_error=True)
        if not delete_cron_job(args.name):
            return ToolResult(output=f"Job not found: {args.name}", is_error=True)
        return ToolResult(output=f"Removed cron job '{args.name}'")

    async def _run(self, args: CronjobToolInput) -> ToolResult:
        """Trigger a job immediately."""
        if not args.name:
            return ToolResult(output="run requires 'name'", is_error=True)
        job = get_cron_job(args.name)
        if job is None:
            return ToolResult(output=f"Job not found: {args.name}", is_error=True)
        # Execute the job now.
        entry = await execute_job(job)
        append_history(entry)
        status = entry.get("status", "unknown")
        stdout = (entry.get("stdout") or "")[:500]
        return ToolResult(
            output=f"Ran job '{args.name}' — status: {status}\n\nstdout:\n{stdout}",
            metadata=entry,
        )
