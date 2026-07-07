"""Tool for creating local cron-style jobs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from niaharness.services.cron import upsert_cron_job, validate_cron_expression
from niaharness.services.cron_delivery import validate_delivery_config
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class CronCreateToolInput(BaseModel):
    """Arguments for cron job creation."""

    name: str = Field(description="Unique cron job name")
    schedule: str = Field(
        description=(
            "Cron schedule expression (e.g. '*/5 * * * *' for every 5 minutes, "
            "'0 9 * * 1-5' for weekdays at 9am)"
        ),
    )
    command: str = Field(description="Shell command to run when triggered")
    cwd: str | None = Field(default=None, description="Optional working directory override")
    enabled: bool = Field(default=True, description="Whether the job is active")
    delivery: dict | None = Field(
        default=None,
        description=(
            "Optional delivery config. Send job results to email/webhook after "
            "execution. Shape: "
            '{"email": {"to": ["addr"], "smtp_host": "...", "smtp_port": 587, '
            '"smtp_user": "...", "smtp_password_env": "NIA_SMTP_PASSWORD", '
            '"use_tls": true, "subject": "..."}, "webhook": {"url": "https://...", '
            '"on_success": true, "on_failure": true}}. '
            "SMTP passwords are read from the env var named in smtp_password_env "
            "at delivery time (never stored in the job)."
        ),
    )


class CronCreateTool(BaseTool):
    """Create or replace a local cron job."""

    name = "cron_create"
    description = (
        "Create or replace a local cron job with a standard cron expression. "
        "Supports optional delivery to email/webhook after execution. "
        "Use 'nia cron start' to run the scheduler daemon."
    )
    input_model = CronCreateToolInput

    async def execute(
        self,
        arguments: CronCreateToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not validate_cron_expression(arguments.schedule):
            return ToolResult(
                output=(
                    f"Invalid cron expression: {arguments.schedule!r}\n"
                    "Use standard 5-field format: minute hour day month weekday\n"
                    "Examples: '*/5 * * * *' (every 5 min), '0 9 * * 1-5' (weekdays 9am)"
                ),
                is_error=True,
            )

        # Validate delivery config if provided.
        if arguments.delivery:
            errors = validate_delivery_config(arguments.delivery)
            if errors:
                return ToolResult(
                    output="Delivery config errors:\n  - " + "\n  - ".join(errors),
                    is_error=True,
                )

        job_dict = {
            "name": arguments.name,
            "schedule": arguments.schedule,
            "command": arguments.command,
            "cwd": arguments.cwd or str(context.cwd),
            "enabled": arguments.enabled,
        }
        if arguments.delivery:
            job_dict["delivery"] = arguments.delivery

        upsert_cron_job(job_dict)
        status = "enabled" if arguments.enabled else "disabled"
        delivery_note = ""
        if arguments.delivery:
            channels = []
            if "email" in arguments.delivery:
                channels.append("email")
            if "webhook" in arguments.delivery:
                channels.append("webhook")
            delivery_note = f" + delivery via {', '.join(channels)}"
        return ToolResult(
            output=f"Created cron job '{arguments.name}' [{arguments.schedule}] ({status}){delivery_note}"
        )
