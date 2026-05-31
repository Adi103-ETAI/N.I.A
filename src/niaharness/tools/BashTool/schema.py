"""BashTool schema definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BashToolInput(BaseModel):
    """Input schema for BashTool."""

    command: str = Field(description="The command to execute")
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds (max 600)",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Clear, concise description of what this command does in active voice. "
            "Never use words like 'complex' or 'risk' in the description."
        ),
    )
    run_in_background: bool = Field(
        default=False,
        description="Set to true to run this command in the background. Use Read to read the output later.",
    )
