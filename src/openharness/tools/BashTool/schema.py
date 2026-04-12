"""Schema."""
from pydantic import BaseModel, Field


class BashToolInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int | None = Field(default=None, description="Optional timeout in seconds (max 600)")
