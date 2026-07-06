"""Schema definitions for FileWriteTool."""

from pydantic import BaseModel, ConfigDict, Field


class FileWriteToolInput(BaseModel):
    """Input schema for FileWriteTool.

    Accepts both ``file_path`` (canonical) and ``path`` (alias used by some
    integration tests and external callers) for the same field.
    """

    model_config = ConfigDict(populate_by_name=True)

    file_path: str = Field(
        alias="path",
        description="The absolute path to the file to write",
    )
    content: str = Field(description="The content to write to the file")
