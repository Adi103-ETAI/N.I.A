"""Schema definitions for FileWriteTool."""

from pydantic import BaseModel, Field


class FileWriteToolInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")
