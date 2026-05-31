"""FileReadTool schema definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileReadToolInput(BaseModel):
    """Input schema for FileReadTool."""

    file_path: str = Field(description="The absolute path to the file to read")
    offset: int | None = Field(
        default=None,
        description=(
            "The line number to start reading from (1-indexed). "
            "Only provide if the file is too large to read at once"
        ),
    )
    limit: int | None = Field(
        default=None,
        description="The number of lines to read. Only provide if the file is too large to read at once.",
    )
    pages: str | None = Field(
        default=None,
        description=(
            'Page range for PDF files (e.g., "1-5", "3", "10-20"). '
            "Only applicable to PDF files. Maximum 50 pages per request."
        ),
    )


class FileReadToolOutput(BaseModel):
    """Output schema for FileReadTool."""

    file_path: str = Field(description="The path to the file that was read")
    content: str = Field(description="The content of the file with line numbers")
    num_lines: int = Field(description="Number of lines in the returned content")
    start_line: int = Field(description="The starting line number")
    total_lines: int = Field(description="Total number of lines in the file")


class ImageOutput(BaseModel):
    """Output for image files."""

    type: str = "image"
    base64: str = Field(description="Base64-encoded image data")
    media_type: str = Field(description="MIME type of the image")
    original_size: int = Field(description="Original file size in bytes")
    width: int | None = None
    height: int | None = None


class NotebookOutput(BaseModel):
    """Output for Jupyter notebook files."""

    type: str = "notebook"
    file_path: str
    cells: list[dict]


class PDFOutput(BaseModel):
    """Output for PDF files."""

    type: str = "pdf"
    file_path: str
    base64: str
    original_size: int
