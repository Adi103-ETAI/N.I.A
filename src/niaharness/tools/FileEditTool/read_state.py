"""FileEditTool read state tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FileReadState:
    """Tracks the state of a file that has been read."""

    content: str
    timestamp: float
    offset: Optional[int] = None
    limit: Optional[int] = None
    is_partial_view: bool = False


class ReadStateTracker:
    """Tracks read state for files to detect concurrent modifications."""

    def __init__(self) -> None:
        self._states: dict[str, FileReadState] = {}

    def get_state(self, file_path: str) -> Optional[FileReadState]:
        """Get the read state for a file."""
        return self._states.get(file_path)

    def update_after_read(
        self,
        file_path: str,
        content: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        """Update read state after reading a file."""
        timestamp = os.path.getmtime(file_path) if os.path.exists(file_path) else 0.0
        self._states[file_path] = FileReadState(
            content=content,
            timestamp=timestamp,
            offset=offset,
            limit=limit,
            is_partial_view=offset is not None or limit is not None,
        )

    def update_after_edit(
        self,
        file_path: str,
        content: str,
        timestamp: float,
    ) -> None:
        """Update read state after editing a file."""
        self._states[file_path] = FileReadState(
            content=content,
            timestamp=timestamp,
            offset=None,
            limit=None,
            is_partial_view=False,
        )

    def clear(self, file_path: Optional[str] = None) -> None:
        """Clear read state for a file or all files."""
        if file_path:
            self._states.pop(file_path, None)
        else:
            self._states.clear()


# Global singleton
_tracker: Optional[ReadStateTracker] = None


def get_read_state_tracker() -> ReadStateTracker:
    """Get the global read state tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ReadStateTracker()
    return _tracker
