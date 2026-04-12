"""Read state tracking for file staleness detection."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FileReadState:
    """Track when a file was read and its content."""
    
    content: str
    timestamp: float  # Unix timestamp
    offset: Optional[int] = None
    limit: Optional[int] = None
    is_partial_view: bool = False


class ReadStateTracker:
    """Track read state of files for staleness detection."""
    
    def __init__(self):
        self._state: Dict[str, FileReadState] = {}
    
    def record_read(
        self,
        file_path: str,
        content: str,
        timestamp: float,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        """Record that a file was read."""
        is_partial = offset is not None or limit is not None
        self._state[file_path] = FileReadState(
            content=content,
            timestamp=timestamp,
            offset=offset,
            limit=limit,
            is_partial_view=is_partial,
        )
    
    def get_state(self, file_path: str) -> Optional[FileReadState]:
        """Get the read state for a file."""
        return self._state.get(file_path)
    
    def update_after_edit(
        self,
        file_path: str,
        new_content: str,
        timestamp: float,
    ) -> None:
        """Update state after editing a file."""
        self._state[file_path] = FileReadState(
            content=new_content,
            timestamp=timestamp,
            offset=None,
            limit=None,
            is_partial_view=False,
        )
    
    def clear(self, file_path: str) -> None:
        """Clear read state for a file."""
        self._state.pop(file_path, None)
