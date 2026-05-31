"""Shared state between tools."""

from niaharness.tools.FileEditTool.read_state import ReadStateTracker

# Global read state tracker shared between FileReadTool and FileEditTool
_read_state_tracker = ReadStateTracker()

def get_read_state_tracker() -> ReadStateTracker:
    """Get the global read state tracker."""
    return _read_state_tracker
