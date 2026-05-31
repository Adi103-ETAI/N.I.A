"""N.I.A State - Tracks overall system state.

Maintains the state of N.I.A's components and their interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SystemState(Enum):
    """Overall N.I.A system state."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentState:
    """State of a specific component."""
    name: str
    status: str
    last_active: float = 0.0
    error: str | None = None


class StateManager:
    """Manages N.I.A's overall state.

    Tracks:
    - System state
    - Component states
    - Active operations
    - Error states
    """

    def __init__(self) -> None:
        self._system_state: SystemState = SystemState.INITIALIZING
        self._components: dict[str, ComponentState] = {}
        self._active_operations: list[str] = []
        self._error_log: list[str] = []

    @property
    def system_state(self) -> SystemState:
        return self._system_state

    @system_state.setter
    def system_state(self, value: SystemState) -> None:
        self._system_state = value

    def update_component(self, name: str, status: str, error: str | None = None) -> None:
        """Update a component's state."""
        import time
        self._components[name] = ComponentState(
            name=name,
            status=status,
            last_active=time.time(),
            error=error,
        )

    def start_operation(self, operation: str) -> None:
        """Track an active operation."""
        self._active_operations.append(operation)
        if self._system_state == SystemState.READY:
            self._system_state = SystemState.PROCESSING

    def complete_operation(self, operation: str) -> None:
        """Mark an operation as complete."""
        if operation in self._active_operations:
            self._active_operations.remove(operation)
        if not self._active_operations and self._system_state == SystemState.PROCESSING:
            self._system_state = SystemState.READY

    def log_error(self, error: str) -> None:
        """Log an error."""
        self._error_log.append(error)
        self._system_state = SystemState.ERROR

    def clear_error(self) -> None:
        """Clear error state."""
        if self._system_state == SystemState.ERROR:
            self._system_state = SystemState.READY

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "system": self._system_state.value,
            "components": {
                name: {
                    "status": comp.status,
                    "error": comp.error,
                }
                for name, comp in self._components.items()
            },
            "active_operations": self._active_operations,
            "recent_errors": self._error_log[-5:] if self._error_log else [],
        }

    def is_ready(self) -> bool:
        """Check if the system is ready for new tasks."""
        return self._system_state == SystemState.READY

    def reset(self) -> None:
        """Reset state to initial."""
        self._system_state = SystemState.READY
        self._active_operations.clear()
        self._error_log.clear()
