"""N.I.A Context - Situational awareness.

Tracks time, user state, environment, and provides contextual intelligence.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TimeOfDay(Enum):
    """Time periods for context."""
    MORNING = "morning"  # 6-12
    AFTERNOON = "afternoon"  # 12-17
    EVENING = "evening"  # 17-21
    NIGHT = "night"  # 21-6


class UserState(Enum):
    """Detected user state."""
    ACTIVE = "active"
    IDLE = "idle"
    FOCUSED = "focused"
    FRUSTRATED = "frustrated"
    EXPLORING = "exploring"


@dataclass
class Environment:
    """Current environment context."""
    working_directory: str = ""
    platform: str = ""
    shell: str = ""
    python_version: str = ""
    git_branch: str | None = None
    project_type: str | None = None  # "python", "node", "rust", etc.


@dataclass
class SessionContext:
    """Current session context."""
    start_time: float = field(default_factory=time.time)
    message_count: int = 0
    tasks_completed: int = 0
    tasks_pending: int = 0
    errors_encountered: int = 0
    last_activity: float = field(default_factory=time.time)


class Context:
    """N.I.A's context awareness system.

    Tracks:
    - Time of day
    - User state (active, idle, focused)
    - Environment (cwd, platform, project)
    - Session progress
    """

    def __init__(self) -> None:
        self._environment = Environment()
        self._session = SessionContext()
        self._user_name: str | None = None
        self._custom_context: dict[str, Any] = {}

    @property
    def time_of_day(self) -> TimeOfDay:
        """Get current time of day."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    @property
    def user_state(self) -> UserState:
        """Infer user state from activity patterns."""
        idle_time = time.time() - self._session.last_activity

        if idle_time > 300:  # 5 minutes
            return UserState.IDLE
        elif self._session.errors_encountered > 3:
            return UserState.FRUSTRATED
        elif self._session.tasks_completed > 5:
            return UserState.FOCUSED
        elif self._session.message_count < 3:
            return UserState.EXPLORING
        else:
            return UserState.ACTIVE

    def detect_environment(self, cwd: str | None = None) -> Environment:
        """Detect the current environment."""
        self._environment.working_directory = cwd or os.getcwd()
        self._environment.platform = os.name

        # Detect shell
        self._environment.shell = os.environ.get("SHELL", "unknown")

        # Detect Python version
        import sys
        self._environment.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Detect git branch
        self._environment.git_branch = self._detect_git_branch()

        # Detect project type
        self._environment.project_type = self._detect_project_type()

        return self._environment

    def track_activity(self) -> None:
        """Update last activity timestamp."""
        self._session.last_activity = time.time()
        self._session.message_count += 1

    def task_completed(self) -> None:
        """Record a task completion."""
        self._session.tasks_completed += 1

    def task_pending(self) -> None:
        """Record a pending task."""
        self._session.tasks_pending += 1

    def error_occurred(self) -> None:
        """Record an error."""
        self._session.errors_encountered += 1

    def set_user_name(self, name: str) -> None:
        """Store user's name for personalization."""
        self._user_name = name

    def set_custom(self, key: str, value: Any) -> None:
        """Set custom context value."""
        self._custom_context[key] = value

    def get_custom(self, key: str) -> Any:
        """Get custom context value."""
        return self._custom_context.get(key)

    def get_summary(self) -> str:
        """Get a summary of current context."""
        parts = [
            f"Time: {self.time_of_day.value}",
            f"User: {self.user_state.value}",
            f"Session: {self._session.message_count} messages, "
            f"{self._session.tasks_completed} tasks done",
        ]

        if self._environment.project_type:
            parts.append(f"Project: {self._environment.project_type}")

        if self._environment.git_branch:
            parts.append(f"Branch: {self._environment.git_branch}")

        return " | ".join(parts)

    def get_full_context(self) -> dict[str, Any]:
        """Get full context as a dictionary."""
        return {
            "time_of_day": self.time_of_day.value,
            "user_state": self.user_state.value,
            "environment": {
                "working_directory": self._environment.working_directory,
                "platform": self._environment.platform,
                "shell": self._environment.shell,
                "python_version": self._environment.python_version,
                "git_branch": self._environment.git_branch,
                "project_type": self._environment.project_type,
            },
            "session": {
                "message_count": self._session.message_count,
                "tasks_completed": self._session.tasks_completed,
                "tasks_pending": self._session.tasks_pending,
                "errors_encountered": self._session.errors_encountered,
            },
            "user_name": self._user_name,
            "custom": self._custom_context,
        }

    def _detect_git_branch(self) -> str | None:
        """Detect current git branch."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _detect_project_type(self) -> str | None:
        """Detect project type from files in working directory."""
        cwd = Path(self._environment.working_directory) if self._environment.working_directory else Path.cwd()

        indicators = {
            "pyproject.toml": "python",
            "setup.py": "python",
            "requirements.txt": "python",
            "package.json": "node",
            "Cargo.toml": "rust",
            "go.mod": "go",
            "pom.xml": "java",
            "build.gradle": "java",
            "Makefile": "c/cpp",
            "CMakeLists.txt": "c/cpp",
        }

        for filename, project_type in indicators.items():
            if (cwd / filename).exists():
                return project_type

        return None


# Need to import Path for _detect_project_type
from pathlib import Path
