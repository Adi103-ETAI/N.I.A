"""N.I.A Coordinator - Multi-agent orchestration.

Coordinates between N.I.A's brain and multiple OpenHarness workers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker agent."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Worker:
    """A worker agent that executes tasks."""
    id: str
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: str | None = None
    tasks_completed: int = 0


@dataclass
class CoordinationPlan:
    """A plan for coordinating multiple workers."""
    tasks: list[dict[str, Any]]
    parallel_groups: list[list[str]]  # Groups of tasks that can run in parallel
    sequential_chain: list[str]  # Tasks that must run in order


class Coordinator:
    """Coordinates task execution across multiple workers.

    When N.I.A's brain decides to do multiple things,
    the coordinator figures out:
    - Which tasks can run in parallel
    - Which tasks have dependencies
    - How to allocate workers
    """

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._task_assignments: dict[str, str] = {}  # task_id -> worker_id
        self._worker_counter: int = 0

    def create_worker(self, name: str | None = None) -> Worker:
        """Create a new worker agent."""
        self._worker_counter += 1
        worker_id = f"worker-{self._worker_counter}"
        worker = Worker(
            id=worker_id,
            name=name or f"Worker {self._worker_counter}",
        )
        self._workers[worker_id] = worker
        logger.info(f"Created worker: {worker_id} ({worker.name})")
        return worker

    def plan_execution(self, tasks: list[dict[str, Any]]) -> CoordinationPlan:
        """Plan how to execute a set of tasks."""
        # Analyze dependencies
        parallel_groups: list[list[str]] = []
        sequential_chain: list[str] = []

        # Simple heuristic: tasks without file conflicts can run in parallel
        file_access: dict[str, list[str]] = {}  # file -> list of task_ids

        for task in tasks:
            task_id = task.get("id", "")
            files = task.get("files_accessed", [])
            for f in files:
                if f not in file_access:
                    file_access[f] = []
                file_access[f].append(task_id)

        # Group tasks by file access conflicts
        conflicting_tasks: set[str] = set()
        for file_tasks in file_access.values():
            if len(file_tasks) > 1:
                conflicting_tasks.update(file_tasks)

        # Non-conflicting tasks can be parallel
        independent = [t for t in tasks if t.get("id") not in conflicting_tasks]
        if independent:
            parallel_groups.append([t.get("id", "") for t in independent])

        # Conflicting tasks must be sequential
        for task in tasks:
            if task.get("id") in conflicting_tasks:
                sequential_chain.append(task.get("id", ""))

        return CoordinationPlan(
            tasks=tasks,
            parallel_groups=parallel_groups,
            sequential_chain=sequential_chain,
        )

    def assign_task(self, task_id: str, worker_id: str | None = None) -> str:
        """Assign a task to a worker."""
        if worker_id is None:
            worker_id = self._get_idle_worker()

        if worker_id is None:
            worker_id = self.create_worker().id

        worker = self._workers[worker_id]
        worker.status = WorkerStatus.BUSY
        worker.current_task = task_id
        self._task_assignments[task_id] = worker_id

        logger.info(f"Assigned task {task_id} to worker {worker_id}")
        return worker_id

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        worker_id = self._task_assignments.get(task_id)
        if worker_id and worker_id in self._workers:
            worker = self._workers[worker_id]
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            worker.tasks_completed += 1
            del self._task_assignments[task_id]

    def get_worker_status(self) -> list[dict[str, Any]]:
        """Get status of all workers."""
        return [
            {
                "id": w.id,
                "name": w.name,
                "status": w.status.value,
                "current_task": w.current_task,
                "tasks_completed": w.tasks_completed,
            }
            for w in self._workers.values()
        ]

    def _get_idle_worker(self) -> str | None:
        """Find an idle worker."""
        for worker_id, worker in self._workers.items():
            if worker.status == WorkerStatus.IDLE:
                return worker_id
        return None
