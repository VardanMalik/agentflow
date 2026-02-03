"""Workflow and step state management."""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class Status(str, enum.Enum):
    """Unified status enum for workflows and steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self in (Status.COMPLETED, Status.FAILED, Status.CANCELLED, Status.SKIPPED)


# Valid state transitions for workflows
_WORKFLOW_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PENDING: {Status.RUNNING, Status.CANCELLED},
    Status.RUNNING: {Status.COMPLETED, Status.FAILED, Status.CANCELLED},
    Status.FAILED: {Status.RUNNING},  # retry
}

# Valid state transitions for steps
_STEP_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PENDING: {Status.RUNNING, Status.SKIPPED, Status.CANCELLED},
    Status.RUNNING: {Status.COMPLETED, Status.FAILED, Status.CANCELLED},
    Status.FAILED: {Status.RUNNING},  # retry
}


def can_transition(current: Status, target: Status, *, is_step: bool = False) -> bool:
    """Check whether a state transition is valid."""
    table = _STEP_TRANSITIONS if is_step else _WORKFLOW_TRANSITIONS
    return target in table.get(current, set())


@dataclass
class StepState:
    """Runtime state of a single workflow step."""

    id: UUID = field(default_factory=uuid4)
    step_order: int = 0
    agent_type: str = ""
    status: Status = Status.PENDING
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    parallel_group: str | None = None

    def mark_running(self) -> None:
        self.status = Status.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, output: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc)
        self.status = Status.COMPLETED
        self.output_data = output
        self.completed_at = now
        if self.started_at:
            self.duration_ms = int((now - self.started_at).total_seconds() * 1000)

    def mark_failed(self, error: str) -> None:
        now = datetime.now(timezone.utc)
        self.status = Status.FAILED
        self.error = error
        self.completed_at = now
        if self.started_at:
            self.duration_ms = int((now - self.started_at).total_seconds() * 1000)

    def mark_cancelled(self) -> None:
        self.status = Status.CANCELLED
        self.completed_at = datetime.now(timezone.utc)

    def reset_for_retry(self) -> None:
        self.status = Status.PENDING
        self.error = None
        self.output_data = None
        self.duration_ms = None
        self.started_at = None
        self.completed_at = None


@dataclass
class WorkflowState:
    """Runtime state of a complete workflow execution."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    status: Status = Status.PENDING
    config: dict[str, Any] = field(default_factory=dict)
    steps: list[StepState] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Listeners notified on every status change
    _listeners: list[asyncio.Queue[tuple[UUID, Status]]] = field(
        default_factory=list, repr=False,
    )

    # ---- status helpers ----

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == Status.COMPLETED)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == Status.FAILED)

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        return round(self.completed_steps / self.total_steps * 100, 1)

    # ---- state transitions ----

    def mark_running(self) -> None:
        self.status = Status.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self._emit(Status.RUNNING)

    def mark_completed(self) -> None:
        self.status = Status.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self._emit(Status.COMPLETED)

    def mark_failed(self) -> None:
        self.status = Status.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self._emit(Status.FAILED)

    def mark_cancelled(self) -> None:
        self.status = Status.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        for step in self.steps:
            if not step.status.is_terminal:
                step.mark_cancelled()
        self._emit(Status.CANCELLED)

    def reset_failed_steps(self) -> int:
        """Reset all failed steps to pending for retry. Returns count reset."""
        count = 0
        for step in self.steps:
            if step.status == Status.FAILED:
                step.reset_for_retry()
                count += 1
        return count

    # ---- event system ----

    def subscribe(self) -> asyncio.Queue[tuple[UUID, Status]]:
        """Return a queue that receives (workflow_id, new_status) on changes."""
        q: asyncio.Queue[tuple[UUID, Status]] = asyncio.Queue()
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[tuple[UUID, Status]]) -> None:
        self._listeners = [l for l in self._listeners if l is not q]

    def _emit(self, new_status: Status) -> None:
        for q in self._listeners:
            q.put_nowait((self.id, new_status))

    # ---- parallel grouping helpers ----

    def sequential_steps(self) -> list[list[StepState]]:
        """Group steps into execution batches.

        Steps with the same ``parallel_group`` run concurrently within a
        batch.  Steps without a group each form their own batch.
        """
        batches: list[list[StepState]] = []
        current_group: str | None = None
        current_batch: list[StepState] = []

        for step in sorted(self.steps, key=lambda s: s.step_order):
            if step.parallel_group and step.parallel_group == current_group:
                current_batch.append(step)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [step]
                current_group = step.parallel_group

        if current_batch:
            batches.append(current_batch)

        return batches

    # ---- serialisation ----

    def to_dict(self) -> dict[str, Any]:
        """Serialise state for persistence or API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status.value,
            "config": self.config,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "progress_pct": self.progress_pct,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "steps": [
                {
                    "id": str(s.id),
                    "step_order": s.step_order,
                    "agent_type": s.agent_type,
                    "status": s.status.value,
                    "error": s.error,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Restore state from a persisted dict."""
        steps = [
            StepState(
                id=UUID(s["id"]),
                step_order=s["step_order"],
                agent_type=s["agent_type"],
                status=Status(s["status"]),
                error=s.get("error"),
                duration_ms=s.get("duration_ms"),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            status=Status(data["status"]),
            config=data.get("config", {}),
            steps=steps,
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
        )
