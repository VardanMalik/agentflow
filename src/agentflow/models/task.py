"""Task database model."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentflow.models.base import Base


class TaskPriority(str, enum.Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, enum.Enum):
    """Possible states of a task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(Base):
    """A discrete unit of work within a workflow."""

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_workflow_status", "workflow_id", "status"),
        Index("ix_tasks_priority_status", "priority", "status"),
        Index("ix_tasks_deadline", "deadline"),
    )

    workflow_id: Mapped[UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(TaskPriority, name="task_priority", native_enum=False),
        default=TaskPriority.MEDIUM,
        index=True,
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, name="task_status", native_enum=False),
        default=TaskStatus.PENDING,
        index=True,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )

    # Relationships
    workflow: Mapped["Workflow"] = relationship(back_populates="tasks")  # noqa: F821

    @property
    def can_retry(self) -> bool:
        """Check whether the task has retries remaining."""
        return self.retry_count < self.max_retries

    def record_retry(self) -> None:
        """Increment retry count and reset status to pending."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
