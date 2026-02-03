"""Workflow and workflow step database models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentflow.models.base import Base


class WorkflowStatus(str, enum.Enum):
    """Possible states of a workflow."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, enum.Enum):
    """Possible states of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class Workflow(Base):
    """A workflow definition with its execution state."""

    __tablename__ = "workflows"
    __table_args__ = (
        Index("ix_workflows_status_created", "status", "created_at"),
        Index("ix_workflows_created_by", "created_by"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(WorkflowStatus, name="workflow_status", native_enum=False),
        default=WorkflowStatus.PENDING,
        index=True,
    )
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    created_by: Mapped[str | None] = mapped_column(String(255), default=None)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )

    # Relationships
    steps: Mapped[list[WorkflowStep]] = relationship(
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="WorkflowStep.step_order",
        lazy="selectin",
    )
    tasks: Mapped[list["Task"]] = relationship(  # noqa: F821
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def mark_running(self) -> None:
        """Transition workflow to running state."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = func.now()

    def mark_completed(self) -> None:
        """Transition workflow to completed state."""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = func.now()

    def mark_failed(self) -> None:
        """Transition workflow to failed state."""
        self.status = WorkflowStatus.FAILED
        self.completed_at = func.now()


class WorkflowStep(Base):
    """A single step within a workflow execution."""

    __tablename__ = "workflow_steps"
    __table_args__ = (
        Index("ix_steps_workflow_order", "workflow_id", "step_order", unique=True),
        Index("ix_steps_status", "status"),
    )

    workflow_id: Mapped[UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_type: Mapped[str] = mapped_column(String(100), nullable=False)
    input_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    output_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    status: Mapped[StepStatus] = mapped_column(
        Enum(StepStatus, name="step_status", native_enum=False),
        default=StepStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    duration_ms: Mapped[int | None] = mapped_column(Integer, default=None)

    # Relationships
    workflow: Mapped[Workflow] = relationship(back_populates="steps")
    executions: Mapped[list["AgentExecution"]] = relationship(  # noqa: F821
        back_populates="workflow_step",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
