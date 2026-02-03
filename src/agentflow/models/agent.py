"""Agent and agent execution database models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentflow.models.base import Base


class AgentType(str, enum.Enum):
    """Supported agent types."""

    LLM = "llm"
    TOOL = "tool"
    ROUTER = "router"
    VALIDATOR = "validator"
    CUSTOM = "custom"


class ExecutionStatus(str, enum.Enum):
    """Possible states of an agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class Agent(Base):
    """A registered AI agent that can execute workflow steps."""

    __tablename__ = "agents"
    __table_args__ = (
        Index("ix_agents_type_active", "type", "is_active"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    type: Mapped[AgentType] = mapped_column(
        Enum(AgentType, name="agent_type", native_enum=False),
        default=AgentType.LLM,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Relationships
    executions: Mapped[list[AgentExecution]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class AgentExecution(Base):
    """A record of a single agent execution against a workflow step."""

    __tablename__ = "agent_executions"
    __table_args__ = (
        Index("ix_executions_agent_status", "agent_id", "status"),
        Index("ix_executions_step", "workflow_step_id"),
        Index("ix_executions_started_at", "started_at"),
    )

    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workflow_step_id: Mapped[UUID] = mapped_column(
        ForeignKey("workflow_steps.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[ExecutionStatus] = mapped_column(
        Enum(ExecutionStatus, name="execution_status", native_enum=False),
        default=ExecutionStatus.PENDING,
        index=True,
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_model: Mapped[str | None] = mapped_column(String(100), default=None)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)

    # Relationships
    agent: Mapped[Agent] = relationship(back_populates="executions")
    workflow_step: Mapped["WorkflowStep"] = relationship(  # noqa: F821
        back_populates="executions",
    )

    @property
    def total_tokens(self) -> int:
        """Return total token usage for this execution."""
        return self.input_tokens + self.output_tokens

    @property
    def duration_ms(self) -> int | None:
        """Calculate execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
