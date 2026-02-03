"""Pydantic schemas for API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Workflow schemas
# ---------------------------------------------------------------------------

class WorkflowStepCreate(BaseModel):
    """Schema for defining a single step when creating a workflow."""

    agent_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of agent to execute this step.",
        json_schema_extra={"examples": ["llm"]},
    )
    input_data: dict[str, Any] | None = Field(
        default=None,
        description="Input payload passed to the agent.",
        json_schema_extra={"examples": [{"prompt": "Summarise the document."}]},
    )


class WorkflowCreate(BaseModel):
    """Request body for creating a new workflow."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable workflow name.",
        json_schema_extra={"examples": ["Data Processing Pipeline"]},
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Optional description of what the workflow does.",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary configuration passed to the workflow engine.",
    )
    steps: list[WorkflowStepCreate] = Field(
        ...,
        min_length=1,
        description="Ordered list of steps to execute.",
    )


class WorkflowStepResponse(BaseModel):
    """Response representation of a workflow step."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    step_order: int
    agent_type: str
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    status: str
    error_message: str | None = None
    duration_ms: int | None = None


class WorkflowResponse(BaseModel):
    """Full workflow response with steps."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    status: str
    config: dict[str, Any] | None = None
    created_by: str | None = None
    steps: list[WorkflowStepResponse] = []
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowSummary(BaseModel):
    """Lightweight workflow representation for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowStatusResponse(BaseModel):
    """Workflow execution progress."""

    id: UUID
    status: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Agent schemas
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """Full agent response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    type: str
    description: str
    config: dict[str, Any] | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AgentSummary(BaseModel):
    """Lightweight agent representation for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    type: str
    is_active: bool


class AgentTestRequest(BaseModel):
    """Request body for testing an agent."""

    input_data: dict[str, Any] = Field(
        ...,
        description="Sample input payload to send to the agent.",
        json_schema_extra={"examples": [{"prompt": "Hello, world!"}]},
    )
    model: str | None = Field(
        default=None,
        description="Override the default LLM model for this test.",
        json_schema_extra={"examples": ["gpt-4o"]},
    )


class AgentTestResponse(BaseModel):
    """Result of an agent test execution."""

    agent_id: UUID
    status: str
    output: Any = None
    error: str | None = None
    duration_ms: int | None = None
    tokens_used: int | None = None


# ---------------------------------------------------------------------------
# Health schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Application health check response."""

    status: str = Field(description="Overall health status.")
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Detailed readiness check with dependency status."""

    status: str
    checks: dict[str, ServiceCheck]


class ServiceCheck(BaseModel):
    """Health status of an individual service dependency."""

    status: str = Field(description="'ok' or 'error'.")
    latency_ms: float | None = Field(
        default=None,
        description="Round-trip latency in milliseconds.",
    )
    error: str | None = None


# Rebuild ReadinessResponse now that ServiceCheck is defined
ReadinessResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""

    items: list[Any]
    total: int
    skip: int
    limit: int
