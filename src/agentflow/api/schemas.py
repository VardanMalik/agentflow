"""Pydantic schemas for API request and response models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic wrapper for paginated list responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"items": [], "total": 0, "page": 1, "page_size": 20}]
        }
    )

    items: list[T]
    total: int
    page: int
    page_size: int


class WebSocketMessage(BaseModel):
    """Schema for messages pushed over a WebSocket connection."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "workflow.started",
                    "payload": {"workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ]
        }
    )

    type: str = Field(description="Event type, e.g. 'workflow.started'.")
    payload: dict[str, Any] = Field(description="Event-specific data.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was generated.",
    )


# ---------------------------------------------------------------------------
# Workflow schemas
# ---------------------------------------------------------------------------


class WorkflowStepCreate(BaseModel):
    """Schema for defining a single step when creating a workflow."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"agent_type": "llm", "input_data": {"prompt": "Summarise."}}]
        }
    )

    agent_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of agent to execute this step.",
    )
    input_data: dict[str, Any] | None = Field(
        default=None,
        description="Input payload passed to the agent.",
    )


class WorkflowCreate(BaseModel):
    """Request body for creating a new workflow."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Data Processing Pipeline",
                    "description": "Processes and summarises incoming data.",
                    "config": {},
                    "steps": [{"agent_type": "llm", "input_data": {"prompt": "Summarise."}}],
                }
            ]
        }
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable workflow name.",
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

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "name": "Data Processing Pipeline",
                    "description": "Processes and summarises incoming data.",
                    "status": "completed",
                    "config": None,
                    "created_by": None,
                    "steps": [],
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:01:00Z",
                    "started_at": "2024-01-01T00:00:01Z",
                    "completed_at": "2024-01-01T00:01:00Z",
                }
            ]
        },
    )

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


# WorkflowResponse already includes steps; alias for semantic clarity.
WorkflowDetailResponse = WorkflowResponse


class WorkflowSummary(BaseModel):
    """Lightweight workflow representation for list endpoints."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "name": "Data Processing Pipeline",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00Z",
                    "started_at": "2024-01-01T00:00:01Z",
                    "completed_at": "2024-01-01T00:01:00Z",
                }
            ]
        },
    )

    id: UUID
    name: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowStatusResponse(BaseModel):
    """Workflow execution progress."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "status": "running",
                    "total_steps": 3,
                    "completed_steps": 1,
                    "failed_steps": 0,
                    "started_at": "2024-01-01T00:00:01Z",
                    "completed_at": None,
                }
            ]
        }
    )

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

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "name": "My LLM Agent",
                    "type": "llm",
                    "description": "Processes natural language prompts.",
                    "config": {"model": "gpt-4o"},
                    "is_active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        },
    )

    id: UUID
    name: str
    type: str
    description: str
    config: dict[str, Any] | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


# AgentResponse already contains full detail; alias for semantic clarity.
AgentDetailResponse = AgentResponse


class AgentSummary(BaseModel):
    """Lightweight agent representation for list endpoints."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "name": "My LLM Agent",
                    "type": "llm",
                    "is_active": True,
                }
            ]
        },
    )

    id: UUID
    name: str
    type: str
    is_active: bool


# Typed alias for paginated agent list responses.
AgentListResponse = PaginatedResponse[AgentSummary]


class AgentTypeInfo(BaseModel):
    """Metadata describing a registered agent type."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "llm",
                    "description": "Large Language Model agent for text generation.",
                    "config_schema": {"model": "string", "temperature": "number"},
                }
            ]
        }
    )

    type: str
    description: str
    config_schema: dict[str, Any] | None = None


class AgentExecutionResponse(BaseModel):
    """A single recorded agent execution."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "agent_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "workflow_id": None,
                    "status": "completed",
                    "input_data": {"prompt": "Hello"},
                    "output_data": {"result": "Hi there!"},
                    "error_message": None,
                    "duration_ms": 123,
                    "tokens_used": 50,
                    "started_at": "2024-01-01T00:00:00Z",
                    "completed_at": "2024-01-01T00:00:01Z",
                }
            ]
        }
    )

    id: UUID
    agent_id: UUID
    workflow_id: UUID | None = None
    status: str
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error_message: str | None = None
    duration_ms: int | None = None
    tokens_used: int | None = None
    started_at: datetime
    completed_at: datetime | None = None


class AgentTestRequest(BaseModel):
    """Request body for testing an agent."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"input_data": {"prompt": "Hello, world!"}, "model": "gpt-4o"}]
        }
    )

    input_data: dict[str, Any] = Field(
        ...,
        description="Sample input payload to send to the agent.",
    )
    model: str | None = Field(
        default=None,
        description="Override the default LLM model for this test.",
    )


class AgentTestResponse(BaseModel):
    """Result of an agent test execution."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "agent_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "status": "completed",
                    "output": {"message": "Hello!"},
                    "error": None,
                    "duration_ms": 45,
                    "tokens_used": 20,
                }
            ]
        }
    )

    agent_id: UUID
    status: str
    output: Any = None
    error: str | None = None
    duration_ms: int | None = None
    tokens_used: int | None = None


# ---------------------------------------------------------------------------
# Fault tolerance schemas
# ---------------------------------------------------------------------------


class DLQEntryResponse(BaseModel):
    """A dead-letter queue entry."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "queue_name": "workflow.execute",
                    "payload": {"workflow_id": "..."},
                    "error": "Connection timeout",
                    "retry_count": 3,
                    "created_at": "2024-01-01T00:00:00Z",
                    "last_attempted_at": "2024-01-01T00:05:00Z",
                }
            ]
        }
    )

    id: UUID
    queue_name: str
    payload: dict[str, Any]
    error: str
    retry_count: int
    created_at: datetime
    last_attempted_at: datetime | None = None


class CircuitBreakerStatus(BaseModel):
    """Status of a single circuit breaker."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "database",
                    "state": "closed",
                    "failure_count": 0,
                    "success_count": 42,
                    "last_failure_at": None,
                }
            ]
        }
    )

    name: str
    state: str = Field(description="'closed', 'open', or 'half_open'.")
    failure_count: int
    success_count: int
    last_failure_at: datetime | None = None


class FaultToleranceStatusResponse(BaseModel):
    """Aggregated fault tolerance system status."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "circuit_breakers": [
                        {
                            "name": "database",
                            "state": "closed",
                            "failure_count": 0,
                            "success_count": 42,
                            "last_failure_at": None,
                        }
                    ],
                    "dlq_size": 0,
                    "active_bulkheads": 2,
                }
            ]
        }
    )

    circuit_breakers: list[CircuitBreakerStatus] = []
    dlq_size: int
    active_bulkheads: int


# ---------------------------------------------------------------------------
# Health schemas
# ---------------------------------------------------------------------------


class ServiceCheck(BaseModel):
    """Health status of an individual service dependency."""

    status: str = Field(description="'ok' or 'error'.")
    latency_ms: float | None = Field(
        default=None,
        description="Round-trip latency in milliseconds.",
    )
    error: str | None = None


class HealthResponse(BaseModel):
    """Application health check response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "healthy", "version": "0.1.0", "environment": "development"}]
        }
    )

    status: str = Field(description="Overall health status.")
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Detailed readiness check with dependency status."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "checks": {
                        "database": {"status": "ok", "latency_ms": 1.23, "error": None},
                        "redis": {"status": "ok", "latency_ms": 0.5, "error": None},
                    },
                }
            ]
        }
    )

    status: str
    checks: dict[str, ServiceCheck]
