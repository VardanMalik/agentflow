"""Agent management endpoints."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from agentflow.api.schemas import (
    AgentDetailResponse,
    AgentExecutionResponse,
    AgentListResponse,
    AgentResponse,
    AgentSummary,
    AgentTestRequest,
    AgentTestResponse,
    AgentTypeInfo,
    PaginatedResponse,
)

logger = structlog.get_logger()

router = APIRouter(tags=["agents"])

# ---------------------------------------------------------------------------
# In-memory stores (placeholder until service layer is wired)
# ---------------------------------------------------------------------------
_agents: dict[UUID, dict] = {}
_executions: dict[UUID, list[dict]] = {}  # agent_id → execution records

# ---------------------------------------------------------------------------
# Static registry of built-in agent types
# ---------------------------------------------------------------------------
_AGENT_TYPES: list[AgentTypeInfo] = [
    AgentTypeInfo(
        type="llm",
        description="Large Language Model agent for text generation and reasoning.",
        config_schema={"model": "string", "temperature": "number", "max_tokens": "integer"},
    ),
    AgentTypeInfo(
        type="tool",
        description="Agent that executes deterministic tool calls (e.g. API requests, calculations).",
        config_schema={"tool_name": "string", "timeout_ms": "integer"},
    ),
    AgentTypeInfo(
        type="retrieval",
        description="Retrieval-augmented agent that fetches context from a vector store.",
        config_schema={"index_name": "string", "top_k": "integer"},
    ),
    AgentTypeInfo(
        type="code",
        description="Agent that generates or executes code in a sandboxed environment.",
        config_schema={"language": "string", "timeout_ms": "integer"},
    ),
]


def _get_agent_or_404(agent_id: UUID) -> dict:
    record = _agents.get(agent_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found.",
        )
    return record


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/types",
    response_model=list[AgentTypeInfo],
    summary="List agent types",
    description="Return all registered agent types with their configuration schemas.",
)
async def list_agent_types() -> list[AgentTypeInfo]:
    """Return the registry of available agent types."""
    return _AGENT_TYPES


@router.get(
    "/",
    response_model=AgentListResponse,
    summary="List agents",
    description="Return a paginated list of registered agents.",
)
async def list_agents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    active_only: bool = Query(default=False, description="Filter to active agents only."),
) -> PaginatedResponse[AgentSummary]:
    """List available agents with optional filtering."""
    agents = list(_agents.values())
    if active_only:
        agents = [a for a in agents if a["is_active"]]
    agents.sort(key=lambda a: a["created_at"], reverse=True)
    skip = (page - 1) * page_size
    page_items = agents[skip : skip + page_size]
    return PaginatedResponse(
        items=[AgentSummary(**a) for a in page_items],
        total=len(agents),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="Get agent",
    description="Retrieve full details of a registered agent.",
)
async def get_agent(agent_id: UUID) -> AgentResponse:
    """Retrieve a single agent by ID."""
    return AgentResponse(**_get_agent_or_404(agent_id))


@router.get(
    "/{agent_id}/executions",
    response_model=PaginatedResponse[AgentExecutionResponse],
    summary="List agent executions",
    description="Return a paginated history of executions for the given agent.",
)
async def list_agent_executions(
    agent_id: UUID,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
) -> PaginatedResponse[AgentExecutionResponse]:
    """Return execution history for an agent."""
    _get_agent_or_404(agent_id)
    executions = sorted(
        _executions.get(agent_id, []),
        key=lambda e: e["started_at"],
        reverse=True,
    )
    skip = (page - 1) * page_size
    page_items = executions[skip : skip + page_size]
    return PaginatedResponse(
        items=[AgentExecutionResponse(**e) for e in page_items],
        total=len(executions),
        page=page,
        page_size=page_size,
    )


@router.post(
    "/{agent_id}/test",
    response_model=AgentTestResponse,
    summary="Test agent",
    description="Execute a single test invocation against an agent with sample input.",
)
async def test_agent(agent_id: UUID, payload: AgentTestRequest) -> AgentTestResponse:
    """Run a test invocation of an agent."""
    record = _get_agent_or_404(agent_id)
    if not record["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent {agent_id} is not active.",
        )

    start = time.monotonic()
    # TODO: dispatch to actual agent execution
    duration_ms = int((time.monotonic() - start) * 1000)

    # Record the execution in the history store.
    now = datetime.now(timezone.utc)
    execution = {
        "id": uuid4(),
        "agent_id": agent_id,
        "workflow_id": None,
        "status": "completed",
        "input_data": payload.input_data,
        "output_data": {"message": "Test execution placeholder"},
        "error_message": None,
        "duration_ms": duration_ms,
        "tokens_used": 0,
        "started_at": now,
        "completed_at": now,
    }
    _executions.setdefault(agent_id, []).append(execution)

    await logger.ainfo("Agent test executed", agent_id=str(agent_id))
    return AgentTestResponse(
        agent_id=agent_id,
        status="completed",
        output=execution["output_data"],
        duration_ms=duration_ms,
        tokens_used=0,
    )
