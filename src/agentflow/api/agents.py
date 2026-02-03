"""Agent management endpoints."""

from __future__ import annotations

import time
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from agentflow.api.schemas import (
    AgentResponse,
    AgentSummary,
    AgentTestRequest,
    AgentTestResponse,
    PaginatedResponse,
)

logger = structlog.get_logger()

router = APIRouter(tags=["agents"])

# ---------------------------------------------------------------------------
# In-memory store (placeholder until service layer is wired)
# ---------------------------------------------------------------------------
_agents: dict[UUID, dict] = {}


@router.get(
    "/",
    response_model=PaginatedResponse,
    summary="List agents",
    description="Return a paginated list of registered agents.",
)
async def list_agents(
    skip: int = Query(default=0, ge=0, description="Number of records to skip."),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return."),
    active_only: bool = Query(default=False, description="Filter to active agents only."),
) -> PaginatedResponse:
    """List available agents with optional filtering."""
    agents = list(_agents.values())
    if active_only:
        agents = [a for a in agents if a["is_active"]]
    agents.sort(key=lambda a: a["created_at"], reverse=True)
    page = agents[skip : skip + limit]
    return PaginatedResponse(
        items=[AgentSummary(**a) for a in page],
        total=len(agents),
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent",
    description="Retrieve full details of a registered agent.",
)
async def get_agent(agent_id: UUID) -> AgentResponse:
    """Retrieve a single agent by ID."""
    record = _agents.get(agent_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found.",
        )
    return AgentResponse(**record)


@router.post(
    "/{agent_id}/test",
    response_model=AgentTestResponse,
    summary="Test agent",
    description="Execute a single test invocation against an agent with sample input.",
)
async def test_agent(agent_id: UUID, payload: AgentTestRequest) -> AgentTestResponse:
    """Run a test invocation of an agent."""
    record = _agents.get(agent_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found.",
        )
    if not record["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent {agent_id} is not active.",
        )

    start = time.monotonic()
    # TODO: dispatch to actual agent execution
    duration_ms = int((time.monotonic() - start) * 1000)
    await logger.ainfo("Agent test executed", agent_id=str(agent_id))

    return AgentTestResponse(
        agent_id=agent_id,
        status="completed",
        output={"message": "Test execution placeholder"},
        duration_ms=duration_ms,
        tokens_used=0,
    )
