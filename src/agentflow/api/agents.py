"""Agent management endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class AgentCreate(BaseModel):
    """Request schema for registering an agent."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="")
    model: str = Field(default="gpt-4o")
    system_prompt: str = Field(default="")
    capabilities: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Response schema for an agent."""

    id: str
    name: str
    description: str
    model: str
    status: str
    capabilities: list[str]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def register_agent(payload: AgentCreate) -> AgentResponse:
    """Register a new agent."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Agent registration not yet implemented",
    )


@router.get("/{agent_id}")
async def get_agent(agent_id: UUID) -> AgentResponse:
    """Retrieve an agent by ID."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Agent retrieval not yet implemented",
    )


@router.get("/")
async def list_agents(skip: int = 0, limit: int = 20) -> list[AgentResponse]:
    """List all registered agents."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Agent listing not yet implemented",
    )
