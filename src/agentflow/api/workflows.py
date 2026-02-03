"""Workflow management endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class WorkflowCreate(BaseModel):
    """Request schema for creating a workflow."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="")
    steps: list[dict] = Field(default_factory=list)


class WorkflowResponse(BaseModel):
    """Response schema for a workflow."""

    id: str
    name: str
    description: str
    status: str
    steps: list[dict]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_workflow(payload: WorkflowCreate) -> WorkflowResponse:
    """Create a new workflow."""
    # TODO: implement with service layer
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Workflow creation not yet implemented",
    )


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Retrieve a workflow by ID."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Workflow retrieval not yet implemented",
    )


@router.get("/")
async def list_workflows(skip: int = 0, limit: int = 20) -> list[WorkflowResponse]:
    """List all workflows with pagination."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Workflow listing not yet implemented",
    )
