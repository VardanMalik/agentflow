"""Workflow management endpoints."""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from agentflow.api.schemas import (
    PaginatedResponse,
    WorkflowCreate,
    WorkflowResponse,
    WorkflowStatusResponse,
    WorkflowSummary,
)

logger = structlog.get_logger()

router = APIRouter(tags=["workflows"])

# ---------------------------------------------------------------------------
# In-memory store (placeholder until service layer is wired)
# ---------------------------------------------------------------------------
_workflows: dict[UUID, dict] = {}


@router.post(
    "/",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow",
    description="Define a new workflow with an ordered list of steps.",
)
async def create_workflow(payload: WorkflowCreate) -> WorkflowResponse:
    """Create a new workflow definition."""
    from datetime import datetime, timezone
    from uuid import uuid4

    now = datetime.now(timezone.utc)
    workflow_id = uuid4()

    steps = []
    for idx, step in enumerate(payload.steps):
        steps.append({
            "id": uuid4(),
            "step_order": idx,
            "agent_type": step.agent_type,
            "input_data": step.input_data,
            "output_data": None,
            "status": "pending",
            "error_message": None,
            "duration_ms": None,
        })

    record = {
        "id": workflow_id,
        "name": payload.name,
        "description": payload.description,
        "status": "pending",
        "config": payload.config,
        "created_by": None,
        "steps": steps,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _workflows[workflow_id] = record
    await logger.ainfo("Workflow created", workflow_id=str(workflow_id))
    return WorkflowResponse(**record)


@router.get(
    "/",
    response_model=PaginatedResponse,
    summary="List workflows",
    description="Return a paginated list of workflows ordered by creation time.",
)
async def list_workflows(
    skip: int = Query(default=0, ge=0, description="Number of records to skip."),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return."),
) -> PaginatedResponse:
    """List all workflows with pagination."""
    all_workflows = sorted(_workflows.values(), key=lambda w: w["created_at"], reverse=True)
    page = all_workflows[skip : skip + limit]
    return PaginatedResponse(
        items=[WorkflowSummary(**w) for w in page],
        total=len(all_workflows),
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get workflow",
    description="Retrieve a workflow by its ID, including all steps.",
)
async def get_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Retrieve a single workflow by ID."""
    record = _workflows.get(workflow_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    return WorkflowResponse(**record)


@router.get(
    "/{workflow_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status",
    description="Return execution progress including step completion counts.",
)
async def get_workflow_status(workflow_id: UUID) -> WorkflowStatusResponse:
    """Return execution progress for a workflow."""
    record = _workflows.get(workflow_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    steps = record["steps"]
    return WorkflowStatusResponse(
        id=record["id"],
        status=record["status"],
        total_steps=len(steps),
        completed_steps=sum(1 for s in steps if s["status"] == "completed"),
        failed_steps=sum(1 for s in steps if s["status"] == "failed"),
        started_at=record["started_at"],
        completed_at=record["completed_at"],
    )


@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowResponse,
    summary="Execute workflow",
    description="Start execution of a pending workflow.",
)
async def execute_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Kick off workflow execution."""
    from datetime import datetime, timezone

    record = _workflows.get(workflow_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    if record["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workflow is '{record['status']}', expected 'pending'.",
        )

    now = datetime.now(timezone.utc)
    record["status"] = "running"
    record["started_at"] = now
    record["updated_at"] = now
    await logger.ainfo("Workflow execution started", workflow_id=str(workflow_id))
    # TODO: dispatch to Celery task queue
    return WorkflowResponse(**record)


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowResponse,
    summary="Cancel workflow",
    description="Cancel a running workflow and mark incomplete steps as cancelled.",
)
async def cancel_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Cancel a running workflow."""
    from datetime import datetime, timezone

    record = _workflows.get(workflow_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    if record["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel workflow in '{record['status']}' state.",
        )

    now = datetime.now(timezone.utc)
    record["status"] = "cancelled"
    record["completed_at"] = now
    record["updated_at"] = now
    for step in record["steps"]:
        if step["status"] in ("pending", "running"):
            step["status"] = "cancelled"
    await logger.ainfo("Workflow cancelled", workflow_id=str(workflow_id))
    return WorkflowResponse(**record)


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workflow",
    description="Permanently delete a workflow and its steps.",
)
async def delete_workflow(workflow_id: UUID) -> None:
    """Delete a workflow by ID."""
    if workflow_id not in _workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    del _workflows[workflow_id]
    await logger.ainfo("Workflow deleted", workflow_id=str(workflow_id))
