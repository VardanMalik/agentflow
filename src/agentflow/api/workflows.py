"""Workflow management endpoints."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from agentflow.api.schemas import (
    PaginatedResponse,
    WorkflowCreate,
    WorkflowDetailResponse,
    WorkflowResponse,
    WorkflowStatusResponse,
    WorkflowStepResponse,
    WorkflowSummary,
)

logger = structlog.get_logger()

router = APIRouter(tags=["workflows"])

# ---------------------------------------------------------------------------
# In-memory store (placeholder until service layer is wired)
# ---------------------------------------------------------------------------
_workflows: dict[UUID, dict] = {}


def _get_or_404(workflow_id: UUID) -> dict:
    record = _workflows.get(workflow_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found.",
        )
    return record


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=WorkflowDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow",
    description="Define a new workflow with an ordered list of steps.",
)
async def create_workflow(payload: WorkflowCreate) -> WorkflowResponse:
    """Create a new workflow definition."""
    now = datetime.now(timezone.utc)
    workflow_id = uuid4()

    steps = [
        {
            "id": uuid4(),
            "step_order": idx,
            "agent_type": step.agent_type,
            "input_data": step.input_data,
            "output_data": None,
            "status": "pending",
            "error_message": None,
            "duration_ms": None,
        }
        for idx, step in enumerate(payload.steps)
    ]

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
    response_model=PaginatedResponse[WorkflowSummary],
    summary="List workflows",
    description="Return a paginated list of workflows ordered by creation time.",
)
async def list_workflows(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    status_filter: str | None = Query(
        default=None,
        alias="status",
        description="Filter by workflow status (pending, running, completed, failed, cancelled).",
    ),
) -> PaginatedResponse[WorkflowSummary]:
    """List all workflows with pagination and optional status filtering."""
    workflows = sorted(_workflows.values(), key=lambda w: w["created_at"], reverse=True)
    if status_filter:
        workflows = [w for w in workflows if w["status"] == status_filter]
    skip = (page - 1) * page_size
    page_items = workflows[skip : skip + page_size]
    return PaginatedResponse(
        items=[WorkflowSummary(**w) for w in page_items],
        total=len(workflows),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowDetailResponse,
    summary="Get workflow",
    description="Retrieve a workflow by its ID, including all steps.",
)
async def get_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Retrieve a single workflow by ID."""
    return WorkflowResponse(**_get_or_404(workflow_id))


@router.get(
    "/{workflow_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status",
    description="Return execution progress including step completion counts.",
)
async def get_workflow_status(workflow_id: UUID) -> WorkflowStatusResponse:
    """Return execution progress for a workflow."""
    record = _get_or_404(workflow_id)
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


@router.get(
    "/{workflow_id}/steps",
    response_model=list[WorkflowStepResponse],
    summary="Get workflow steps",
    description="Return all steps for a workflow, ordered by step_order.",
)
async def get_workflow_steps(workflow_id: UUID) -> list[WorkflowStepResponse]:
    """Return steps for a workflow."""
    record = _get_or_404(workflow_id)
    return [WorkflowStepResponse(**s) for s in record["steps"]]


@router.get(
    "/{workflow_id}/events",
    summary="Stream workflow events (SSE)",
    description=(
        "Subscribe to Server-Sent Events for a specific workflow. "
        "Streams workflow and step lifecycle events in real time."
    ),
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def workflow_events(workflow_id: UUID) -> StreamingResponse:
    """Stream workflow events as Server-Sent Events."""
    _get_or_404(workflow_id)

    from agentflow.api.websocket import event_bus

    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
    event_bus.add_sse_queue(queue)

    async def event_stream():
        try:
            # Send a connected confirmation immediately.
            yield f"event: connected\ndata: {json.dumps({'workflow_id': str(workflow_id)})}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=25)
                    # Filter to events relevant to this workflow.
                    data = item.get("data", {})
                    if data.get("workflow_id") and str(data["workflow_id"]) != str(workflow_id):
                        continue
                    payload = json.dumps({"type": item["type"], "data": data})
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive comment to prevent proxy timeouts.
                    yield ": keepalive\n\n"
        finally:
            event_bus.remove_sse_queue(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowDetailResponse,
    summary="Execute workflow",
    description="Start execution of a pending workflow.",
)
async def execute_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Kick off workflow execution."""
    record = _get_or_404(workflow_id)
    if record["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workflow is '{record['status']}', expected 'pending'.",
        )

    now = datetime.now(timezone.utc)
    record["status"] = "running"
    record["started_at"] = now
    record["updated_at"] = now

    from agentflow.api.websocket import event_bus

    await event_bus.publish("workflow.started", {"workflow_id": str(workflow_id)})
    await logger.ainfo("Workflow execution started", workflow_id=str(workflow_id))
    return WorkflowResponse(**record)


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowDetailResponse,
    summary="Cancel workflow",
    description="Cancel a running workflow and mark incomplete steps as cancelled.",
)
async def cancel_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Cancel a running workflow."""
    record = _get_or_404(workflow_id)
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

    from agentflow.api.websocket import event_bus

    await event_bus.publish("workflow.cancelled", {"workflow_id": str(workflow_id)})
    await logger.ainfo("Workflow cancelled", workflow_id=str(workflow_id))
    return WorkflowResponse(**record)


@router.post(
    "/{workflow_id}/retry",
    response_model=WorkflowDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Retry workflow",
    description="Create a new workflow run by copying a failed or cancelled workflow.",
)
async def retry_workflow(workflow_id: UUID) -> WorkflowResponse:
    """Retry a failed or cancelled workflow by cloning it as a new pending run."""
    record = _get_or_404(workflow_id)
    if record["status"] not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Can only retry failed or cancelled workflows, not '{record['status']}'.",
        )

    now = datetime.now(timezone.utc)
    new_id = uuid4()
    new_steps = [
        {
            **{k: v for k, v in step.items() if k not in ("id", "status", "output_data", "error_message", "duration_ms")},
            "id": uuid4(),
            "status": "pending",
            "output_data": None,
            "error_message": None,
            "duration_ms": None,
        }
        for step in record["steps"]
    ]
    new_record = {
        **{k: v for k, v in record.items() if k not in ("id", "status", "steps", "created_at", "updated_at", "started_at", "completed_at")},
        "id": new_id,
        "status": "pending",
        "steps": new_steps,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _workflows[new_id] = new_record
    await logger.ainfo("Workflow retried", original_id=str(workflow_id), new_id=str(new_id))
    return WorkflowResponse(**new_record)


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workflow",
    description="Permanently delete a workflow and its steps.",
)
async def delete_workflow(workflow_id: UUID) -> None:
    """Delete a workflow by ID."""
    _get_or_404(workflow_id)
    del _workflows[workflow_id]
    await logger.ainfo("Workflow deleted", workflow_id=str(workflow_id))
