"""Dead-letter queue inspection and management endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, Response, status

from agentflow.core.fault_tolerance import DeadLetterEntry, DeadLetterQueue

logger = structlog.get_logger()

router = APIRouter(tags=["dlq"])

# ---------------------------------------------------------------------------
# In-memory store (placeholder until a persistent DLQ backend is wired)
# ---------------------------------------------------------------------------
_dlq_store: DeadLetterQueue = DeadLetterQueue()


def _serialise(entry: DeadLetterEntry) -> dict[str, Any]:
    """Convert a DLQ entry into the JSON shape expected by the dashboard."""
    return {
        "id": str(entry.id),
        "task_id": entry.task_id,
        "workflow_id": str(entry.workflow_id) if entry.workflow_id else None,
        "error": entry.error,
        "payload": entry.payload,
        "retry_count": entry.retry_count,
        "max_retries_reached": entry.max_retries_reached,
        "created_at": entry.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/",
    summary="List DLQ entries",
    description="Return a paginated list of dead-letter queue entries in insertion order.",
)
async def list_dlq_entries(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
) -> dict[str, Any]:
    """Return a page of dead-letter queue entries."""
    total = await _dlq_store.size()
    offset = (page - 1) * page_size
    entries = await _dlq_store.list_entries(limit=page_size, offset=offset)
    return {
        "entries": [_serialise(e) for e in entries],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post(
    "/{entry_id}/retry",
    summary="Retry DLQ entry",
    description="Mark a dead-letter queue entry for retry and remove it from the queue.",
)
async def retry_dlq_entry(entry_id: UUID) -> dict[str, str]:
    """Mark a single DLQ entry for retry."""
    found = await _dlq_store.retry_entry(entry_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DLQ entry {entry_id} not found.",
        )
    await logger.ainfo("DLQ entry retried", entry_id=str(entry_id))
    return {"status": "retried", "id": str(entry_id)}


@router.delete(
    "/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Purge DLQ entry",
    description="Permanently remove a single dead-letter queue entry.",
)
async def purge_dlq_entry(entry_id: UUID) -> Response:
    """Permanently remove a single DLQ entry."""
    found = await _dlq_store.remove(entry_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DLQ entry {entry_id} not found.",
        )
    await logger.ainfo("DLQ entry purged", entry_id=str(entry_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/",
    summary="Purge all DLQ entries",
    description="Permanently remove every entry from the dead-letter queue.",
)
async def purge_all_dlq() -> dict[str, int]:
    """Permanently remove all DLQ entries."""
    count = await _dlq_store.purge()
    await logger.ainfo("DLQ purged", count=count)
    return {"purged": count}
