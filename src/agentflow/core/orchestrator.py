"""Workflow orchestrator for coordinating agent execution."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class WorkflowStatus(StrEnum):
    """Possible states of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Orchestrator:
    """Coordinates workflow execution across distributed agents.

    Manages the lifecycle of workflows, dispatches tasks to agents,
    handles retries, and tracks execution state.
    """

    def __init__(self) -> None:
        self._active_workflows: dict[UUID, WorkflowStatus] = {}

    async def submit_workflow(self, workflow_id: UUID) -> UUID:
        """Submit a workflow for execution.

        Args:
            workflow_id: The ID of the workflow to execute.

        Returns:
            The execution run ID.
        """
        run_id = uuid4()
        self._active_workflows[run_id] = WorkflowStatus.PENDING
        await logger.ainfo(
            "Workflow submitted",
            workflow_id=str(workflow_id),
            run_id=str(run_id),
        )
        return run_id

    async def cancel_workflow(self, run_id: UUID) -> None:
        """Cancel a running workflow.

        Args:
            run_id: The execution run ID to cancel.
        """
        if run_id in self._active_workflows:
            self._active_workflows[run_id] = WorkflowStatus.CANCELLED
            await logger.ainfo("Workflow cancelled", run_id=str(run_id))

    async def get_status(self, run_id: UUID) -> WorkflowStatus | None:
        """Get the current status of a workflow execution.

        Args:
            run_id: The execution run ID.

        Returns:
            The current workflow status, or None if not found.
        """
        return self._active_workflows.get(run_id)
