"""Celery tasks for workflow and step execution."""

from __future__ import annotations

import time
from typing import Any

import structlog
from celery import chain, group

from agentflow.core.celery_app import celery_app
from agentflow.core.tasks import BaseTask, handle_task_errors, timed_task

logger = structlog.get_logger()


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="agentflow.core.tasks.workflow_tasks.execute_workflow",
)
@handle_task_errors
@timed_task
def execute_workflow(self: BaseTask, workflow_id: str) -> dict[str, Any]:
    """Execute an entire workflow by chaining its steps.

    Fetches the workflow definition, builds a Celery chain of
    execute_step tasks, and dispatches them. Each step receives
    the output of the previous step.

    Args:
        workflow_id: UUID of the workflow to execute.

    Returns:
        Summary dict with workflow_id and final status.
    """
    logger.info("Starting workflow execution", workflow_id=workflow_id)

    # TODO: load steps from database once service layer is wired
    # Placeholder: simulate a workflow with 3 sequential steps
    steps: list[dict[str, Any]] = [
        {"step_id": f"{workflow_id}-step-0", "step_order": 0, "agent_type": "llm"},
        {"step_id": f"{workflow_id}-step-1", "step_order": 1, "agent_type": "tool"},
        {"step_id": f"{workflow_id}-step-2", "step_order": 2, "agent_type": "validator"},
    ]

    if not steps:
        logger.warning("Workflow has no steps", workflow_id=workflow_id)
        return {"workflow_id": workflow_id, "status": "completed", "steps_executed": 0}

    # Build a chain: each step feeds its result into the next
    step_chain = chain(
        execute_step.s(workflow_id=workflow_id, step=step) for step in steps
    )
    step_chain.apply_async(
        link=cleanup_workflow.si(workflow_id=workflow_id, status="completed"),
        link_error=cleanup_workflow.si(workflow_id=workflow_id, status="failed"),
    )

    return {"workflow_id": workflow_id, "status": "running", "steps_dispatched": len(steps)}


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="agentflow.core.tasks.workflow_tasks.execute_step",
)
@handle_task_errors
@timed_task
def execute_step(
    self: BaseTask,
    previous_result: dict[str, Any] | None = None,
    *,
    workflow_id: str,
    step: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single workflow step.

    Receives the result of the previous step (or None for the first
    step) and produces output for the next step in the chain.

    Args:
        previous_result: Output from the preceding step, if any.
        workflow_id: Parent workflow UUID.
        step: Step configuration dict.

    Returns:
        Step result dict with status and output data.
    """
    step_id = step.get("step_id", "unknown")
    agent_type = step.get("agent_type", "unknown")

    logger.info(
        "Executing step",
        workflow_id=workflow_id,
        step_id=step_id,
        agent_type=agent_type,
    )

    start = time.monotonic()

    # TODO: dispatch to the appropriate agent based on agent_type
    output: dict[str, Any] = {
        "step_id": step_id,
        "agent_type": agent_type,
        "input": previous_result,
        "result": f"Processed by {agent_type}",
    }

    duration_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "Step completed",
        workflow_id=workflow_id,
        step_id=step_id,
        duration_ms=duration_ms,
    )

    return {
        "workflow_id": workflow_id,
        "step_id": step_id,
        "status": "completed",
        "output": output,
        "duration_ms": duration_ms,
    }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="agentflow.core.tasks.workflow_tasks.cleanup_workflow",
)
@handle_task_errors
def cleanup_workflow(self: BaseTask, *, workflow_id: str, status: str) -> dict[str, Any]:
    """Finalise a workflow after all steps complete or on failure.

    Updates the workflow record with the terminal status and
    performs any necessary cleanup (releasing resources, sending
    notifications, etc.).

    Args:
        workflow_id: UUID of the workflow.
        status: Terminal status ("completed" or "failed").

    Returns:
        Summary dict with workflow_id and final status.
    """
    logger.info(
        "Cleaning up workflow",
        workflow_id=workflow_id,
        status=status,
    )

    # TODO: update workflow status in database
    # TODO: emit workflow completion event

    return {"workflow_id": workflow_id, "status": status}
