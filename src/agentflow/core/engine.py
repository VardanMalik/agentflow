"""Workflow execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class StepResult:
    """Result of executing a single workflow step."""

    step_id: UUID
    status: str
    output: Any = None
    error: str | None = None


@dataclass
class WorkflowDefinition:
    """Definition of a workflow and its steps."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate the workflow definition.

        Returns:
            A list of validation error messages. Empty if valid.
        """
        errors: list[str] = []
        if not self.name:
            errors.append("Workflow name is required")
        if not self.steps:
            errors.append("Workflow must have at least one step")
        return errors


class WorkflowEngine:
    """Executes workflow definitions by running steps in order.

    Handles step sequencing, parallel execution groups,
    and error propagation.
    """

    async def execute(self, definition: WorkflowDefinition) -> list[StepResult]:
        """Execute a workflow definition.

        Args:
            definition: The workflow to execute.

        Returns:
            A list of step results in execution order.
        """
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        results: list[StepResult] = []
        for step in definition.steps:
            result = await self._execute_step(step)
            results.append(result)
            if result.error:
                await logger.aerror(
                    "Step failed, aborting workflow",
                    step_id=str(result.step_id),
                    error=result.error,
                )
                break

        return results

    async def _execute_step(self, step: dict[str, Any]) -> StepResult:
        """Execute a single workflow step.

        Args:
            step: The step configuration.

        Returns:
            The result of the step execution.
        """
        step_id = uuid4()
        await logger.ainfo("Executing step", step_id=str(step_id), step=step)
        # TODO: dispatch to appropriate agent via Celery
        return StepResult(step_id=step_id, status="completed")
