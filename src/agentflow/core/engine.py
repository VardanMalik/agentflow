"""Workflow execution engine with state machine and step orchestration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import structlog

from agentflow.agents.base import AgentContext, AgentResult, BaseAgent
from agentflow.core.exceptions import (
    AgentNotFoundError,
    StepError,
    ValidationError,
    WorkflowNotFoundError,
    WorkflowStateError,
    WorkflowTimeoutError,
)
from agentflow.core.state import Status, StepState, WorkflowState, can_transition

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Workflow definition (input)
# ---------------------------------------------------------------------------

@dataclass
class StepDefinition:
    """Definition of a single step within a workflow."""

    agent_type: str
    input_data: dict[str, Any] | None = None
    parallel_group: str | None = None


@dataclass
class WorkflowDefinition:
    """Declarative definition of a workflow and its steps."""

    name: str = ""
    description: str = ""
    steps: list[StepDefinition] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation error messages. Empty if valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("Workflow name is required")
        if not self.steps:
            errors.append("Workflow must have at least one step")
        for i, step in enumerate(self.steps):
            if not step.agent_type:
                errors.append(f"Step {i} is missing agent_type")
        return errors


# ---------------------------------------------------------------------------
# Step result (output)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing a single workflow step."""

    step_id: UUID
    status: str
    output: Any = None
    error: str | None = None
    duration_ms: int | None = None


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Registry mapping agent type strings to BaseAgent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent_type: str, agent: BaseAgent) -> None:
        self._agents[agent_type] = agent

    def get(self, agent_type: str) -> BaseAgent | None:
        return self._agents.get(agent_type)

    def has(self, agent_type: str) -> bool:
        return agent_type in self._agents

    @property
    def available_types(self) -> list[str]:
        return list(self._agents.keys())


# ---------------------------------------------------------------------------
# Workflow Engine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """Core engine that creates, executes, and manages workflows.

    Implements a state machine (PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED)
    with support for sequential and parallel step execution.
    """

    def __init__(self, agent_registry: AgentRegistry | None = None) -> None:
        self._workflows: dict[UUID, WorkflowState] = {}
        self._agents = agent_registry or AgentRegistry()

    # ---- CRUD ----

    async def create_workflow(
        self,
        definition: WorkflowDefinition,
    ) -> WorkflowState:
        """Create a new workflow from a definition.

        Args:
            definition: The workflow definition to create from.

        Returns:
            The initial WorkflowState.

        Raises:
            ValidationError: If the definition is invalid.
        """
        errors = definition.validate()
        if errors:
            raise ValidationError(errors)

        steps = [
            StepState(
                id=uuid4(),
                step_order=idx,
                agent_type=step.agent_type,
                input_data=step.input_data,
                parallel_group=step.parallel_group,
            )
            for idx, step in enumerate(definition.steps)
        ]

        state = WorkflowState(
            id=uuid4(),
            name=definition.name,
            config=definition.config,
            steps=steps,
        )
        self._workflows[state.id] = state

        await logger.ainfo(
            "Workflow created",
            workflow_id=str(state.id),
            total_steps=state.total_steps,
        )
        return state

    def get_workflow(self, workflow_id: UUID) -> WorkflowState:
        """Retrieve a workflow state by ID.

        Raises:
            WorkflowNotFoundError: If the workflow does not exist.
        """
        state = self._workflows.get(workflow_id)
        if state is None:
            raise WorkflowNotFoundError(workflow_id)
        return state

    # ---- Execution ----

    async def execute_workflow(self, workflow_id: UUID) -> WorkflowState:
        """Start executing a workflow.

        Transitions the workflow to RUNNING and processes step batches.

        Raises:
            WorkflowNotFoundError: If not found.
            WorkflowStateError: If not in PENDING (or FAILED for retry).
        """
        state = self.get_workflow(workflow_id)
        self._transition(state, Status.RUNNING)
        state.mark_running()

        await logger.ainfo("Workflow execution started", workflow_id=str(workflow_id))

        try:
            batches = state.sequential_steps()
            previous_output: dict[str, Any] | None = None

            for batch in batches:
                if state.status == Status.CANCELLED:
                    break

                if len(batch) == 1:
                    result = await self._execute_step(state, batch[0], previous_output)
                    if result.error:
                        state.mark_failed()
                        await logger.aerror(
                            "Workflow failed",
                            workflow_id=str(workflow_id),
                            failed_step=str(result.step_id),
                        )
                        return state
                    previous_output = result.output
                else:
                    results = await self._execute_parallel(state, batch, previous_output)
                    failures = [r for r in results if r.error]
                    if failures:
                        state.mark_failed()
                        await logger.aerror(
                            "Workflow failed during parallel batch",
                            workflow_id=str(workflow_id),
                            failed_steps=[str(r.step_id) for r in failures],
                        )
                        return state
                    # Merge parallel outputs
                    previous_output = {
                        str(r.step_id): r.output for r in results
                    }

            if state.status == Status.RUNNING:
                state.mark_completed()
                await logger.ainfo("Workflow completed", workflow_id=str(workflow_id))

        except Exception as exc:
            state.mark_failed()
            await logger.aexception(
                "Workflow execution error",
                workflow_id=str(workflow_id),
            )
            raise StepError(workflow_id, str(exc)) from exc

        return state

    async def cancel_workflow(self, workflow_id: UUID) -> WorkflowState:
        """Cancel a running or pending workflow.

        Raises:
            WorkflowNotFoundError: If not found.
            WorkflowStateError: If in a terminal state.
        """
        state = self.get_workflow(workflow_id)
        self._transition(state, Status.CANCELLED)
        state.mark_cancelled()
        await logger.ainfo("Workflow cancelled", workflow_id=str(workflow_id))
        return state

    async def retry_workflow(self, workflow_id: UUID) -> WorkflowState:
        """Retry a failed workflow by re-running its failed steps.

        Resets failed steps to PENDING and re-executes the workflow.

        Raises:
            WorkflowNotFoundError: If not found.
            WorkflowStateError: If not in FAILED state.
        """
        state = self.get_workflow(workflow_id)

        if state.status != Status.FAILED:
            raise WorkflowStateError(workflow_id, state.status.value, Status.RUNNING.value)

        reset_count = state.reset_failed_steps()
        await logger.ainfo(
            "Retrying workflow",
            workflow_id=str(workflow_id),
            steps_reset=reset_count,
        )

        state.status = Status.PENDING
        return await self.execute_workflow(workflow_id)

    async def get_workflow_status(self, workflow_id: UUID) -> dict[str, Any]:
        """Return current workflow status with progress details.

        Raises:
            WorkflowNotFoundError: If not found.
        """
        state = self.get_workflow(workflow_id)
        return state.to_dict()

    # ---- Internal step execution ----

    async def _execute_step(
        self,
        workflow: WorkflowState,
        step: StepState,
        previous_output: dict[str, Any] | None,
    ) -> StepResult:
        """Execute a single step, delegating to the registered agent."""
        step.mark_running()

        await logger.ainfo(
            "Step started",
            workflow_id=str(workflow.id),
            step_id=str(step.id),
            agent_type=step.agent_type,
        )

        agent = self._agents.get(step.agent_type)
        if agent is None:
            error_msg = f"No agent registered for type '{step.agent_type}'"
            step.mark_failed(error_msg)
            return StepResult(step_id=step.id, status="failed", error=error_msg)

        context = AgentContext(
            run_id=workflow.id,
            step_id=step.id,
            inputs=step.input_data or {},
            metadata={"previous_output": previous_output, **workflow.config},
        )

        try:
            result: AgentResult = await agent.execute(context)

            if result.error:
                step.mark_failed(result.error)
                return StepResult(
                    step_id=step.id,
                    status="failed",
                    error=result.error,
                    duration_ms=step.duration_ms,
                )

            step.mark_completed(result.output if isinstance(result.output, dict) else {"result": result.output})
            return StepResult(
                step_id=step.id,
                status="completed",
                output=result.output,
                duration_ms=step.duration_ms,
            )

        except Exception as exc:
            step.mark_failed(str(exc))
            await logger.aexception(
                "Step execution error",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
            )
            return StepResult(
                step_id=step.id,
                status="failed",
                error=str(exc),
                duration_ms=step.duration_ms,
            )

    async def _execute_parallel(
        self,
        workflow: WorkflowState,
        steps: list[StepState],
        previous_output: dict[str, Any] | None,
    ) -> list[StepResult]:
        """Execute a batch of steps concurrently."""
        tasks = [
            self._execute_step(workflow, step, previous_output) for step in steps
        ]
        return list(await asyncio.gather(*tasks))

    # ---- State machine ----

    def _transition(self, state: WorkflowState, target: Status) -> None:
        """Validate and enforce a state transition.

        Raises:
            WorkflowStateError: If the transition is not allowed.
        """
        if not can_transition(state.status, target):
            raise WorkflowStateError(state.id, state.status.value, target.value)
