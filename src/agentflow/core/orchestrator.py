"""High-level orchestrator interface for workflow submission and monitoring."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import structlog

from agentflow.agents.base import BaseAgent
from agentflow.core.engine import (
    AgentRegistry,
    StepDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)
from agentflow.core.exceptions import (
    AgentNotFoundError,
    ValidationError,
    WorkflowNotFoundError,
    WorkflowTimeoutError,
)
from agentflow.core.state import Status, WorkflowState

logger = structlog.get_logger()


class Orchestrator:
    """High-level interface for submitting and monitoring workflows.

    Wraps WorkflowEngine with definition parsing, agent validation,
    and blocking wait-for-completion semantics.
    """

    def __init__(self, engine: WorkflowEngine | None = None) -> None:
        self._engine = engine or WorkflowEngine()

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._engine._agents

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register an agent implementation for a given type."""
        self._engine._agents.register(agent_type, agent)

    # ---- Submission ----

    async def submit(
        self,
        workflow_definition: dict[str, Any],
    ) -> UUID:
        """Parse a workflow definition dict and create it for execution.

        Args:
            workflow_definition: Raw definition with 'name', 'steps',
                and optional 'description' and 'config' keys.

        Returns:
            The created workflow's UUID.

        Raises:
            ValidationError: If the definition is malformed.
            AgentNotFoundError: If a step references an unregistered agent.
        """
        definition = self._parse_definition(workflow_definition)
        self._validate_agents(definition)

        state = await self._engine.create_workflow(definition)

        await logger.ainfo(
            "Workflow submitted",
            workflow_id=str(state.id),
            name=state.name,
        )
        return state.id

    async def submit_and_execute(
        self,
        workflow_definition: dict[str, Any],
    ) -> WorkflowState:
        """Submit a workflow and immediately start execution.

        Returns:
            The final WorkflowState after execution.
        """
        workflow_id = await self.submit(workflow_definition)
        return await self._engine.execute_workflow(workflow_id)

    # ---- Monitoring ----

    async def wait_for_completion(
        self,
        workflow_id: UUID,
        timeout: float | None = None,
    ) -> WorkflowState:
        """Block until a workflow reaches a terminal state.

        Args:
            workflow_id: The workflow to wait for.
            timeout: Max seconds to wait. None means wait forever.

        Returns:
            The final WorkflowState.

        Raises:
            WorkflowNotFoundError: If not found.
            WorkflowTimeoutError: If timeout exceeded.
        """
        state = self._engine.get_workflow(workflow_id)

        if state.status.is_terminal:
            return state

        q = state.subscribe()
        try:
            deadline = timeout
            while True:
                try:
                    _, new_status = await asyncio.wait_for(q.get(), timeout=deadline)
                    if new_status.is_terminal:
                        return self._engine.get_workflow(workflow_id)
                except asyncio.TimeoutError:
                    raise WorkflowTimeoutError(workflow_id, timeout or 0)
        finally:
            state.unsubscribe(q)

    async def get_results(self, workflow_id: UUID) -> dict[str, Any]:
        """Collect all step outputs for a workflow.

        Args:
            workflow_id: The workflow to get results for.

        Returns:
            Dict with workflow status, progress, and per-step outputs.

        Raises:
            WorkflowNotFoundError: If not found.
        """
        state = self._engine.get_workflow(workflow_id)
        return {
            "workflow_id": str(state.id),
            "name": state.name,
            "status": state.status.value,
            "progress_pct": state.progress_pct,
            "steps": [
                {
                    "step_id": str(step.id),
                    "step_order": step.step_order,
                    "agent_type": step.agent_type,
                    "status": step.status.value,
                    "output": step.output_data,
                    "error": step.error,
                    "duration_ms": step.duration_ms,
                }
                for step in state.steps
            ],
        }

    async def cancel(self, workflow_id: UUID) -> WorkflowState:
        """Cancel a running workflow."""
        return await self._engine.cancel_workflow(workflow_id)

    async def retry(self, workflow_id: UUID) -> WorkflowState:
        """Retry a failed workflow."""
        return await self._engine.retry_workflow(workflow_id)

    # ---- Internal helpers ----

    def _parse_definition(self, raw: dict[str, Any]) -> WorkflowDefinition:
        """Convert a raw dict into a WorkflowDefinition."""
        steps_raw = raw.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ValidationError(["'steps' must be a list"])

        steps = [
            StepDefinition(
                agent_type=s.get("agent_type", ""),
                input_data=s.get("input_data"),
                parallel_group=s.get("parallel_group"),
            )
            for s in steps_raw
        ]

        return WorkflowDefinition(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            steps=steps,
            config=raw.get("config", {}),
        )

    def _validate_agents(self, definition: WorkflowDefinition) -> None:
        """Ensure all referenced agent types are registered.

        Raises:
            AgentNotFoundError: If any step uses an unregistered agent type.
        """
        for step in definition.steps:
            if not self._engine._agents.has(step.agent_type):
                raise AgentNotFoundError(step.agent_type)
