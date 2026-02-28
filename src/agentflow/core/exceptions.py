"""Custom exceptions for the workflow engine."""

from __future__ import annotations

from uuid import UUID


class AgentFlowError(Exception):
    """Base exception for all AgentFlow errors."""


class WorkflowError(AgentFlowError):
    """Error related to workflow lifecycle or execution."""

    def __init__(self, workflow_id: UUID | str, message: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"Workflow {workflow_id}: {message}")


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow cannot be found."""

    def __init__(self, workflow_id: UUID | str) -> None:
        super().__init__(workflow_id, "not found")


class WorkflowStateError(WorkflowError):
    """Raised on an invalid workflow state transition."""

    def __init__(self, workflow_id: UUID | str, current: str, target: str) -> None:
        self.current_status = current
        self.target_status = target
        super().__init__(
            workflow_id,
            f"cannot transition from '{current}' to '{target}'",
        )


class StepError(AgentFlowError):
    """Error during execution of a single workflow step."""

    def __init__(self, step_id: UUID | str, message: str) -> None:
        self.step_id = step_id
        super().__init__(f"Step {step_id}: {message}")


class AgentError(AgentFlowError):
    """Error raised by an agent during execution."""

    def __init__(self, agent_id: UUID | str, message: str) -> None:
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id}: {message}")


class AgentNotFoundError(AgentError):
    """Raised when the requested agent is not registered or inactive."""

    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type
        super().__init__(agent_type, f"no active agent of type '{agent_type}'")


class WorkflowTimeoutError(AgentFlowError):
    """Raised when a workflow or step exceeds its time limit."""

    def __init__(self, workflow_id: UUID | str, timeout_seconds: float) -> None:
        self.workflow_id = workflow_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Workflow {workflow_id}: timed out after {timeout_seconds}s",
        )


class ValidationError(AgentFlowError):
    """Raised when workflow or step definitions fail validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class CircuitOpenError(AgentFlowError):
    """Raised when a circuit breaker is in the OPEN state and rejects a call."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Circuit breaker '{name}' is OPEN — call rejected")


class BulkheadFullError(AgentFlowError):
    """Raised when a bulkhead has no capacity and rejects a call."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Bulkhead '{name}' is full — call rejected")


class RetryExhaustedError(AgentFlowError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: BaseException | None = None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        detail = f": {last_error}" if last_error is not None else ""
        super().__init__(f"Retry exhausted after {attempts} attempt(s){detail}")
