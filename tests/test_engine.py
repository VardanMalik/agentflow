"""Tests for the workflow engine."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from agentflow.agents.base import AgentContext, AgentResult, BaseAgent
from agentflow.core.engine import (
    AgentRegistry,
    StepDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)
from agentflow.core.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
    WorkflowStateError,
)
from agentflow.core.state import Status


# ---------------------------------------------------------------------------
# Test agent
# ---------------------------------------------------------------------------

class EchoAgent(BaseAgent):
    """Agent that echoes its input for testing."""

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(output={"echo": context.inputs})


class FailingAgent(BaseAgent):
    """Agent that always returns an error."""

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(error="deliberately failed")


def _make_engine(*agent_types: str) -> WorkflowEngine:
    registry = AgentRegistry()
    for t in agent_types:
        if t == "failing":
            registry.register(t, FailingAgent())
        else:
            registry.register(t, EchoAgent())
    return WorkflowEngine(agent_registry=registry)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_workflow_validation_empty_name():
    """Workflow with no name fails validation."""
    d = WorkflowDefinition(steps=[StepDefinition(agent_type="llm")])
    errors = d.validate()
    assert "Workflow name is required" in errors


def test_workflow_validation_no_steps():
    """Workflow with no steps fails validation."""
    d = WorkflowDefinition(name="test")
    errors = d.validate()
    assert "Workflow must have at least one step" in errors


def test_workflow_validation_missing_agent_type():
    """Step without agent_type fails validation."""
    d = WorkflowDefinition(name="test", steps=[StepDefinition(agent_type="")])
    errors = d.validate()
    assert any("agent_type" in e for e in errors)


@pytest.mark.asyncio
async def test_engine_rejects_invalid_workflow():
    """Engine raises ValidationError for invalid definitions."""
    engine = _make_engine("llm")
    with pytest.raises(ValidationError):
        await engine.create_workflow(WorkflowDefinition())


# ---------------------------------------------------------------------------
# Creation & lookup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_workflow():
    """create_workflow returns a pending WorkflowState."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="test",
        steps=[StepDefinition(agent_type="llm")],
    )
    state = await engine.create_workflow(definition)
    assert state.name == "test"
    assert state.status == Status.PENDING
    assert state.total_steps == 1


@pytest.mark.asyncio
async def test_get_workflow_not_found():
    """get_workflow raises WorkflowNotFoundError for unknown IDs."""
    engine = _make_engine()
    with pytest.raises(WorkflowNotFoundError):
        engine.get_workflow(uuid4())


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_sequential_workflow():
    """Engine executes sequential steps and completes."""
    engine = _make_engine("llm", "tool")
    definition = WorkflowDefinition(
        name="seq-workflow",
        steps=[
            StepDefinition(agent_type="llm", input_data={"prompt": "hello"}),
            StepDefinition(agent_type="tool"),
        ],
    )
    state = await engine.create_workflow(definition)
    result = await engine.execute_workflow(state.id)

    assert result.status == Status.COMPLETED
    assert result.completed_steps == 2
    assert all(s.status == Status.COMPLETED for s in result.steps)


@pytest.mark.asyncio
async def test_execute_parallel_steps():
    """Steps with the same parallel_group run concurrently."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="parallel-workflow",
        steps=[
            StepDefinition(agent_type="llm", parallel_group="batch1"),
            StepDefinition(agent_type="llm", parallel_group="batch1"),
            StepDefinition(agent_type="llm"),  # runs after batch
        ],
    )
    state = await engine.create_workflow(definition)
    result = await engine.execute_workflow(state.id)

    assert result.status == Status.COMPLETED
    assert result.completed_steps == 3


@pytest.mark.asyncio
async def test_execute_fails_on_step_error():
    """Workflow transitions to FAILED when a step errors."""
    engine = _make_engine("llm", "failing")
    definition = WorkflowDefinition(
        name="fail-workflow",
        steps=[
            StepDefinition(agent_type="llm"),
            StepDefinition(agent_type="failing"),
        ],
    )
    state = await engine.create_workflow(definition)
    result = await engine.execute_workflow(state.id)

    assert result.status == Status.FAILED
    assert result.completed_steps == 1
    assert result.failed_steps == 1


@pytest.mark.asyncio
async def test_execute_missing_agent():
    """Step with unregistered agent type fails gracefully."""
    engine = _make_engine()  # no agents registered
    definition = WorkflowDefinition(
        name="no-agent",
        steps=[StepDefinition(agent_type="unknown")],
    )
    state = await engine.create_workflow(definition)
    result = await engine.execute_workflow(state.id)

    assert result.status == Status.FAILED
    assert result.steps[0].error is not None


# ---------------------------------------------------------------------------
# Cancel & retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_pending_workflow():
    """Pending workflows can be cancelled."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="cancel-test",
        steps=[StepDefinition(agent_type="llm")],
    )
    state = await engine.create_workflow(definition)
    cancelled = await engine.cancel_workflow(state.id)

    assert cancelled.status == Status.CANCELLED
    assert all(s.status == Status.CANCELLED for s in cancelled.steps)


@pytest.mark.asyncio
async def test_cancel_completed_raises():
    """Cannot cancel a completed workflow."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="done",
        steps=[StepDefinition(agent_type="llm")],
    )
    state = await engine.create_workflow(definition)
    await engine.execute_workflow(state.id)

    with pytest.raises(WorkflowStateError):
        await engine.cancel_workflow(state.id)


@pytest.mark.asyncio
async def test_retry_failed_workflow():
    """Failed workflows can be retried."""
    engine = _make_engine("failing")
    definition = WorkflowDefinition(
        name="retry-test",
        steps=[StepDefinition(agent_type="failing")],
    )
    state = await engine.create_workflow(definition)
    await engine.execute_workflow(state.id)
    assert state.status == Status.FAILED

    # Swap in a working agent and retry
    engine._agents.register("failing", EchoAgent())
    retried = await engine.retry_workflow(state.id)
    assert retried.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_retry_non_failed_raises():
    """Cannot retry a workflow that is not failed."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="ok",
        steps=[StepDefinition(agent_type="llm")],
    )
    state = await engine.create_workflow(definition)

    with pytest.raises(WorkflowStateError):
        await engine.retry_workflow(state.id)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_workflow_status():
    """get_workflow_status returns a serialisable dict."""
    engine = _make_engine("llm")
    definition = WorkflowDefinition(
        name="status-test",
        steps=[StepDefinition(agent_type="llm")],
    )
    state = await engine.create_workflow(definition)
    status = await engine.get_workflow_status(state.id)

    assert status["status"] == "pending"
    assert status["total_steps"] == 1
    assert status["progress_pct"] == 0.0
