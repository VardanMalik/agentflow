"""Tests for the workflow engine."""

import pytest

from agentflow.core.engine import WorkflowDefinition, WorkflowEngine


@pytest.mark.asyncio
async def test_workflow_validation_empty_name():
    """Workflow with no name fails validation."""
    definition = WorkflowDefinition(steps=[{"action": "test"}])
    errors = definition.validate()
    assert "Workflow name is required" in errors


@pytest.mark.asyncio
async def test_workflow_validation_no_steps():
    """Workflow with no steps fails validation."""
    definition = WorkflowDefinition(name="test")
    errors = definition.validate()
    assert "Workflow must have at least one step" in errors


@pytest.mark.asyncio
async def test_engine_rejects_invalid_workflow():
    """Engine raises ValueError for invalid workflow definitions."""
    engine = WorkflowEngine()
    definition = WorkflowDefinition()
    with pytest.raises(ValueError, match="Invalid workflow"):
        await engine.execute(definition)


@pytest.mark.asyncio
async def test_engine_executes_valid_workflow():
    """Engine executes all steps of a valid workflow."""
    engine = WorkflowEngine()
    definition = WorkflowDefinition(
        name="test-workflow",
        steps=[{"action": "step1"}, {"action": "step2"}],
    )
    results = await engine.execute(definition)
    assert len(results) == 2
    assert all(r.status == "completed" for r in results)
