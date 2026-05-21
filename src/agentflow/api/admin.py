"""Administrative and demo-data utilities.

These endpoints are intended for local development, demos, and screenshot/
recording sessions. They are gated behind ``settings.debug`` and must never be
enabled in a production deployment.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, status

from agentflow.config import get_settings
from agentflow.core.fault_tolerance import DeadLetterQueue

logger = structlog.get_logger()

router = APIRouter(tags=["admin"])

# ---------------------------------------------------------------------------
# Demo-data blueprints
# ---------------------------------------------------------------------------

# (name, status, description) — 7 completed, 3 running, 1 failed, 1 pending.
_WORKFLOW_BLUEPRINTS: list[tuple[str, str, str]] = [
    (
        "Content Research Pipeline",
        "completed",
        "Researches, analyses, and drafts long-form content.",
    ),
    (
        "Code Review Automation",
        "completed",
        "Automated multi-agent review of incoming pull requests.",
    ),
    ("Market Analysis Q4", "running", "End-of-quarter market sizing and competitor positioning."),
    (
        "Customer Sentiment Audit",
        "completed",
        "Aggregates and scores sentiment across support channels.",
    ),
    (
        "API Documentation Generator",
        "completed",
        "Generates reference docs from the OpenAPI specification.",
    ),
    (
        "Security Vulnerability Scan",
        "failed",
        "Scans dependencies and surfaces exploitable findings.",
    ),
    (
        "Data Migration Validator",
        "completed",
        "Validates row counts and integrity after a data migration.",
    ),
    ("Onboarding Materials", "running", "Builds onboarding guides for new enterprise customers."),
    ("Product Launch Brief", "completed", "Compiles a go-to-market brief for the upcoming launch."),
    ("Competitor Analysis", "completed", "Profiles competitor features, pricing, and positioning."),
    (
        "User Feedback Synthesis",
        "running",
        "Synthesises themes from recent user feedback submissions.",
    ),
    (
        "Quarterly Report Draft",
        "pending",
        "Drafts the quarterly business review from source metrics.",
    ),
]

# Ordered agent sequences keyed by step count — research first, delivery last.
_STEP_SEQUENCES: dict[int, list[str]] = {
    2: ["research", "writer"],
    3: ["research", "analysis", "writer"],
    4: ["research", "analysis", "code", "writer"],
}

_STEP_PROMPTS: dict[str, str] = {
    "research": "Gather and summarise source material on the workflow topic.",
    "analysis": "Identify key trends, risks, and signals from the research output.",
    "writer": "Draft a polished deliverable from the analysed findings.",
    "code": "Generate and validate supporting scripts or schema definitions.",
}

_STEP_ERRORS: list[str] = [
    "Agent returned malformed JSON output.",
    "Upstream LLM call failed: 502 Bad Gateway.",
    "Validation failed: required field 'summary' missing.",
]

# (agent_type, display name, description) — one agent per built-in type.
_AGENT_BLUEPRINTS: list[tuple[str, str, str]] = [
    ("research", "Research Agent", "Gathers and synthesises information from multiple sources."),
    ("analysis", "Analysis Agent", "Extracts insights, trends, and risks from structured input."),
    ("writer", "Writer Agent", "Produces polished long-form written deliverables."),
    ("code", "Code Agent", "Generates and reviews code in a sandboxed environment."),
]

_EXECUTION_PROMPTS: dict[str, list[str]] = {
    "research": [
        "Compile recent industry benchmarks for the target market.",
        "Summarise the top five competitor product launches this quarter.",
        "Collect customer interview notes relevant to onboarding friction.",
    ],
    "analysis": [
        "Score sentiment across the latest batch of support tickets.",
        "Identify churn-risk signals in the Q4 usage dataset.",
        "Compare projected vs. actual revenue for the launch cohort.",
    ],
    "writer": [
        "Draft the executive summary for the quarterly report.",
        "Write release notes for the v2.3 platform update.",
        "Produce an onboarding checklist for new enterprise customers.",
    ],
    "code": [
        "Generate a migration script for the new schema revision.",
        "Refactor the retry handler to use exponential backoff.",
        "Add unit tests for the workflow cancellation path.",
    ],
}

_EXECUTION_ERRORS: list[str] = [
    "OpenAI API rate limit exceeded.",
    "Context length exceeded model maximum.",
    "Tool call returned a non-zero exit code.",
]

# (error message, retry count) for dead-letter queue entries.
_DLQ_BLUEPRINTS: list[tuple[str, int]] = [
    ("OpenAI API rate limit exceeded", 5),
    ("Timeout: agent execution exceeded 300s", 3),
    ("Invalid JSON in agent response", 4),
]

_EXECUTION_COUNT = 50
_EXECUTION_FAILURES = 3
_DAY_MINUTES = 24 * 60


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _build_steps(rng: random.Random, wf_status: str, step_count: int) -> list[dict[str, Any]]:
    """Build a realistic set of workflow steps for the given workflow status."""
    agent_types = _STEP_SEQUENCES[step_count]

    running_idx = failed_idx = -1
    if wf_status == "completed":
        progressed = step_count
    elif wf_status == "running":
        running_idx = rng.randint(0, step_count - 1)
        progressed = running_idx
    elif wf_status == "failed":
        failed_idx = rng.randint(0, step_count - 1)
        progressed = failed_idx
    else:  # pending
        progressed = 0

    steps: list[dict[str, Any]] = []
    for idx, agent_type in enumerate(agent_types):
        if idx < progressed:
            step_status = "completed"
        elif idx == running_idx:
            step_status = "running"
        elif idx == failed_idx:
            step_status = "failed"
        else:
            step_status = "pending"

        output_data: dict[str, Any] | None = None
        error_message: str | None = None
        duration_ms: int | None = None
        if step_status == "completed":
            duration_ms = rng.randint(800, 12000)
            output_data = {
                "summary": f"{agent_type.capitalize()} step completed successfully.",
                "tokens_used": rng.randint(200, 2400),
            }
        elif step_status == "failed":
            duration_ms = rng.randint(800, 12000)
            error_message = rng.choice(_STEP_ERRORS)

        steps.append(
            {
                "id": uuid4(),
                "step_order": idx,
                "agent_type": agent_type,
                "input_data": {"prompt": _STEP_PROMPTS[agent_type]},
                "output_data": output_data,
                "status": step_status,
                "error_message": error_message,
                "duration_ms": duration_ms,
            }
        )
    return steps


def _seed_workflows(
    store: dict[UUID, dict],
    rng: random.Random,
    now: datetime,
) -> list[UUID]:
    """Populate the workflow store and return the created workflow IDs."""
    workflow_ids: list[UUID] = []
    for name, wf_status, description in _WORKFLOW_BLUEPRINTS:
        workflow_id = uuid4()
        created_at = now - timedelta(minutes=rng.randint(15, _DAY_MINUTES))
        steps = _build_steps(rng, wf_status, rng.randint(2, 4))

        started_at: datetime | None = None
        completed_at: datetime | None = None
        if wf_status in ("completed", "running", "failed"):
            started_at = created_at + timedelta(seconds=rng.randint(3, 40))
        if wf_status == "completed":
            completed_at = started_at + timedelta(milliseconds=rng.randint(2000, 45000))
        elif wf_status == "failed":
            completed_at = started_at + timedelta(milliseconds=rng.randint(2000, 20000))

        store[workflow_id] = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "status": wf_status,
            "config": {},
            "created_by": "demo-seed",
            "steps": steps,
            "created_at": created_at,
            "updated_at": completed_at or now,
            "started_at": started_at,
            "completed_at": completed_at,
        }
        workflow_ids.append(workflow_id)
    return workflow_ids


def _seed_agents(
    store: dict[UUID, dict],
    rng: random.Random,
    now: datetime,
) -> dict[str, UUID]:
    """Populate the agent store and return a mapping of agent type to ID."""
    agent_ids: dict[str, UUID] = {}
    for agent_type, name, description in _AGENT_BLUEPRINTS:
        agent_id = uuid4()
        store[agent_id] = {
            "id": agent_id,
            "name": name,
            "type": agent_type,
            "description": description,
            "config": {"model": "claude-opus-4-7", "temperature": 0.7},
            "is_active": True,
            "created_at": now - timedelta(days=rng.randint(5, 30)),
            "updated_at": now,
        }
        agent_ids[agent_type] = agent_id
    return agent_ids


def _seed_executions(
    store: dict[UUID, list[dict]],
    agent_ids: dict[str, UUID],
    workflow_ids: list[UUID],
    rng: random.Random,
    now: datetime,
) -> int:
    """Populate the agent execution history store and return the count created."""
    failure_slots = set(rng.sample(range(_EXECUTION_COUNT), _EXECUTION_FAILURES))
    agent_items = list(agent_ids.items())

    for i in range(_EXECUTION_COUNT):
        agent_type, agent_id = rng.choice(agent_items)
        duration_ms = rng.randint(400, 9000)
        started_at = now - timedelta(minutes=rng.randint(5, _DAY_MINUTES))
        completed_at = started_at + timedelta(milliseconds=duration_ms)
        input_tokens = rng.randint(100, 2000)
        output_tokens = rng.randint(50, 1500)

        if i in failure_slots:
            ex_status = "failed"
            output_data: dict[str, Any] | None = None
            error_message: str | None = rng.choice(_EXECUTION_ERRORS)
            tokens_used = input_tokens  # no output produced on failure
        else:
            ex_status = "completed"
            output_data = {
                "result": f"{agent_type.capitalize()} task completed.",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            error_message = None
            tokens_used = input_tokens + output_tokens

        store.setdefault(agent_id, []).append(
            {
                "id": uuid4(),
                "agent_id": agent_id,
                "workflow_id": rng.choice(workflow_ids),
                "status": ex_status,
                "input_data": {"prompt": rng.choice(_EXECUTION_PROMPTS[agent_type])},
                "output_data": output_data,
                "error_message": error_message,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )
    return _EXECUTION_COUNT


async def _seed_dlq(
    dlq_store: DeadLetterQueue,
    workflow_ids: list[UUID],
    rng: random.Random,
    now: datetime,
) -> int:
    """Populate the dead-letter queue and return the count created."""
    for error, retry_count in _DLQ_BLUEPRINTS:
        workflow_id = rng.choice(workflow_ids)
        entry = await dlq_store.add(
            task_id=f"task-{uuid4().hex[:12]}",
            error=error,
            payload={
                "workflow_id": str(workflow_id),
                "agent_type": rng.choice(["research", "analysis", "writer", "code"]),
                "attempt": retry_count,
            },
            workflow_id=workflow_id,
            retry_count=retry_count,
            max_retries_reached=True,
        )
        # add() stamps created_at to "now"; backdate it for a realistic spread.
        entry.created_at = now - timedelta(minutes=rng.randint(20, 12 * 60))
    return len(_DLQ_BLUEPRINTS)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/seed-demo",
    summary="Seed demo data",
    description=(
        "Populate the in-memory workflow, agent, execution, and dead-letter "
        "queue stores with rich demo data for screenshots and recordings. "
        "Existing in-memory data is cleared first so repeated calls are "
        "idempotent. Available only when the application runs in debug mode."
    ),
)
async def seed_demo_data() -> dict[str, int]:
    """Reset the in-memory stores and fill them with demo data."""
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints are only available when debug mode is enabled.",
        )

    # Imported lazily so the stores resolve to the live module-level objects.
    from agentflow.api.agents import _agents, _executions
    from agentflow.api.dlq import _dlq_store
    from agentflow.api.workflows import _workflows

    rng = random.Random(20260521)
    now = datetime.now(UTC)

    # Reset every store so the returned summary is exact and repeatable.
    _workflows.clear()
    _agents.clear()
    _executions.clear()
    await _dlq_store.purge()

    workflow_ids = _seed_workflows(_workflows, rng, now)
    agent_ids = _seed_agents(_agents, rng, now)
    executions_created = _seed_executions(_executions, agent_ids, workflow_ids, rng, now)
    dlq_entries_created = await _seed_dlq(_dlq_store, workflow_ids, rng, now)

    summary = {
        "workflows_created": len(workflow_ids),
        "executions_created": executions_created,
        "dlq_entries_created": dlq_entries_created,
    }
    await logger.ainfo("Demo data seeded", **summary)
    return summary
