"""Dashboard-specific endpoints providing aggregated system stats and health."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter

from agentflow.api.schemas import HealthResponse, ServiceCheck

logger = structlog.get_logger()

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workflow_store() -> dict:
    from agentflow.api.workflows import _workflows

    return _workflows


def _agent_store() -> dict:
    from agentflow.api.agents import _agents

    return _agents


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    summary="System statistics",
    description="Return aggregated statistics: total workflows, success rate, active agents, and average duration.",
)
async def get_stats() -> dict[str, Any]:
    """Compute and return overall system statistics."""
    workflows = list(_workflow_store().values())
    agents = list(_agent_store().values())

    total = len(workflows)
    completed = sum(1 for w in workflows if w["status"] == "completed")
    failed = sum(1 for w in workflows if w["status"] == "failed")
    running = sum(1 for w in workflows if w["status"] == "running")
    success_rate = round(completed / total * 100, 2) if total else 0.0

    durations = [
        (w["completed_at"] - w["started_at"]).total_seconds() * 1000
        for w in workflows
        if w.get("started_at") and w.get("completed_at")
    ]
    avg_duration_ms = round(sum(durations) / len(durations), 2) if durations else 0.0

    active_agents = sum(1 for a in agents if a.get("is_active"))

    return {
        "workflows": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate_pct": success_rate,
        },
        "agents": {
            "total": len(agents),
            "active": active_agents,
        },
        "performance": {
            "avg_workflow_duration_ms": avg_duration_ms,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/recent",
    summary="Recent workflow activity",
    description="Return the most recent workflow runs ordered by creation time.",
)
async def get_recent(
    limit: int = 10,
) -> dict[str, Any]:
    """Return the most recently created workflows."""
    workflows = sorted(
        _workflow_store().values(),
        key=lambda w: w["created_at"],
        reverse=True,
    )[:limit]

    return {
        "workflows": [
            {
                "id": str(w["id"]),
                "name": w["name"],
                "status": w["status"],
                "created_at": w["created_at"].isoformat(),
                "started_at": w["started_at"].isoformat() if w.get("started_at") else None,
                "completed_at": w["completed_at"].isoformat() if w.get("completed_at") else None,
                "step_count": len(w.get("steps", [])),
            }
            for w in workflows
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health",
    description="Combine application, database, and Redis health into a single response.",
)
async def get_health() -> HealthResponse:
    """Aggregate health across all system components."""
    from agentflow.api.health import _check_database, _check_redis
    from agentflow.config import get_settings

    settings = get_settings()

    db_check: ServiceCheck = await _check_database()
    redis_check: ServiceCheck = await _check_redis()

    all_ok = db_check.status == "ok" and redis_check.status == "ok"
    overall = "healthy" if all_ok else "degraded"

    await logger.ainfo(
        "Dashboard health checked",
        database=db_check.status,
        redis=redis_check.status,
    )

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
    )
