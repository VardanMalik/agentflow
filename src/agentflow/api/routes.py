"""API route definitions."""

from fastapi import APIRouter

from agentflow.api import health, workflows, agents

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
router.include_router(agents.router, prefix="/agents", tags=["agents"])
