"""Main API router that aggregates all sub-routers."""

from fastapi import APIRouter

from agentflow.api import agents, health, workflows

router = APIRouter()

router.include_router(health.router)
router.include_router(workflows.router, prefix="/workflows")
router.include_router(agents.router, prefix="/agents")
