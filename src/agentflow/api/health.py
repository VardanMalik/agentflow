"""Health and readiness check endpoints."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter

from agentflow.api.schemas import HealthResponse, ReadinessResponse, ServiceCheck
from agentflow.config import get_settings

logger = structlog.get_logger()

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns basic application health status.",
)
async def health_check() -> HealthResponse:
    """Return application health status."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Verifies connectivity to PostgreSQL and Redis.",
)
async def readiness_check() -> ReadinessResponse:
    """Check that all backing services are reachable."""
    checks: dict[str, ServiceCheck] = {
        "database": await _check_database(),
        "redis": await _check_redis(),
    }
    overall = "ready" if all(c.status == "ok" for c in checks.values()) else "degraded"
    return ReadinessResponse(status=overall, checks=checks)


async def _check_database() -> ServiceCheck:
    """Ping the PostgreSQL database."""
    from sqlalchemy import text

    from agentflow.models.base import get_engine

    try:
        engine = get_engine()
        start = time.monotonic()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.monotonic() - start) * 1000
        return ServiceCheck(status="ok", latency_ms=round(latency, 2))
    except Exception as exc:
        await logger.awarning("Database health check failed", error=str(exc))
        return ServiceCheck(status="error", error=str(exc))


async def _check_redis() -> ServiceCheck:
    """Ping the Redis instance."""
    import redis.asyncio as aioredis

    from agentflow.config import get_settings

    try:
        settings = get_settings()
        start = time.monotonic()
        client = aioredis.from_url(str(settings.redis_url))
        await client.ping()
        await client.aclose()
        latency = (time.monotonic() - start) * 1000
        return ServiceCheck(status="ok", latency_ms=round(latency, 2))
    except Exception as exc:
        await logger.awarning("Redis health check failed", error=str(exc))
        return ServiceCheck(status="error", error=str(exc))
