"""AgentFlow application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agentflow.api.middleware import LoggingMiddleware, MetricsMiddleware, RequestTracingMiddleware
from agentflow.config import get_settings
from agentflow.observability.logging_config import configure_logging
from agentflow.observability.metrics import MetricsCollector
from agentflow.observability.tracing import TracingConfig, TracingProvider

logger = structlog.get_logger()

_OPENAPI_TAGS = [
    {
        "name": "workflows",
        "description": "Create, execute, monitor, and manage workflows.",
    },
    {
        "name": "agents",
        "description": "Register and interact with AI agents.",
    },
    {
        "name": "dashboard",
        "description": "Aggregated system statistics and health overview.",
    },
    {
        "name": "websocket",
        "description": "Real-time event streaming over WebSocket connections.",
    },
    {
        "name": "health",
        "description": "Application liveness and readiness probes.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events."""
    settings = get_settings()

    configure_logging(
        level=settings.log_level,
        environment=settings.environment,
        json_output=settings.environment == "production",
    )

    TracingProvider.setup(
        TracingConfig(
            service_name=settings.otel_service_name,
            exporter_endpoint=settings.otel_exporter_endpoint,
        )
    )

    MetricsCollector.get_instance()

    # Ensure WebSocket singletons are initialised before accepting connections.
    from agentflow.api.websocket import connection_manager, event_bus  # noqa: F401

    await logger.ainfo(
        "Starting AgentFlow",
        version=settings.app_version,
        environment=settings.environment,
    )

    yield

    await logger.ainfo("Shutting down AgentFlow")
    TracingProvider.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "**AgentFlow** is a distributed AI agent orchestration platform.\n\n"
            "Use the REST API to define and execute multi-step agent workflows, "
            "stream real-time events over WebSocket, and monitor system health."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_tags=_OPENAPI_TAGS,
        lifespan=lifespan,
    )

    # Middleware is applied in reverse add order: last added = outermost.
    # Desired request order: CORS → RequestTracing → Logging → Metrics → handler
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register API routers."""
    from agentflow.api.router import router as api_router
    from agentflow.api.websocket import router as ws_router

    app.include_router(api_router, prefix="/api/v1")
    # WebSocket endpoints live at root so WS clients use /ws and /ws/{workflow_id}.
    app.include_router(ws_router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app = create_app()


def cli() -> None:
    """Run the application via CLI."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agentflow.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    cli()
