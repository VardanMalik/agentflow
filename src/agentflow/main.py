"""AgentFlow application entrypoint."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentflow.config import get_settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events."""
    settings = get_settings()
    await logger.ainfo(
        "Starting AgentFlow",
        version=settings.app_version,
        environment=settings.environment,
    )
    yield
    await logger.ainfo("Shutting down AgentFlow")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Distributed AI Agent Orchestration Platform",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

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
    from agentflow.api.routes import router as api_router

    app.include_router(api_router, prefix="/api/v1")


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
