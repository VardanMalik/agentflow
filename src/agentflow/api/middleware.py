"""Observability middleware for FastAPI: tracing, metrics, and logging."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from agentflow.observability.logging_config import _request_id_var
from agentflow.observability.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Create an OpenTelemetry span per request and attach trace/request IDs."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = str(uuid.uuid4())
        _request_id_var.set(request_id)

        tracer = trace.get_tracer(__name__)
        span_name = f"{request.method} {request.url.path}"

        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.request_id", request_id)

            try:
                response: Response = await call_next(request)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 500:
                span.set_status(StatusCode.ERROR)

            ctx = span.get_span_context()
            if ctx is not None and ctx.is_valid:
                response.headers["X-Trace-ID"] = format(ctx.trace_id, "032x")
            response.headers["X-Request-ID"] = request_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP request duration and totals in Prometheus."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start

        metrics = MetricsCollector.get_instance()
        labels = {
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": str(response.status_code),
        }
        metrics.request_duration_seconds.labels(**labels).observe(duration)
        metrics.requests_total.labels(**labels).inc()

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request start and completion with method, path, status, and duration."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        start = time.perf_counter()

        await logger.ainfo(
            "Request started",
            method=request.method,
            path=request.url.path,
        )

        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        await logger.ainfo(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response
