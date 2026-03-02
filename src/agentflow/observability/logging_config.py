"""Enhanced structured logging configuration with OpenTelemetry correlation."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

import structlog
from opentelemetry import trace

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _add_otel_context(
    logger: Any,  # noqa: ARG001
    method: str,  # noqa: ARG001
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject OpenTelemetry trace_id and span_id into every log record."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is not None and ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _add_request_id(
    logger: Any,  # noqa: ARG001
    method: str,  # noqa: ARG001
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject the current request_id into every log record."""
    request_id = _request_id_var.get()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(
    level: str = "INFO",
    environment: str = "development",
    json_output: bool = False,
) -> None:
    """Configure structlog with environment-appropriate settings.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        environment: Runtime environment; "production" forces JSON output.
        json_output: Explicitly request JSON output regardless of environment.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        _add_otel_context,
        _add_request_id,
    ]

    use_json = json_output or environment == "production"
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(message)s",
    )


class LogContext:
    """Context manager that temporarily binds extra fields to all log records."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __enter__(self) -> LogContext:
        structlog.contextvars.bind_contextvars(**self._fields)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self._fields.keys())


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to *name*."""
    return structlog.get_logger(name)
