"""OpenTelemetry distributed tracing."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased, TraceIdRatioBased
from opentelemetry.trace import StatusCode

from agentflow.config import get_settings


@dataclass
class TracingConfig:
    """Configuration for the OpenTelemetry tracing provider."""

    service_name: str = field(default_factory=lambda: get_settings().otel_service_name)
    exporter_endpoint: str = field(default_factory=lambda: get_settings().otel_exporter_endpoint)
    enabled: bool = True
    sample_rate: float = 1.0


class TracingProvider:
    """Manages the global OpenTelemetry TracerProvider lifecycle."""

    _provider: SDKTracerProvider | None = None

    @classmethod
    def setup(cls, config: TracingConfig | None = None) -> None:
        """Configure OpenTelemetry with OTLP gRPC exporter and BatchSpanProcessor."""
        cfg = config or TracingConfig()
        if not cfg.enabled:
            return

        resource = Resource.create({SERVICE_NAME: cfg.service_name})

        sampler = ALWAYS_ON if cfg.sample_rate >= 1.0 else ParentBased(
            root=TraceIdRatioBased(cfg.sample_rate)
        )

        exporter = OTLPSpanExporter(endpoint=cfg.exporter_endpoint)
        provider = SDKTracerProvider(resource=resource, sampler=sampler)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        cls._provider = provider

    @classmethod
    def get_tracer(cls, name: str) -> trace.Tracer:
        """Return a tracer for the given instrumentation scope name."""
        return trace.get_tracer(name)

    @classmethod
    def shutdown(cls) -> None:
        """Flush pending spans and shut down the provider."""
        if cls._provider is not None:
            cls._provider.shutdown()
            cls._provider = None


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Decorator that wraps an async function in an OpenTelemetry span.

    Records function arguments as span attributes, captures exceptions,
    and attaches workflow_id / step_id when present in the signature.
    """
    def decorator(func: Any) -> Any:
        span_name = name or func.__qualname__
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, val in attributes.items():
                        span.set_attribute(key, str(val))

                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    for param_name, value in bound.arguments.items():
                        if param_name in ("self", "cls"):
                            continue
                        if param_name in ("workflow_id", "step_id"):
                            span.set_attribute(param_name, str(value))
                        else:
                            span.set_attribute(f"arg.{param_name}", str(value)[:256])
                except Exception:
                    pass

                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return wrapper

    return decorator


def inject_context() -> dict[str, str]:
    """Serialize the current trace context into a carrier dict.

    Use this before dispatching a Celery task to propagate the trace.
    """
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_context(carrier: dict[str, str]) -> otel_context.Context:
    """Restore a trace context from a carrier dict received in a Celery task."""
    return propagate.extract(carrier)
