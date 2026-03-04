"""Tests for agentflow.observability: metrics, tracing, logging, and middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset MetricsCollector singleton before and after each test."""
    from agentflow.observability import metrics as metrics_mod

    metrics_mod.MetricsCollector._instance = None
    yield
    metrics_mod.MetricsCollector._instance = None


@pytest.fixture(autouse=True)
def reset_tracing_provider():
    """Reset TracingProvider singleton before and after each test."""
    from agentflow.observability import tracing as tracing_mod

    tracing_mod.TracingProvider._provider = None
    yield
    tracing_mod.TracingProvider._provider = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Lambdas ensure each metric constructor call returns a fresh, spec-free
# MagicMock regardless of what positional args prometheus_client passes.
_mock_counter = lambda *a, **kw: MagicMock()  # noqa: E731
_mock_gauge = lambda *a, **kw: MagicMock()  # noqa: E731
_mock_histogram = lambda *a, **kw: MagicMock()  # noqa: E731


def _make_collector():
    """Create a MetricsCollector with mocked prometheus_client primitives."""
    with (
        patch("agentflow.observability.metrics.Counter", _mock_counter),
        patch("agentflow.observability.metrics.Gauge", _mock_gauge),
        patch("agentflow.observability.metrics.Histogram", _mock_histogram),
    ):
        from agentflow.observability.metrics import MetricsCollector

        return MetricsCollector.get_instance()


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_singleton_returns_same_instance(self):
        with (
            patch("agentflow.observability.metrics.Counter", _mock_counter),
            patch("agentflow.observability.metrics.Gauge", _mock_gauge),
            patch("agentflow.observability.metrics.Histogram", _mock_histogram),
        ):
            from agentflow.observability.metrics import MetricsCollector

            instance1 = MetricsCollector.get_instance()
            instance2 = MetricsCollector.get_instance()

        assert instance1 is instance2

    def test_record_workflow_started_increments_active(self):
        collector = _make_collector()
        collector.record_workflow_started()
        collector.workflows_active.inc.assert_called_once()

    def test_record_workflow_completed_updates_counter_histogram_and_gauge(self):
        collector = _make_collector()
        collector.record_workflow_completed(name="my_workflow", duration=1.5)

        collector.workflows_total.labels.assert_called_with(status="completed")
        collector.workflows_total.labels().inc.assert_called_once()

        collector.workflow_duration_seconds.labels.assert_called_with(workflow_name="my_workflow")
        collector.workflow_duration_seconds.labels().observe.assert_called_with(1.5)

        collector.workflows_active.dec.assert_called_once()

    def test_record_workflow_failed_decrements_active_and_increments_counter(self):
        collector = _make_collector()
        collector.record_workflow_failed()

        collector.workflows_total.labels.assert_called_with(status="failed")
        collector.workflows_total.labels().inc.assert_called_once()
        collector.workflows_active.dec.assert_called_once()

    def test_record_agent_execution_tracks_duration_and_tokens(self):
        collector = _make_collector()
        collector.record_agent_execution(
            agent_type="research",
            status="success",
            duration=2.0,
            input_tokens=100,
            output_tokens=200,
        )

        collector.agent_executions_total.labels.assert_called_with(
            agent_type="research", status="success"
        )
        collector.agent_executions_total.labels().inc.assert_called_once()

        collector.agent_execution_duration_seconds.labels.assert_called_with(agent_type="research")
        collector.agent_execution_duration_seconds.labels().observe.assert_called_with(2.0)

        token_calls = collector.agent_tokens_total.labels.call_args_list
        assert call(agent_type="research", direction="input") in token_calls
        assert call(agent_type="research", direction="output") in token_calls

    def test_record_agent_execution_skips_zero_tokens(self):
        collector = _make_collector()
        collector.record_agent_execution(
            agent_type="writer",
            status="success",
            duration=0.5,
            input_tokens=0,
            output_tokens=0,
        )

        collector.agent_tokens_total.labels.assert_not_called()

    def test_record_retry_increments_counter(self):
        collector = _make_collector()
        collector.record_retry(operation="db_query")

        collector.retry_attempts_total.labels.assert_called_with(operation="db_query")
        collector.retry_attempts_total.labels().inc.assert_called_once()

    def test_record_circuit_breaker_state_maps_strings_to_numbers(self):
        collector = _make_collector()

        collector.record_circuit_breaker_state("svc_a", "closed")
        collector.circuit_breaker_state.labels.assert_called_with(name="svc_a")
        collector.circuit_breaker_state.labels().set.assert_called_with(0)

        collector.circuit_breaker_state.reset_mock()
        collector.record_circuit_breaker_state("svc_b", "open")
        collector.circuit_breaker_state.labels().set.assert_called_with(1)

        collector.circuit_breaker_state.reset_mock()
        collector.record_circuit_breaker_state("svc_c", "half_open")
        collector.circuit_breaker_state.labels().set.assert_called_with(2)

    def test_record_circuit_breaker_unknown_state_defaults_to_zero(self):
        collector = _make_collector()
        collector.record_circuit_breaker_state("svc", "unknown_state")
        collector.circuit_breaker_state.labels().set.assert_called_with(0)


# ---------------------------------------------------------------------------
# TracingProvider
# ---------------------------------------------------------------------------


class TestTracingProvider:
    def test_setup_configures_sdk_tracer_provider(self):
        with (
            patch("agentflow.observability.tracing.OTLPSpanExporter") as mock_exporter,
            patch("agentflow.observability.tracing.SDKTracerProvider") as mock_provider_cls,
            patch("agentflow.observability.tracing.BatchSpanProcessor") as mock_processor,
            patch("agentflow.observability.tracing.trace") as mock_trace,
        ):
            from agentflow.observability.tracing import TracingConfig, TracingProvider

            config = TracingConfig(
                service_name="test-svc",
                exporter_endpoint="grpc://localhost:4317",
                enabled=True,
                sample_rate=1.0,
            )
            TracingProvider.setup(config)

            mock_exporter.assert_called_once_with(endpoint="grpc://localhost:4317")
            mock_provider_cls.assert_called_once()
            mock_trace.set_tracer_provider.assert_called_once()

    def test_setup_skips_when_disabled(self):
        with (
            patch("agentflow.observability.tracing.SDKTracerProvider") as mock_provider_cls,
            patch("agentflow.observability.tracing.trace") as mock_trace,
        ):
            from agentflow.observability.tracing import TracingConfig, TracingProvider

            TracingProvider.setup(TracingConfig(enabled=False))

            mock_provider_cls.assert_not_called()
            mock_trace.set_tracer_provider.assert_not_called()

    def test_get_tracer_delegates_to_otel(self):
        with patch("agentflow.observability.tracing.trace") as mock_trace:
            from agentflow.observability.tracing import TracingProvider

            mock_trace.get_tracer.return_value = MagicMock()
            TracingProvider.get_tracer("my.scope")
            mock_trace.get_tracer.assert_called_once_with("my.scope")

    def test_shutdown_calls_provider_shutdown(self):
        mock_provider = MagicMock()

        with patch("agentflow.observability.tracing.trace"):
            from agentflow.observability.tracing import TracingProvider

            TracingProvider._provider = mock_provider
            TracingProvider.shutdown()

        mock_provider.shutdown.assert_called_once()
        assert TracingProvider._provider is None

    def test_shutdown_is_noop_when_not_configured(self):
        from agentflow.observability.tracing import TracingProvider

        TracingProvider._provider = None
        TracingProvider.shutdown()  # must not raise

    @pytest.mark.asyncio
    async def test_traced_decorator_creates_span_and_returns_result(self):
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with patch("agentflow.observability.tracing.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer

            from agentflow.observability.tracing import traced

            @traced(name="my_span")
            async def sample(x: int) -> str:
                return f"value={x}"

            result = await sample(42)

        assert result == "value=42"
        mock_tracer.start_as_current_span.assert_called_once_with("my_span")

    @pytest.mark.asyncio
    async def test_traced_decorator_records_exception_and_reraises(self):
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with patch("agentflow.observability.tracing.trace") as mock_trace:
            from opentelemetry.trace import StatusCode

            mock_trace.get_tracer.return_value = mock_tracer

            from agentflow.observability.tracing import traced

            @traced()
            async def failing():
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await failing()

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_traced_decorator_uses_function_qualname_when_no_name_given(self):
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with patch("agentflow.observability.tracing.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer

            from agentflow.observability.tracing import traced

            @traced()
            async def my_operation():
                return "ok"

            await my_operation()

        span_name = mock_tracer.start_as_current_span.call_args[0][0]
        assert "my_operation" in span_name


# ---------------------------------------------------------------------------
# logging_config
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    def test_configure_logging_calls_structlog_configure(self):
        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            mock_structlog.get_level_from_name.return_value = 20
            mock_structlog.make_filtering_bound_logger.return_value = MagicMock()

            from agentflow.observability.logging_config import configure_logging

            configure_logging(level="DEBUG", environment="development")

            mock_structlog.configure.assert_called_once()

    def test_configure_logging_uses_json_renderer_in_production(self):
        captured_kwargs: dict = {}

        def fake_configure(**kwargs):
            captured_kwargs.update(kwargs)

        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            mock_structlog.configure.side_effect = fake_configure
            mock_structlog.get_level_from_name.return_value = 20
            mock_structlog.make_filtering_bound_logger.return_value = MagicMock()
            mock_structlog.processors.JSONRenderer.return_value = "json_renderer"
            mock_structlog.dev.ConsoleRenderer.return_value = "console_renderer"
            mock_structlog.contextvars.merge_contextvars = MagicMock()
            mock_structlog.processors.add_log_level = MagicMock()
            mock_structlog.processors.StackInfoRenderer.return_value = MagicMock()
            mock_structlog.dev.set_exc_info = MagicMock()
            mock_structlog.processors.TimeStamper.return_value = MagicMock()
            mock_structlog.processors.CallsiteParameterAdder.return_value = MagicMock()
            mock_structlog.processors.CallsiteParameter = MagicMock()
            mock_structlog.PrintLoggerFactory.return_value = MagicMock()

            from agentflow.observability.logging_config import configure_logging

            configure_logging(level="INFO", environment="production")

        processors = captured_kwargs.get("processors", [])
        assert "json_renderer" in processors

    def test_log_context_binds_fields_on_enter(self):
        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            from agentflow.observability.logging_config import LogContext

            ctx = LogContext(workflow_id="wf-123", step="step-1")
            ctx.__enter__()

            mock_structlog.contextvars.bind_contextvars.assert_called_once_with(
                workflow_id="wf-123", step="step-1"
            )

    def test_log_context_unbinds_fields_on_exit(self):
        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            from agentflow.observability.logging_config import LogContext

            ctx = LogContext(workflow_id="wf-123")
            ctx.__enter__()
            ctx.__exit__(None, None, None)

            mock_structlog.contextvars.unbind_contextvars.assert_called_once_with("workflow_id")

    def test_log_context_as_context_manager_binds_and_unbinds(self):
        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            from agentflow.observability.logging_config import LogContext

            with LogContext(request_id="req-abc"):
                mock_structlog.contextvars.bind_contextvars.assert_called_once_with(
                    request_id="req-abc"
                )

            mock_structlog.contextvars.unbind_contextvars.assert_called_once_with("request_id")

    def test_get_logger_returns_structlog_logger(self):
        with patch("agentflow.observability.logging_config.structlog") as mock_structlog:
            mock_structlog.get_logger.return_value = MagicMock()

            from agentflow.observability.logging_config import get_logger

            logger = get_logger("my.module")
            mock_structlog.get_logger.assert_called_once_with("my.module")
            assert logger is mock_structlog.get_logger.return_value


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TestRequestTracingMiddleware:
    @pytest.mark.asyncio
    async def test_adds_trace_and_request_id_headers(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def fake_call_next(req):
            return mock_response

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        ctx = MagicMock()
        ctx.is_valid = True
        ctx.trace_id = 0xDEADBEEF
        ctx.span_id = 0xCAFEBABE
        mock_span.get_span_context.return_value = ctx

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/workflows"
        mock_request.url.__str__ = lambda s: "http://localhost/api/v1/workflows"

        with patch("agentflow.api.middleware.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer

            from agentflow.api.middleware import RequestTracingMiddleware

            middleware = RequestTracingMiddleware(MagicMock())
            response = await middleware.dispatch(mock_request, fake_call_next)

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_sets_error_status_on_exception(self):
        async def raising_call_next(req):
            raise RuntimeError("downstream failure")

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_span_context.return_value = MagicMock(is_valid=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"
        mock_request.url.__str__ = lambda s: "http://localhost/api/v1/workflows"

        with patch("agentflow.api.middleware.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer

            from agentflow.api.middleware import RequestTracingMiddleware

            middleware = RequestTracingMiddleware(MagicMock())
            with pytest.raises(RuntimeError, match="downstream failure"):
                await middleware.dispatch(mock_request, raising_call_next)

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()


class TestMetricsMiddleware:
    @pytest.mark.asyncio
    async def test_records_request_duration_and_total(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def fake_call_next(req):
            return mock_response

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/health"

        mock_collector = MagicMock()

        with patch("agentflow.api.middleware.MetricsCollector") as mock_cls:
            mock_cls.get_instance.return_value = mock_collector

            from agentflow.api.middleware import MetricsMiddleware

            middleware = MetricsMiddleware(MagicMock())
            response = await middleware.dispatch(mock_request, fake_call_next)

        assert response is mock_response

        expected_labels = {
            "method": "GET",
            "endpoint": "/api/v1/health",
            "status_code": "200",
        }
        mock_collector.request_duration_seconds.labels.assert_called_once_with(**expected_labels)
        mock_collector.request_duration_seconds.labels().observe.assert_called_once()

        mock_collector.requests_total.labels.assert_called_once_with(**expected_labels)
        mock_collector.requests_total.labels().inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_metrics_for_error_responses(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        async def fake_call_next(req):
            return mock_response

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"

        mock_collector = MagicMock()

        with patch("agentflow.api.middleware.MetricsCollector") as mock_cls:
            mock_cls.get_instance.return_value = mock_collector

            from agentflow.api.middleware import MetricsMiddleware

            middleware = MetricsMiddleware(MagicMock())
            await middleware.dispatch(mock_request, fake_call_next)

        mock_collector.requests_total.labels.assert_called_once_with(
            method="POST",
            endpoint="/api/v1/workflows",
            status_code="500",
        )
