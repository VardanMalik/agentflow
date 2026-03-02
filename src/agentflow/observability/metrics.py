"""Prometheus metrics collection for workflows, agents, and system requests."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


class MetricsCollector:
    """Singleton collector that owns all Prometheus metric objects."""

    _instance: MetricsCollector | None = None

    def __init__(self) -> None:
        # --- Workflow metrics ---
        self.workflows_total = Counter(
            "workflows_total",
            "Total workflows executed",
            ["status"],
        )
        self.workflow_duration_seconds = Histogram(
            "workflow_duration_seconds",
            "Workflow execution duration in seconds",
            ["workflow_name"],
        )
        self.workflows_active = Gauge(
            "workflows_active",
            "Currently active workflows",
        )

        # --- Agent / Step metrics ---
        self.agent_executions_total = Counter(
            "agent_executions_total",
            "Total agent executions",
            ["agent_type", "status"],
        )
        self.agent_execution_duration_seconds = Histogram(
            "agent_execution_duration_seconds",
            "Agent execution duration in seconds",
            ["agent_type"],
        )
        self.agent_tokens_total = Counter(
            "agent_tokens_total",
            "Total tokens consumed by agents",
            ["agent_type", "direction"],
        )

        # --- Fault tolerance metrics ---
        self.circuit_breaker_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["name"],
        )
        self.retry_attempts_total = Counter(
            "retry_attempts_total",
            "Total retry attempts",
            ["operation"],
        )
        self.dlq_entries_total = Gauge(
            "dlq_entries_total",
            "Total entries in the dead letter queue",
        )
        self.bulkhead_active_calls = Gauge(
            "bulkhead_active_calls",
            "Active concurrent calls inside a bulkhead",
            ["name"],
        )
        self.bulkhead_rejected_total = Counter(
            "bulkhead_rejected_total",
            "Total calls rejected by a bulkhead",
            ["name"],
        )

        # --- System / HTTP metrics ---
        self.request_duration_seconds = Histogram(
            "request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint", "status_code"],
        )
        self.requests_total = Counter(
            "requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Return the singleton MetricsCollector, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- Workflow helpers ---

    def record_workflow_started(self) -> None:
        self.workflows_active.inc()

    def record_workflow_completed(self, name: str, duration: float) -> None:
        self.workflows_total.labels(status="completed").inc()
        self.workflow_duration_seconds.labels(workflow_name=name).observe(duration)
        self.workflows_active.dec()

    def record_workflow_failed(self) -> None:
        self.workflows_total.labels(status="failed").inc()
        self.workflows_active.dec()

    # --- Agent helpers ---

    def record_agent_execution(
        self,
        agent_type: str,
        status: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.agent_executions_total.labels(agent_type=agent_type, status=status).inc()
        self.agent_execution_duration_seconds.labels(agent_type=agent_type).observe(duration)
        if input_tokens:
            self.agent_tokens_total.labels(agent_type=agent_type, direction="input").inc(
                input_tokens
            )
        if output_tokens:
            self.agent_tokens_total.labels(agent_type=agent_type, direction="output").inc(
                output_tokens
            )

    # --- Fault tolerance helpers ---

    def record_retry(self, operation: str) -> None:
        self.retry_attempts_total.labels(operation=operation).inc()

    def record_circuit_breaker_state(self, name: str, state: str) -> None:
        """Record circuit breaker state (closed=0, open=1, half_open=2)."""
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        self.circuit_breaker_state.labels(name=name).set(state_map.get(state, 0))
