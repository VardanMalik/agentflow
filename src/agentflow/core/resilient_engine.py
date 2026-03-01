"""Fault-tolerant wrapper around WorkflowEngine."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog

from agentflow.agents.base import AgentContext, AgentResult
from agentflow.core.engine import StepResult, WorkflowEngine
from agentflow.core.exceptions import BulkheadFullError, CircuitOpenError, RetryExhaustedError
from agentflow.core.fault_tolerance import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    DeadLetterEntry,
    DeadLetterQueue,
    RetryExecutor,
    RetryPolicy,
)
from agentflow.core.state import StepState, WorkflowState

logger = structlog.get_logger(__name__)


class ResilientWorkflowEngine(WorkflowEngine):
    """Fault-tolerant wrapper around :class:`~agentflow.core.engine.WorkflowEngine`.

    Adds four resilience layers to every step execution:

    1. **Bulkhead** — limits the number of concurrently executing steps.
    2. **Circuit Breaker** — short-circuits calls when the downstream agent is
       repeatedly failing, preventing cascading failures.
    3. **Retry** — retries individual agent calls with exponential back-off and
       optional jitter.
    4. **Dead-Letter Queue** — captures steps that exhaust all retry attempts
       for later inspection or manual replay.

    The layers are applied in this order for each step::

        Bulkhead -> Circuit Breaker -> Retry -> Agent.execute()

    All workflow management operations (create, get, execute, cancel, retry)
    are inherited from :class:`WorkflowEngine` and operate on the shared
    workflow state store.

    Usage::

        base_engine = WorkflowEngine(agent_registry=registry)
        resilient = ResilientWorkflowEngine(
            engine=base_engine,
            retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
            bulkhead_config=BulkheadConfig(max_concurrent=10),
            dead_letter_queue=DeadLetterQueue(),
        )
        state = await resilient.create_workflow(definition)
        await resilient.execute_workflow(state.id)
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        retry_policy: RetryPolicy,
        circuit_breaker_config: CircuitBreakerConfig,
        bulkhead_config: BulkheadConfig,
        dead_letter_queue: DeadLetterQueue,
    ) -> None:
        """Initialise the resilient engine by wrapping an existing engine.

        The resilient engine shares the wrapped engine's agent registry and
        workflow state store so that workflows created on either instance are
        mutually visible.

        Args:
            engine: The underlying :class:`WorkflowEngine` to wrap.
            retry_policy: Governs retry attempts and exponential back-off.
            circuit_breaker_config: Configuration for the internal
                :class:`~agentflow.core.fault_tolerance.CircuitBreaker`.
            bulkhead_config: Configuration for the internal
                :class:`~agentflow.core.fault_tolerance.Bulkhead`.
            dead_letter_queue: Queue that receives steps failing after all
                retries are exhausted.
        """
        super().__init__(agent_registry=engine._agents)
        # Share the wrapped engine's workflow store so both views stay in sync.
        self._workflows = engine._workflows

        self._retry_policy = retry_policy
        self._retry_executor = RetryExecutor(retry_policy)
        self._circuit_breaker = CircuitBreaker("workflow-step", circuit_breaker_config)
        self._bulkhead = Bulkhead("workflow-step", bulkhead_config)
        self._dlq = dead_letter_queue

        self._log = logger.bind(component="ResilientWorkflowEngine")

    # ------------------------------------------------------------------
    # Overridden step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        workflow: WorkflowState,
        step: StepState,
        previous_output: dict[str, Any] | None,
    ) -> StepResult:
        """Execute a step with bulkhead, circuit-breaker, and retry protection.

        The resilience layers are applied in order::

            Bulkhead -> Circuit Breaker -> Retry -> Agent.execute()

        If the agent call fails and all retry attempts are exhausted the step
        is recorded in the dead-letter queue before being marked as failed.

        Args:
            workflow: Parent workflow state.
            step: The step to execute.
            previous_output: Output from the preceding step or batch, or
                ``None`` for the first step.

        Returns:
            :class:`~agentflow.core.engine.StepResult` capturing whether the
            step succeeded or failed, including any error detail.
        """
        step.mark_running()

        await self._log.ainfo(
            "resilient_step_started",
            workflow_id=str(workflow.id),
            step_id=str(step.id),
            agent_type=step.agent_type,
        )

        agent = self._agents.get(step.agent_type)
        if agent is None:
            error_msg = f"No agent registered for type '{step.agent_type}'"
            step.mark_failed(error_msg)
            return StepResult(step_id=step.id, status="failed", error=error_msg)

        context = AgentContext(
            run_id=workflow.id,
            step_id=step.id,
            inputs=step.input_data or {},
            metadata={"previous_output": previous_output, **workflow.config},
        )

        async def _call_agent() -> AgentResult:
            return await agent.execute(context)

        async def _with_retry() -> AgentResult:
            return await self._retry_executor.execute(_call_agent)

        async def _with_circuit_breaker() -> AgentResult:
            return await self._circuit_breaker.execute(_with_retry)

        async def _with_bulkhead() -> AgentResult:
            return await self._bulkhead.execute(_with_circuit_breaker)

        try:
            result: AgentResult = await _with_bulkhead()

            if result.error:
                step.mark_failed(result.error)
                await self._log.awarning(
                    "step_agent_error",
                    workflow_id=str(workflow.id),
                    step_id=str(step.id),
                    error=result.error,
                )
                return StepResult(
                    step_id=step.id,
                    status="failed",
                    error=result.error,
                    duration_ms=step.duration_ms,
                )

            step.mark_completed(
                result.output if isinstance(result.output, dict) else {"result": result.output}
            )
            await self._log.ainfo(
                "resilient_step_completed",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
                duration_ms=step.duration_ms,
            )
            return StepResult(
                step_id=step.id,
                status="completed",
                output=result.output,
                duration_ms=step.duration_ms,
            )

        except RetryExhaustedError as exc:
            error_msg = str(exc)
            step.mark_failed(error_msg)
            await self._log.aerror(
                "step_retries_exhausted",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
                attempts=exc.attempts,
                last_error=str(exc.last_error),
            )
            try:
                await self._dlq.add(
                    task_id=str(step.id),
                    error=error_msg,
                    payload={
                        "workflow_id": str(workflow.id),
                        "agent_type": step.agent_type,
                        "input_data": step.input_data or {},
                    },
                    workflow_id=workflow.id,
                    retry_count=exc.attempts,
                    max_retries_reached=True,
                )
            except OverflowError:
                await self._log.aerror(
                    "dlq_overflow_on_step_failure",
                    workflow_id=str(workflow.id),
                    step_id=str(step.id),
                )
            return StepResult(
                step_id=step.id,
                status="failed",
                error=error_msg,
                duration_ms=step.duration_ms,
            )

        except CircuitOpenError as exc:
            error_msg = f"Circuit breaker open: {exc}"
            step.mark_failed(error_msg)
            await self._log.awarning(
                "step_circuit_open",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
            )
            return StepResult(
                step_id=step.id,
                status="failed",
                error=error_msg,
                duration_ms=step.duration_ms,
            )

        except BulkheadFullError as exc:
            error_msg = f"Bulkhead full: {exc}"
            step.mark_failed(error_msg)
            await self._log.awarning(
                "step_bulkhead_full",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
            )
            return StepResult(
                step_id=step.id,
                status="failed",
                error=error_msg,
                duration_ms=step.duration_ms,
            )

        except Exception as exc:
            error_msg = str(exc)
            step.mark_failed(error_msg)
            await self._log.aexception(
                "step_unexpected_error",
                workflow_id=str(workflow.id),
                step_id=str(step.id),
            )
            return StepResult(
                step_id=step.id,
                status="failed",
                error=error_msg,
                duration_ms=step.duration_ms,
            )

    # ------------------------------------------------------------------
    # Status / health inspection
    # ------------------------------------------------------------------

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Return the current circuit breaker state and counters.

        Returns:
            Dictionary with keys:

            * ``state`` — ``"closed"``, ``"open"``, or ``"half_open"``.
            * ``failure_count`` — consecutive failures recorded.
            * ``success_count`` — consecutive successes recorded.
            * ``is_open`` — ``True`` when the circuit is rejecting calls.
            * ``is_half_open`` — ``True`` when the circuit is in recovery mode.
            * ``is_closed`` — ``True`` when the circuit is healthy.
        """
        cb = self._circuit_breaker
        return {
            "state": cb.state.value,
            "failure_count": cb.failure_count,
            "success_count": cb.success_count,
            "is_open": cb.is_open,
            "is_half_open": cb.is_half_open,
            "is_closed": cb.is_closed,
        }

    def get_bulkhead_status(self) -> dict[str, Any]:
        """Return current bulkhead concurrency counters.

        Returns:
            Dictionary with keys:

            * ``active_count`` — calls currently executing.
            * ``queued_count`` — callers waiting for a slot.
            * ``max_concurrent`` — configured concurrency limit.
            * ``max_queue_size`` — configured queue depth limit.
        """
        bh = self._bulkhead
        return {
            "active_count": bh.active_count,
            "queued_count": bh.queued_count,
            "max_concurrent": bh._cfg.max_concurrent,
            "max_queue_size": bh._cfg.max_queue_size,
        }

    async def get_dlq_entries(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DeadLetterEntry]:
        """Return a paginated list of dead-letter queue entries.

        Args:
            limit: Maximum number of entries to return. Defaults to ``100``.
            offset: Number of entries to skip from the start. Defaults to ``0``.

        Returns:
            A list of at most *limit* :class:`~agentflow.core.fault_tolerance.DeadLetterEntry`
            objects in insertion order.
        """
        return await self._dlq.list_entries(limit=limit, offset=offset)

    async def retry_dlq_entry(self, entry_id: UUID) -> bool:
        """Remove a dead-letter entry and mark it for retry.

        The entry is removed from the DLQ. Re-submission to the workflow
        execution pipeline is the caller's responsibility.

        Args:
            entry_id: UUID of the DLQ entry to mark for retry.

        Returns:
            ``True`` if the entry was found and removed; ``False`` otherwise.
        """
        found = await self._dlq.retry_entry(entry_id)
        if found:
            await self._log.ainfo("dlq_entry_marked_for_retry", entry_id=str(entry_id))
        else:
            await self._log.awarning("dlq_entry_not_found", entry_id=str(entry_id))
        return found

    async def get_health(self) -> dict[str, Any]:
        """Return a summary of all fault-tolerance component states.

        The top-level ``status`` field is computed as:

        * ``"healthy"`` — circuit closed and bulkhead has no queued waiters.
        * ``"degraded"`` — circuit is half-open or callers are queued in the
          bulkhead.
        * ``"unhealthy"`` — circuit is open and rejecting all calls.

        Returns:
            Dictionary with keys: ``status``, ``circuit_breaker``,
            ``bulkhead``, ``dead_letter_queue``, and ``retry_policy``.
        """
        cb_status = self.get_circuit_breaker_status()
        bh_status = self.get_bulkhead_status()
        dlq_size = await self._dlq.size()

        if cb_status["is_open"]:
            overall = "unhealthy"
        elif cb_status["is_half_open"] or bh_status["queued_count"] > 0:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "circuit_breaker": cb_status,
            "bulkhead": bh_status,
            "dead_letter_queue": {
                "size": dlq_size,
                "max_size": self._dlq._max_size,
            },
            "retry_policy": {
                "max_retries": self._retry_policy.max_retries,
                "base_delay": self._retry_policy.base_delay,
                "max_delay": self._retry_policy.max_delay,
            },
        }
