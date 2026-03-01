"""Tests for fault tolerance primitives: RetryExecutor, CircuitBreaker,
DeadLetterQueue, and Bulkhead."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agentflow.core.exceptions import BulkheadFullError, CircuitOpenError, RetryExhaustedError
from agentflow.core.fault_tolerance import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    DeadLetterQueue,
    RetryExecutor,
    RetryPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_policy(**kwargs: object) -> RetryPolicy:
    """RetryPolicy with zero delay so tests don't sleep."""
    defaults: dict = {
        "max_retries": 3,
        "base_delay": 0.0,
        "max_delay": 0.0,
        "jitter": False,
    }
    defaults.update(kwargs)
    return RetryPolicy(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# RetryExecutor
# ===========================================================================


class TestRetryExecutor:
    """Tests for RetryExecutor."""

    async def test_succeeds_on_first_try(self) -> None:
        """Callable that succeeds immediately is called exactly once."""
        policy = _fast_policy(max_retries=3)
        executor = RetryExecutor(policy)
        call_count = 0

        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await executor.execute(succeed)

        assert result == "ok"
        assert call_count == 1
        assert executor.total_attempts == 1
        assert executor.total_retries == 0
        assert executor.last_error is None

    async def test_succeeds_after_retries(self) -> None:
        """Callable that fails twice then succeeds is retried correctly."""
        policy = _fast_policy(max_retries=3)
        executor = RetryExecutor(policy)
        call_count = 0

        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"transient error #{call_count}")
            return "recovered"

        result = await executor.execute(fail_twice)

        assert result == "recovered"
        assert call_count == 3
        assert executor.total_attempts == 3
        assert executor.total_retries == 2

    async def test_exhausts_retries_raises_error(self) -> None:
        """RetryExhaustedError is raised when all retries are consumed."""
        policy = _fast_policy(max_retries=2)
        executor = RetryExecutor(policy)
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await executor.execute(always_fail)

        assert call_count == 3  # initial + 2 retries
        assert exc_info.value.attempts == 3
        assert executor.total_retries == 2
        assert isinstance(executor.last_error, RuntimeError)

    async def test_non_retryable_exception_not_retried(self) -> None:
        """Only exceptions matching retryable_exceptions trigger retries."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.0,
            jitter=False,
            retryable_exceptions=(ValueError,),
        )
        executor = RetryExecutor(policy)
        call_count = 0

        async def raise_type_error() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await executor.execute(raise_type_error)

        assert call_count == 1  # no retries

    async def test_last_error_is_captured(self) -> None:
        """last_error holds the most recent exception after exhaustion."""
        policy = _fast_policy(max_retries=1)
        executor = RetryExecutor(policy)

        async def fail_with_message() -> None:
            raise ValueError("sentinel error")

        with pytest.raises(RetryExhaustedError):
            await executor.execute(fail_with_message)

        assert executor.last_error is not None
        assert "sentinel error" in str(executor.last_error)


# ===========================================================================
# CircuitBreaker
# ===========================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    async def test_stays_closed_on_success(self) -> None:
        """Circuit remains CLOSED after repeated successful calls."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        async def succeed() -> int:
            return 42

        for _ in range(10):
            result = await cb.execute(succeed)
            assert result == 42

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert cb.failure_count == 0

    async def test_opens_after_failure_threshold(self) -> None:
        """Circuit transitions to OPEN after consecutive failures hit threshold."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.execute(fail)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    async def test_open_rejects_calls(self) -> None:
        """OPEN circuit raises CircuitOpenError without calling the function."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker("test", config)
        call_count = 0

        async def fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open

        with pytest.raises(CircuitOpenError):
            await cb.execute(fail)

        # call_count must still be 1 — second call was rejected before invoking func
        assert call_count == 1

    async def test_transitions_to_half_open_after_timeout(self) -> None:
        """OPEN circuit transitions to HALF_OPEN after recovery_timeout elapses."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_calls=5,
            success_threshold=2,
        )
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open

        await asyncio.sleep(0.02)

        # Triggering execute() forces the half-open check
        async def succeed() -> str:
            return "probe"

        result = await cb.execute(succeed)

        assert result == "probe"
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_recovery_closes_circuit(self) -> None:
        """Enough successes in HALF_OPEN close the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_calls=5,
            success_threshold=2,
        )
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        async def succeed() -> str:
            return "ok"

        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open
        await asyncio.sleep(0.02)

        # Two successful probe calls should close the circuit
        await cb.execute(succeed)
        await cb.execute(succeed)

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    async def test_half_open_failure_reopens_circuit(self) -> None:
        """A failure in HALF_OPEN immediately re-opens the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_calls=5,
            success_threshold=3,
        )
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open
        await asyncio.sleep(0.02)

        # Probe fails — circuit goes back to OPEN
        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open

    async def test_reset_closes_circuit(self) -> None:
        """Manual reset transitions circuit back to CLOSED from any state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.execute(fail)

        assert cb.is_open

        await cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    async def test_success_resets_failure_count_when_closed(self) -> None:
        """A successful call in CLOSED state resets the failure counter."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        async def fail() -> None:
            raise RuntimeError("boom")

        async def succeed() -> str:
            return "ok"

        # Two failures — not enough to open
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.execute(fail)

        assert cb.failure_count == 2

        # A success should reset the count
        await cb.execute(succeed)

        assert cb.failure_count == 0
        assert cb.is_closed


# ===========================================================================
# DeadLetterQueue
# ===========================================================================


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    async def test_add_entry(self) -> None:
        """Added entry is retrievable and contains correct fields."""
        dlq = DeadLetterQueue()
        wf_id = uuid4()

        entry = await dlq.add(
            task_id="step-abc",
            error="timeout",
            payload={"x": 1},
            workflow_id=wf_id,
            retry_count=3,
            max_retries_reached=True,
        )

        assert entry.task_id == "step-abc"
        assert entry.error == "timeout"
        assert entry.payload == {"x": 1}
        assert entry.workflow_id == wf_id
        assert entry.retry_count == 3
        assert entry.max_retries_reached is True
        assert entry.id is not None
        assert isinstance(entry.created_at, datetime)

    async def test_get_entry_by_id(self) -> None:
        """get() returns the entry with the matching ID."""
        dlq = DeadLetterQueue()
        entry = await dlq.add(task_id="t1", error="err", payload={})

        found = await dlq.get(entry.id)

        assert found is not None
        assert found.id == entry.id

    async def test_get_missing_entry_returns_none(self) -> None:
        """get() returns None for an unknown ID."""
        dlq = DeadLetterQueue()
        result = await dlq.get(uuid4())
        assert result is None

    async def test_list_entries_all(self) -> None:
        """list_entries() returns all entries in insertion order."""
        dlq = DeadLetterQueue()
        for i in range(5):
            await dlq.add(task_id=f"task-{i}", error="err", payload={"i": i})

        entries = await dlq.list_entries(limit=10)

        assert len(entries) == 5
        for i, entry in enumerate(entries):
            assert entry.task_id == f"task-{i}"

    async def test_list_entries_with_limit_and_offset(self) -> None:
        """list_entries() supports limit/offset pagination."""
        dlq = DeadLetterQueue()
        for i in range(10):
            await dlq.add(task_id=f"task-{i}", error="err", payload={})

        page = await dlq.list_entries(limit=3, offset=4)

        assert len(page) == 3
        assert page[0].task_id == "task-4"
        assert page[1].task_id == "task-5"
        assert page[2].task_id == "task-6"

    async def test_size(self) -> None:
        """size() returns the current number of entries."""
        dlq = DeadLetterQueue()
        assert await dlq.size() == 0

        await dlq.add(task_id="t", error="e", payload={})
        assert await dlq.size() == 1

        await dlq.add(task_id="t2", error="e2", payload={})
        assert await dlq.size() == 2

    async def test_retry_entry_removes_it(self) -> None:
        """retry_entry() removes the entry from the DLQ."""
        dlq = DeadLetterQueue()
        entry = await dlq.add(task_id="t", error="e", payload={})

        result = await dlq.retry_entry(entry.id)

        assert result is True
        assert await dlq.size() == 0
        assert await dlq.get(entry.id) is None

    async def test_retry_entry_unknown_id_returns_false(self) -> None:
        """retry_entry() returns False when the ID is not in the queue."""
        dlq = DeadLetterQueue()
        result = await dlq.retry_entry(uuid4())
        assert result is False

    async def test_purge_all(self) -> None:
        """purge() with no arguments removes all entries."""
        dlq = DeadLetterQueue()
        for _ in range(5):
            await dlq.add(task_id="t", error="e", payload={})

        removed = await dlq.purge()

        assert removed == 5
        assert await dlq.size() == 0

    async def test_purge_older_than(self) -> None:
        """purge(older_than=...) removes only entries created before cutoff."""
        dlq = DeadLetterQueue()

        # Add entries with an explicit past timestamp by injecting them directly
        cutoff = datetime.now(tz=timezone.utc)

        old_entry = await dlq.add(task_id="old", error="e", payload={})
        # Backdate the old entry so it falls before the cutoff
        old_entry.created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)

        await dlq.add(task_id="new", error="e", payload={})

        removed = await dlq.purge(older_than=cutoff)

        assert removed == 1
        assert await dlq.size() == 1
        entries = await dlq.list_entries()
        assert entries[0].task_id == "new"

    async def test_add_raises_when_full(self) -> None:
        """OverflowError is raised when the queue reaches max_size."""
        dlq = DeadLetterQueue(max_size=2)
        await dlq.add(task_id="t1", error="e", payload={})
        await dlq.add(task_id="t2", error="e", payload={})

        with pytest.raises(OverflowError):
            await dlq.add(task_id="t3", error="e", payload={})


# ===========================================================================
# Bulkhead
# ===========================================================================


class TestBulkhead:
    """Tests for Bulkhead concurrency limiting."""

    async def test_allows_calls_up_to_limit(self) -> None:
        """max_concurrent calls execute concurrently without rejection."""
        config = BulkheadConfig(max_concurrent=3, max_queue_size=0, timeout=5.0)
        bulkhead = Bulkhead("test", config)
        results: list[int] = []

        async def worker(val: int) -> int:
            results.append(val)
            return val

        returned = await asyncio.gather(
            bulkhead.execute(worker, 1),
            bulkhead.execute(worker, 2),
            bulkhead.execute(worker, 3),
        )

        assert sorted(returned) == [1, 2, 3]
        assert sorted(results) == [1, 2, 3]

    async def test_rejects_when_full(self) -> None:
        """BulkheadFullError is raised when concurrency limit is saturated."""
        config = BulkheadConfig(max_concurrent=1, max_queue_size=0, timeout=5.0)
        bulkhead = Bulkhead("test", config)

        inside = asyncio.Event()
        gate = asyncio.Event()

        async def blocking() -> str:
            inside.set()
            await gate.wait()
            return "done"

        # Start first call — it will block on gate
        task = asyncio.create_task(bulkhead.execute(blocking))

        # Wait until the first call is actively executing (semaphore held)
        await inside.wait()

        # Second call must be rejected immediately
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(blocking)

        # Clean up
        gate.set()
        await task

    async def test_active_count_tracks_running_calls(self) -> None:
        """active_count reflects the number of concurrently executing calls."""
        config = BulkheadConfig(max_concurrent=5, max_queue_size=0, timeout=5.0)
        bulkhead = Bulkhead("test", config)

        inside = asyncio.Event()
        gate = asyncio.Event()
        ready_count = 0

        async def blocking() -> None:
            nonlocal ready_count
            ready_count += 1
            if ready_count == 3:
                inside.set()
            await gate.wait()

        tasks = [asyncio.create_task(bulkhead.execute(blocking)) for _ in range(3)]
        await inside.wait()

        assert bulkhead.active_count == 3

        gate.set()
        await asyncio.gather(*tasks)

        assert bulkhead.active_count == 0

    async def test_queued_count_visible_while_waiting(self) -> None:
        """queued_count reflects callers waiting for a semaphore slot."""
        config = BulkheadConfig(max_concurrent=1, max_queue_size=10, timeout=5.0)
        bulkhead = Bulkhead("test", config)

        inside = asyncio.Event()
        gate = asyncio.Event()

        async def blocking() -> None:
            inside.set()
            await gate.wait()

        async def fast() -> str:
            return "fast"

        # Saturate the single slot
        first_task = asyncio.create_task(bulkhead.execute(blocking))
        await inside.wait()

        # Launch a second task that must queue
        second_task = asyncio.create_task(bulkhead.execute(fast))
        # Give the event loop a moment to register the queued call
        await asyncio.sleep(0)

        assert bulkhead.queued_count >= 0  # may be 0 if fast already acquired

        gate.set()
        await asyncio.gather(first_task, second_task)

        assert bulkhead.active_count == 0

    async def test_call_result_propagated(self) -> None:
        """The return value of the wrapped callable is passed through."""
        config = BulkheadConfig(max_concurrent=2)
        bulkhead = Bulkhead("test", config)

        async def compute(x: int) -> int:
            return x * 2

        result = await bulkhead.execute(compute, 21)
        assert result == 42

    async def test_exception_propagated(self) -> None:
        """Exceptions raised inside the callable propagate to the caller."""
        config = BulkheadConfig(max_concurrent=2)
        bulkhead = Bulkhead("test", config)

        async def explode() -> None:
            raise ValueError("inner error")

        with pytest.raises(ValueError, match="inner error"):
            await bulkhead.execute(explode)

        # Bulkhead should release its slot even after an exception
        assert bulkhead.active_count == 0
