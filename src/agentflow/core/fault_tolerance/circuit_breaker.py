"""Circuit breaker pattern for protecting downstream dependencies."""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from agentflow.core.exceptions import CircuitOpenError

logger = structlog.get_logger(__name__)


class CircuitState(enum.Enum):
    """States of the circuit breaker state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a :class:`CircuitBreaker`.

    Attributes:
        failure_threshold: Number of consecutive failures required to open the
            circuit. Defaults to ``5``.
        recovery_timeout: Seconds to wait in the OPEN state before allowing a
            probe call (transition to HALF_OPEN). Defaults to ``30.0``.
        half_open_max_calls: Maximum concurrent calls permitted in HALF_OPEN
            state. Defaults to ``3``.
        success_threshold: Consecutive successes in HALF_OPEN required to close
            the circuit again. Defaults to ``2``.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Async circuit breaker protecting a downstream resource.

    Transitions:

    * **CLOSED** → **OPEN** when consecutive failure count reaches
      :attr:`~CircuitBreakerConfig.failure_threshold`.
    * **OPEN** → **HALF_OPEN** after
      :attr:`~CircuitBreakerConfig.recovery_timeout` seconds.
    * **HALF_OPEN** → **CLOSED** on
      :attr:`~CircuitBreakerConfig.success_threshold` consecutive successes.
    * **HALF_OPEN** → **OPEN** on any failure.

    All state transitions are protected by an :class:`asyncio.Lock`.

    Usage::

        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("my-service", config)
        result = await cb.execute(call_my_service, arg)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._name = name
        self._cfg = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()
        self._log = logger.bind(circuit_breaker=name)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* through the circuit breaker.

        Args:
            func: Async callable to protect.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitOpenError: When the circuit is OPEN and rejecting calls.
            Exception: Any exception raised by *func* is re-raised after
                recording the failure.
        """
        async with self._lock:
            await self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(self._name)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._cfg.half_open_max_calls:
                    raise CircuitOpenError(self._name)
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
        except Exception:
            async with self._lock:
                await self._record_failure()
            raise

        async with self._lock:
            await self._record_success()

        return result

    # ------------------------------------------------------------------
    # Manual reset
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._log.info("circuit_reset")

    # ------------------------------------------------------------------
    # Internal helpers (must be called under lock)
    # ------------------------------------------------------------------

    async def _maybe_transition_to_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if self._opened_at is None:
            return
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._cfg.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._half_open_calls = 0
            self._log.info(
                "circuit_state_transition",
                from_state=CircuitState.OPEN.value,
                to_state=CircuitState.HALF_OPEN.value,
                elapsed_seconds=round(elapsed, 2),
            )

    async def _record_failure(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            await self._open_circuit()
            return

        self._failure_count += 1
        if self._failure_count >= self._cfg.failure_threshold:
            await self._open_circuit()

    async def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._cfg.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._opened_at = None
                self._log.info(
                    "circuit_state_transition",
                    from_state=CircuitState.HALF_OPEN.value,
                    to_state=CircuitState.CLOSED.value,
                )
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    async def _open_circuit(self) -> None:
        previous = self._state.value
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._success_count = 0
        self._half_open_calls = 0
        self._log.warning(
            "circuit_state_transition",
            from_state=previous,
            to_state=CircuitState.OPEN.value,
            failure_count=self._failure_count,
        )
