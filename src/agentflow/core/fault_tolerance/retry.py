"""Retry policy and executor with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from agentflow.core.exceptions import RetryExhaustedError

logger = structlog.get_logger(__name__)


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts (not counting the
            initial attempt). Defaults to ``3``.
        base_delay: Initial backoff delay in seconds. Defaults to ``1.0``.
        max_delay: Upper bound for the computed backoff delay. Defaults to
            ``60.0``.
        exponential_base: Base used in the exponential backoff formula.
            Defaults to ``2.0``.
        jitter: When ``True``, adds a random 0–50 % fraction to the computed
            delay to prevent thundering-herd effects. Defaults to ``True``.
        retryable_exceptions: Tuple of exception types that trigger a retry.
            Defaults to ``(Exception,)`` — all exceptions are retried.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def delay_for(self, attempt: int) -> float:
        """Return the backoff delay for the given attempt index (0-based).

        Args:
            attempt: Zero-based retry attempt index.

        Returns:
            Delay in seconds, capped at :attr:`max_delay`.
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        if self.jitter:
            delay *= 1.0 + random.random() * 0.5  # noqa: S311
        return delay


class RetryExecutor:
    """Wraps an async callable with retry logic governed by a :class:`RetryPolicy`.

    Usage::

        policy = RetryPolicy(max_retries=3, base_delay=0.5)
        executor = RetryExecutor(policy)
        result = await executor.execute(my_async_func, arg1, kwarg=value)

    Attributes:
        total_attempts: Total number of calls made (including the first try).
        total_retries: Number of times a retry was triggered.
        last_error: The most recent exception caught, or ``None``.
    """

    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy
        self.total_attempts: int = 0
        self.total_retries: int = 0
        self.last_error: BaseException | None = None

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* with retry logic.

        Args:
            func: Async callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func* on success.

        Raises:
            RetryExhaustedError: After all retries are exhausted, wrapping the
                last exception.
        """
        policy = self._policy
        log = logger.bind(func=getattr(func, "__name__", repr(func)))

        for attempt in range(policy.max_retries + 1):
            self.total_attempts += 1
            try:
                return await func(*args, **kwargs)
            except policy.retryable_exceptions as exc:
                self.last_error = exc
                if attempt >= policy.max_retries:
                    log.error(
                        "retry_exhausted",
                        attempt=attempt + 1,
                        max_retries=policy.max_retries,
                        error=str(exc),
                    )
                    raise RetryExhaustedError(
                        attempts=self.total_attempts, last_error=exc
                    ) from exc

                self.total_retries += 1
                delay = policy.delay_for(attempt)
                log.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 3),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

        # Unreachable, but satisfies type checkers.
        raise RetryExhaustedError(attempts=self.total_attempts, last_error=self.last_error)
