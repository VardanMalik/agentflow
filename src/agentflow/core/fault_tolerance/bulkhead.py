"""Bulkhead pattern for limiting concurrent resource usage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import structlog

from agentflow.core.exceptions import BulkheadFullError

logger = structlog.get_logger(__name__)


@dataclass
class BulkheadConfig:
    """Configuration for a :class:`Bulkhead`.

    Attributes:
        max_concurrent: Maximum number of calls that may execute concurrently.
            Defaults to ``10``.
        max_queue_size: Maximum number of callers that may wait for a slot.
            Defaults to ``100``.
        timeout: Seconds to wait for a semaphore slot before raising
            :class:`~agentflow.core.exceptions.BulkheadFullError`.
            Defaults to ``30.0``.
    """

    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout: float = 30.0


class Bulkhead:
    """Limit concurrency and queue depth for a group of async calls.

    Callers beyond *max_concurrent* are queued up to *max_queue_size*. Any
    caller that would exceed the queue limit receives a
    :class:`~agentflow.core.exceptions.BulkheadFullError` immediately.

    Usage::

        config = BulkheadConfig(max_concurrent=5, max_queue_size=20)
        bulkhead = Bulkhead("db-pool", config)
        result = await bulkhead.execute(fetch_from_db, query)
    """

    def __init__(self, name: str, config: BulkheadConfig | None = None) -> None:
        self._name = name
        self._cfg = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self._cfg.max_concurrent)
        self._active_count: int = 0
        self._queued_count: int = 0
        self._lock = asyncio.Lock()
        self._log = logger.bind(bulkhead=name)

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of calls currently executing."""
        return self._active_count

    @property
    def queued_count(self) -> int:
        """Number of callers waiting for a slot."""
        return self._queued_count

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* within the bulkhead constraints.

        Args:
            func: Async callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            BulkheadFullError: When both the concurrency limit and the queue
                are saturated.
            asyncio.TimeoutError: When the caller times out waiting for a slot
                (re-raised as :class:`~agentflow.core.exceptions.BulkheadFullError`).
        """
        async with self._lock:
            total_waiting = self._active_count + self._queued_count
            if total_waiting >= self._cfg.max_concurrent + self._cfg.max_queue_size:
                self._log.warning(
                    "bulkhead_rejected",
                    active=self._active_count,
                    queued=self._queued_count,
                    max_concurrent=self._cfg.max_concurrent,
                    max_queue_size=self._cfg.max_queue_size,
                )
                raise BulkheadFullError(self._name)

            self._queued_count += 1

        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._cfg.timeout,
                )
            except asyncio.TimeoutError:
                async with self._lock:
                    self._queued_count -= 1
                self._log.warning(
                    "bulkhead_timeout",
                    timeout=self._cfg.timeout,
                    active=self._active_count,
                    queued=self._queued_count,
                )
                raise BulkheadFullError(self._name)

            async with self._lock:
                self._queued_count -= 1
                self._active_count += 1

            try:
                return await func(*args, **kwargs)
            finally:
                async with self._lock:
                    self._active_count -= 1
                self._semaphore.release()
        except BulkheadFullError:
            raise
        except Exception:
            raise
