"""Celery task infrastructure with retry logic and instrumentation."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

import structlog
from celery import Task as CeleryTask

from agentflow.config import get_settings
from agentflow.core.celery_app import celery_app

logger = structlog.get_logger()

F = TypeVar("F", bound=Callable[..., Any])


class BaseTask(CeleryTask):
    """Base task class with automatic retry on transient failures.

    Subclass this for tasks that need custom retry behaviour or
    shared setup/teardown logic.
    """

    abstract = True
    autoretry_for = (ConnectionError, TimeoutError, OSError)
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    @property
    def max_retries(self) -> int:
        settings = get_settings()
        return settings.celery_task_max_retries

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        logger.error(
            "Task failed",
            task_name=self.name,
            task_id=task_id,
            error=str(exc),
        )

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        logger.warning(
            "Task retrying",
            task_name=self.name,
            task_id=task_id,
            retry=self.request.retries,
            error=str(exc),
        )

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.info(
            "Task succeeded",
            task_name=self.name,
            task_id=task_id,
        )


def handle_task_errors(func: F) -> F:
    """Decorator that catches unexpected exceptions and logs them.

    Lets Celery's built-in retry machinery handle retryable errors
    while ensuring non-retryable ones are logged with full context.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # The bound task instance is the first arg when using bind=True
        task: CeleryTask | None = args[0] if args and isinstance(args[0], CeleryTask) else None
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError):
            raise  # Let autoretry_for handle these
        except Exception:
            logger.exception(
                "Unhandled task error",
                task_name=task.name if task else func.__name__,
                task_id=task.request.id if task else None,
            )
            raise

    return wrapper  # type: ignore[return-value]


def timed_task(func: F) -> F:
    """Decorator that logs task execution duration in milliseconds."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        task: CeleryTask | None = args[0] if args and isinstance(args[0], CeleryTask) else None
        task_name = task.name if task else func.__name__
        start = time.monotonic()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "Task timing",
                task_name=task_name,
                duration_ms=duration_ms,
            )

    return wrapper  # type: ignore[return-value]


__all__ = [
    "BaseTask",
    "celery_app",
    "handle_task_errors",
    "timed_task",
]
