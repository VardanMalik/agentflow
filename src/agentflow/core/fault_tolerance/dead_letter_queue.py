"""Dead-letter queue for tasks that have exhausted all retry attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class DeadLetterEntry:
    """A single entry in the dead-letter queue.

    Attributes:
        id: Unique identifier for this DLQ entry.
        task_id: The identifier of the failed task.
        workflow_id: The associated workflow ID, if any.
        error: Human-readable description of the terminal error.
        payload: The task payload at the time of failure.
        created_at: UTC timestamp when the entry was created.
        retry_count: Number of retry attempts that were made.
        max_retries_reached: Whether the entry was added because retries
            were exhausted (as opposed to a non-retryable error).
    """

    id: UUID
    task_id: str
    error: str
    payload: dict
    created_at: datetime
    retry_count: int = 0
    max_retries_reached: bool = False
    workflow_id: UUID | None = None


class DeadLetterQueue:
    """In-memory dead-letter queue for failed tasks.

    Entries are stored in insertion order up to *max_size*. When the queue is
    full, new entries are rejected (no silent eviction).

    Usage::

        dlq = DeadLetterQueue(max_size=5000)
        entry = await dlq.add(
            task_id="task-123",
            error="Connection refused",
            payload={"input": "..."},
            retry_count=3,
            max_retries_reached=True,
        )
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._entries: list[DeadLetterEntry] = []
        self._index: dict[UUID, DeadLetterEntry] = {}
        self._retry_set: set[UUID] = set()
        self._log = logger.bind(component="DeadLetterQueue")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add(
        self,
        task_id: str,
        error: str,
        payload: dict,
        workflow_id: UUID | None = None,
        retry_count: int = 0,
        max_retries_reached: bool = False,
    ) -> DeadLetterEntry:
        """Add a failed task to the dead-letter queue.

        Args:
            task_id: Identifier of the failed task.
            error: Description of the terminal error.
            payload: Task payload at the time of failure.
            workflow_id: Associated workflow ID, if any.
            retry_count: Number of retry attempts already made.
            max_retries_reached: Whether failure was due to exhausted retries.

        Returns:
            The newly created :class:`DeadLetterEntry`.

        Raises:
            OverflowError: When the queue has reached *max_size*.
        """
        if len(self._entries) >= self._max_size:
            self._log.error(
                "dlq_full",
                max_size=self._max_size,
                task_id=task_id,
            )
            raise OverflowError(
                f"Dead-letter queue is full (max_size={self._max_size})"
            )

        entry = DeadLetterEntry(
            id=uuid4(),
            task_id=task_id,
            workflow_id=workflow_id,
            error=error,
            payload=payload,
            created_at=datetime.now(tz=timezone.utc),
            retry_count=retry_count,
            max_retries_reached=max_retries_reached,
        )
        self._entries.append(entry)
        self._index[entry.id] = entry

        self._log.info(
            "dlq_entry_added",
            entry_id=str(entry.id),
            task_id=task_id,
            retry_count=retry_count,
            max_retries_reached=max_retries_reached,
        )
        return entry

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get(self, entry_id: UUID) -> DeadLetterEntry | None:
        """Return the entry with the given ID, or ``None`` if not found."""
        return self._index.get(entry_id)

    async def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DeadLetterEntry]:
        """Return a page of DLQ entries in insertion order.

        Args:
            limit: Maximum number of entries to return.
            offset: Number of entries to skip from the start.

        Returns:
            A list of at most *limit* entries.
        """
        return self._entries[offset : offset + limit]

    async def size(self) -> int:
        """Return the current number of entries in the queue."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Retry / purge
    # ------------------------------------------------------------------

    async def retry_entry(self, entry_id: UUID) -> bool:
        """Mark an entry for retry and remove it from the DLQ.

        Args:
            entry_id: ID of the entry to retry.

        Returns:
            ``True`` if the entry was found and removed, ``False`` otherwise.
        """
        entry = self._index.pop(entry_id, None)
        if entry is None:
            return False

        self._entries.remove(entry)
        self._retry_set.add(entry_id)

        self._log.info(
            "dlq_entry_retried",
            entry_id=str(entry_id),
            task_id=entry.task_id,
        )
        return True

    async def purge(self, older_than: datetime | None = None) -> int:
        """Remove entries from the queue.

        Args:
            older_than: When provided, only entries with
                ``created_at < older_than`` are removed. When ``None``,
                *all* entries are removed.

        Returns:
            The number of entries purged.
        """
        if older_than is None:
            count = len(self._entries)
            self._entries.clear()
            self._index.clear()
            self._log.info("dlq_purged", count=count)
            return count

        to_remove = [e for e in self._entries if e.created_at < older_than]
        for entry in to_remove:
            self._entries.remove(entry)
            self._index.pop(entry.id, None)

        self._log.info("dlq_purged", count=len(to_remove), older_than=str(older_than))
        return len(to_remove)
