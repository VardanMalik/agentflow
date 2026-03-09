"""WebSocket support: ConnectionManager, EventBus, and real-time endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agentflow.api.schemas import WebSocketMessage

logger = structlog.get_logger()

router = APIRouter(tags=["websocket"])

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "workflow.started",
    "workflow.completed",
    "workflow.failed",
    "workflow.cancelled",
    "step.started",
    "step.completed",
    "step.failed",
    "agent.execution_started",
    "agent.execution_completed",
    "metrics.update",
]


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections grouped by workflow_id room."""

    def __init__(self) -> None:
        self._global: list[WebSocket] = []
        self._rooms: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, workflow_id: str | None = None) -> None:
        """Accept a WebSocket connection and register it."""
        await websocket.accept()
        if workflow_id:
            self._rooms.setdefault(workflow_id, []).append(websocket)
        else:
            self._global.append(websocket)
        await logger.ainfo("WebSocket connected", workflow_id=workflow_id)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from all rooms."""
        if websocket in self._global:
            self._global.remove(websocket)
            return
        for ws_list in self._rooms.values():
            if websocket in ws_list:
                ws_list.remove(websocket)
                break
        await logger.ainfo("WebSocket disconnected")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to every connected client."""
        all_connections = list(self._global)
        for ws_list in self._rooms.values():
            all_connections.extend(ws_list)
        await self._send_to_many(all_connections, message)

    async def send_to_workflow(self, workflow_id: str, message: dict[str, Any]) -> None:
        """Send a message to global subscribers and clients watching a specific workflow."""
        targets = list(self._global) + list(self._rooms.get(workflow_id, []))
        await self._send_to_many(targets, message)

    async def _send_to_many(
        self, connections: list[WebSocket], message: dict[str, Any]
    ) -> None:
        failed: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                failed.append(ws)
        for ws in failed:
            await self.disconnect(ws)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


_EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class EventBus:
    """Singleton pub/sub system connecting workflow engine events to WebSocket broadcasts."""

    _instance: EventBus | None = None

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_EventCallback]] = {}
        self._sse_queues: list[asyncio.Queue[dict[str, Any]]] = []

    @classmethod
    def get_instance(cls) -> EventBus:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: str, callback: _EventCallback) -> None:
        """Register a callback for the given event type."""
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: _EventCallback) -> None:
        """Remove a previously registered callback."""
        callbacks = self._subscribers.get(event_type, [])
        try:
            callbacks.remove(callback)
        except ValueError:
            pass

    def add_sse_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Register a per-connection SSE queue to receive all published events."""
        self._sse_queues.append(queue)

    def remove_sse_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Deregister an SSE queue."""
        try:
            self._sse_queues.remove(queue)
        except ValueError:
            pass

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast an event to all registered callbacks and SSE queues."""
        for callback in list(self._subscribers.get(event_type, [])):
            try:
                await callback(data)
            except Exception as exc:
                await logger.awarning(
                    "EventBus callback error", event_type=event_type, error=str(exc)
                )
        for queue in list(self._sse_queues):
            try:
                queue.put_nowait({"type": event_type, "data": data})
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

connection_manager = ConnectionManager()
event_bus = EventBus.get_instance()


def _make_ws_handler(event_type: str) -> _EventCallback:
    async def handler(data: dict[str, Any]) -> None:
        msg = WebSocketMessage(
            type=event_type,
            payload=data,
            timestamp=datetime.now(timezone.utc),
        )
        serialised = msg.model_dump(mode="json")
        workflow_id = data.get("workflow_id")
        if workflow_id:
            await connection_manager.send_to_workflow(str(workflow_id), serialised)
        else:
            await connection_manager.broadcast(serialised)

    return handler


for _et in EVENT_TYPES:
    event_bus.subscribe(_et, _make_ws_handler(_et))


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def websocket_global(websocket: WebSocket) -> None:
    """Global WebSocket — receives all system events."""
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/{workflow_id}")
async def websocket_workflow(websocket: WebSocket, workflow_id: str) -> None:
    """Workflow-scoped WebSocket — receives events for a specific workflow."""
    await connection_manager.connect(websocket, workflow_id=workflow_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
