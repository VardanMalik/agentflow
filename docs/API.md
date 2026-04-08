# API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive documentation (when `DEBUG=true`):
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Authentication

Authentication is not yet enforced. A future release will add API key validation via the `X-API-Key` header. The header name is configurable via the `API_KEY_HEADER` environment variable.

---

## Health

### `GET /health`

Basic health check.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development"
}
```

### `GET /health/ready`

Readiness check — verifies database and Redis connectivity.

**Response** `200 OK`

```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "redis": "ok"
  }
}
```

**Response** `503 Service Unavailable`

```json
{
  "status": "not_ready",
  "checks": {
    "database": "ok",
    "redis": "error: Connection refused"
  }
}
```

---

## Workflows

### `POST /workflows`

Create a new workflow.

**Request Body**

```json
{
  "name": "Research and Report",
  "description": "Research a topic and generate a report",
  "steps": [
    {
      "agent_type": "research",
      "input_data": {
        "topic": "distributed systems consensus algorithms",
        "depth": "detailed"
      }
    },
    {
      "agent_type": "analysis",
      "input_data": {
        "analysis_type": "comparative"
      }
    },
    {
      "agent_type": "writer",
      "input_data": {
        "tone": "technical",
        "format": "report"
      }
    }
  ],
  "config": {}
}
```

**Response** `201 Created`

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Research and Report",
  "description": "Research a topic and generate a report",
  "status": "pending",
  "steps": [
    {
      "id": "step-uuid-1",
      "step_order": 0,
      "agent_type": "research",
      "status": "pending",
      "input_data": {"topic": "distributed systems consensus algorithms", "depth": "detailed"},
      "output_data": null,
      "error": null
    },
    {
      "id": "step-uuid-2",
      "step_order": 1,
      "agent_type": "analysis",
      "status": "pending",
      "input_data": {"analysis_type": "comparative"},
      "output_data": null,
      "error": null
    },
    {
      "id": "step-uuid-3",
      "step_order": 2,
      "agent_type": "writer",
      "status": "pending",
      "input_data": {"tone": "technical", "format": "report"},
      "output_data": null,
      "error": null
    }
  ],
  "created_at": "2026-04-08T12:00:00Z",
  "updated_at": "2026-04-08T12:00:00Z"
}
```

**Validation Errors** `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "name"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

### `GET /workflows`

List workflows with pagination.

**Query Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 20 | Page size (max 100) |
| `offset` | int | 0 | Offset for pagination |
| `status` | string | — | Filter by status: `pending`, `running`, `completed`, `failed`, `cancelled` |

**Response** `200 OK`

```json
{
  "items": [
    {
      "id": "uuid",
      "name": "Research and Report",
      "status": "completed",
      "created_at": "2026-04-08T12:00:00Z",
      "completed_at": "2026-04-08T12:05:00Z",
      "step_count": 3,
      "completed_steps": 3,
      "failed_steps": 0
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### `GET /workflows/{workflow_id}`

Get workflow details including all step data.

**Response** `200 OK` — Full workflow object (same shape as POST response, with step outputs populated).

**Response** `404 Not Found`

```json
{
  "detail": "Workflow not found"
}
```

### `GET /workflows/{workflow_id}/status`

Get workflow execution progress.

**Response** `200 OK`

```json
{
  "workflow_id": "uuid",
  "status": "running",
  "progress_pct": 66.7,
  "total_steps": 3,
  "completed_steps": 2,
  "failed_steps": 0,
  "started_at": "2026-04-08T12:00:00Z",
  "elapsed_ms": 120000
}
```

### `POST /workflows/{workflow_id}/execute`

Start workflow execution. The workflow must be in `pending` or `failed` status.

**Response** `202 Accepted`

```json
{
  "workflow_id": "uuid",
  "status": "running",
  "message": "Workflow execution started"
}
```

**Response** `409 Conflict`

```json
{
  "detail": "Workflow is already running"
}
```

### `POST /workflows/{workflow_id}/cancel`

Cancel a running workflow. Pending steps are marked as `cancelled`.

**Response** `200 OK`

```json
{
  "workflow_id": "uuid",
  "status": "cancelled"
}
```

### `POST /workflows/{workflow_id}/retry`

Retry a failed workflow. Resets failed steps to `pending` and re-executes.

**Response** `202 Accepted`

```json
{
  "workflow_id": "uuid",
  "status": "running",
  "message": "Workflow retry started"
}
```

---

## Agents

### `GET /agents/types`

List available agent types and their descriptions.

**Response** `200 OK`

```json
{
  "agent_types": [
    {
      "type": "research",
      "description": "Research specialist — investigates topics and produces structured findings",
      "input_schema": {
        "topic": "string (required)",
        "depth": "brief | detailed (default: brief)"
      }
    },
    {
      "type": "analysis",
      "description": "Data analyst — identifies patterns, trends, and produces recommendations",
      "input_schema": {
        "data": "string | object (required)",
        "analysis_type": "sentiment | statistical | comparative"
      }
    },
    {
      "type": "writer",
      "description": "Content writer — generates structured content from briefs",
      "input_schema": {
        "brief": "string (required)",
        "tone": "formal | casual | technical",
        "format": "article | report | summary"
      }
    },
    {
      "type": "code",
      "description": "Code generator — produces production-quality code from requirements",
      "input_schema": {
        "requirements": "string (required)",
        "language": "string (default: python)",
        "style": "concise | verbose"
      }
    }
  ]
}
```

### `GET /agents`

List registered agents with pagination.

**Query Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 20 | Page size |
| `offset` | int | 0 | Offset |

**Response** `200 OK`

```json
{
  "items": [
    {
      "id": "uuid",
      "name": "research-agent-1",
      "type": "research",
      "description": "Research specialist",
      "is_active": true,
      "created_at": "2026-04-08T12:00:00Z"
    }
  ],
  "total": 4,
  "limit": 20,
  "offset": 0
}
```

### `GET /agents/{agent_id}`

Get agent details.

**Response** `200 OK`

```json
{
  "id": "uuid",
  "name": "research-agent-1",
  "type": "research",
  "description": "Research specialist",
  "is_active": true,
  "config": {},
  "created_at": "2026-04-08T12:00:00Z"
}
```

### `GET /agents/{agent_id}/executions`

List execution history for an agent.

**Response** `200 OK`

```json
{
  "items": [
    {
      "id": "uuid",
      "agent_id": "uuid",
      "workflow_step_id": "uuid",
      "status": "completed",
      "duration_ms": 4523,
      "tokens_used": 1250,
      "started_at": "2026-04-08T12:01:00Z",
      "completed_at": "2026-04-08T12:01:04Z"
    }
  ],
  "total": 15,
  "limit": 20,
  "offset": 0
}
```

### `POST /agents/{agent_id}/test`

Test an agent with sample input without creating a workflow.

**Request Body**

```json
{
  "input_data": {
    "topic": "quantum computing basics",
    "depth": "brief"
  }
}
```

**Response** `200 OK`

```json
{
  "output": {
    "findings": "Quantum computing leverages quantum mechanics...",
    "key_points": ["Superposition", "Entanglement", "Quantum gates"],
    "sources": ["arxiv.org", "nature.com"]
  },
  "duration_ms": 3200,
  "tokens_used": 850
}
```

---

## Dashboard

### `GET /dashboard/stats`

Aggregated system statistics.

**Response** `200 OK`

```json
{
  "total_workflows": 156,
  "running_workflows": 3,
  "completed_workflows": 140,
  "failed_workflows": 13,
  "success_rate": 89.7,
  "avg_duration_ms": 45200,
  "active_agents": 4,
  "dlq_size": 2
}
```

---

## WebSocket

### `WS /ws`

Subscribe to all system events.

### `WS /ws/{workflow_id}`

Subscribe to events for a specific workflow.

### Event Types

**Workflow Events**

```json
{
  "type": "workflow.started",
  "workflow_id": "uuid",
  "timestamp": "2026-04-08T12:00:00Z"
}
```

```json
{
  "type": "workflow.completed",
  "workflow_id": "uuid",
  "duration_ms": 45200,
  "timestamp": "2026-04-08T12:00:45Z"
}
```

```json
{
  "type": "workflow.failed",
  "workflow_id": "uuid",
  "error": "Step 2 failed: Circuit breaker is open",
  "timestamp": "2026-04-08T12:00:30Z"
}
```

```json
{
  "type": "workflow.cancelled",
  "workflow_id": "uuid",
  "timestamp": "2026-04-08T12:00:15Z"
}
```

**Step Events**

```json
{
  "type": "step.started",
  "workflow_id": "uuid",
  "step_id": "uuid",
  "agent_type": "research",
  "step_order": 0,
  "timestamp": "2026-04-08T12:00:01Z"
}
```

```json
{
  "type": "step.completed",
  "workflow_id": "uuid",
  "step_id": "uuid",
  "agent_type": "research",
  "duration_ms": 3200,
  "timestamp": "2026-04-08T12:00:04Z"
}
```

```json
{
  "type": "step.failed",
  "workflow_id": "uuid",
  "step_id": "uuid",
  "agent_type": "analysis",
  "error": "RetryExhaustedError: max retries (3) exceeded",
  "timestamp": "2026-04-08T12:00:30Z"
}
```

**Agent Events**

```json
{
  "type": "agent.execution_started",
  "agent_type": "research",
  "workflow_id": "uuid",
  "step_id": "uuid",
  "timestamp": "2026-04-08T12:00:01Z"
}
```

```json
{
  "type": "agent.execution_completed",
  "agent_type": "research",
  "duration_ms": 3200,
  "tokens_used": 850,
  "timestamp": "2026-04-08T12:00:04Z"
}
```

**Metrics Events**

```json
{
  "type": "metrics.update",
  "data": {
    "active_workflows": 3,
    "circuit_breaker_state": "closed",
    "bulkhead_active": 2,
    "dlq_size": 0
  },
  "timestamp": "2026-04-08T12:01:00Z"
}
```

---

## Error Responses

All error responses follow a consistent format:

### 400 Bad Request

```json
{
  "detail": "Invalid workflow definition: name is required"
}
```

### 404 Not Found

```json
{
  "detail": "Workflow not found"
}
```

### 409 Conflict

```json
{
  "detail": "Workflow is already running"
}
```

### 422 Unprocessable Entity

```json
{
  "detail": [
    {
      "loc": ["body", "steps", 0, "agent_type"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Service temporarily unavailable",
  "checks": {
    "database": "error: Connection refused",
    "redis": "ok"
  }
}
```
