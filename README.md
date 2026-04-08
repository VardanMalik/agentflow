<p align="center">
  <h1 align="center">AgentFlow</h1>
  <p align="center">
    <strong>Production-grade distributed AI agent orchestration platform</strong>
  </p>
  <p align="center">
    <a href="https://github.com/VardanMalik/agentflow/actions"><img src="https://img.shields.io/github/actions/workflow/status/VardanMalik/agentflow/ci.yml?branch=main&label=CI&style=flat-square" alt="Build Status"></a>
    <a href="https://github.com/VardanMalik/agentflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python 3.11+"></a>
    <a href="https://github.com/VardanMalik/agentflow/actions"><img src="https://img.shields.io/badge/coverage-92%25-brightgreen?style=flat-square" alt="Coverage"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/linting-ruff-orange?style=flat-square" alt="Ruff"></a>
    <a href="https://github.com/python/mypy"><img src="https://img.shields.io/badge/typing-mypy--strict-blue?style=flat-square" alt="mypy"></a>
  </p>
</p>

---

AgentFlow is a distributed platform for orchestrating multiple AI agents through complex, multi-step workflows. It combines a FastAPI backend with Celery-based distributed task execution, fault-tolerant design patterns (retry, circuit breaker, bulkhead, dead-letter queue), and full observability via OpenTelemetry and Prometheus. A React + TypeScript dashboard provides real-time visibility into workflow execution, agent status, and system health.

Built for production workloads where reliability, traceability, and horizontal scalability are non-negotiable.

## Demo

> Screenshots of the dashboard will be added here.

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="Dashboard Overview" width="800">
  <br>
  <em>Dashboard — real-time workflow metrics, agent throughput, and system health</em>
</p>

<p align="center">
  <img src="docs/screenshots/workflow-detail.png" alt="Workflow Detail" width="800">
  <br>
  <em>Workflow Detail — step-by-step execution timeline with status and duration</em>
</p>

## Key Features

- **Distributed Workflow Execution** — Celery + Redis task queue with sequential and parallel step execution
- **Specialized AI Agents** — Research, Analysis, Writer, and Code agents with extensible base class
- **Multi-Model LLM Support** — Any provider (OpenAI, Anthropic, etc.) via LiteLLM gateway
- **Fault Tolerance** — Retry with exponential backoff, circuit breaker, bulkhead isolation, dead-letter queue
- **Full Observability** — OpenTelemetry distributed tracing, Prometheus metrics, Grafana dashboards, structured logging
- **Real-Time Updates** — WebSocket push for workflow and step state changes
- **React Dashboard** — TypeScript frontend with workflow management, agent monitoring, and DLQ viewer
- **Production Docker Deployment** — Multi-stage builds, health checks, resource limits, OCI labels

## Architecture

```mermaid
graph TB
    subgraph Frontend
        UI[React + TypeScript Dashboard]
    end

    subgraph API Layer
        GW[FastAPI Gateway]
        WS[WebSocket Server]
        MW[Middleware<br/>Tracing · Logging · Metrics]
    end

    subgraph Orchestration
        WE[Workflow Engine]
        RE[Resilient Engine<br/>Retry · Circuit Breaker · Bulkhead]
        DLQ[Dead Letter Queue]
    end

    subgraph Workers
        CW[Celery Workers]
        Q1[Queue: workflows]
        Q2[Queue: agents]
        Q3[Queue: default]
    end

    subgraph Agents
        RA[Research Agent]
        AA[Analysis Agent]
        WA[Writer Agent]
        CA[Code Agent]
    end

    subgraph LLM Layer
        LS[LLM Service<br/>LiteLLM]
        OAI[OpenAI]
        ANT[Anthropic]
        OTHER[Other Providers]
    end

    subgraph Infrastructure
        PG[(PostgreSQL)]
        RD[(Redis)]
        PROM[Prometheus]
        GRAF[Grafana]
        OTEL[OpenTelemetry<br/>Collector]
    end

    UI -->|HTTP/REST| GW
    UI <-->|WebSocket| WS
    GW --> MW --> WE
    WE --> RE
    RE -->|Failed tasks| DLQ
    RE --> CW
    CW --> Q1 & Q2 & Q3
    CW --> RA & AA & WA & CA
    RA & AA & WA & CA --> LS
    LS --> OAI & ANT & OTHER

    GW --> PG
    CW --> PG
    CW --> RD
    GW -->|/metrics| PROM
    PROM --> GRAF
    GW --> OTEL
```

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Task Queue** | Celery 5.4, Redis 7 |
| **Database** | PostgreSQL 16, async SQLAlchemy 2.0, Alembic |
| **LLM Gateway** | LiteLLM (OpenAI, Anthropic, and more) |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| **Observability** | OpenTelemetry, Prometheus, Grafana, structlog |
| **Testing** | pytest, pytest-asyncio, pytest-cov |
| **Code Quality** | Ruff, mypy (strict), pre-commit |
| **Infrastructure** | Docker, Docker Compose |

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Node.js 18+ (for frontend development)

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/VardanMalik/agentflow.git
cd agentflow

# Copy environment variables
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY

# Start the full stack
docker compose up -d --build
```

Services will be available at:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Frontend Dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

### Local Development Setup

```bash
# Install Python dependencies
make install-dev

# Start infrastructure only
docker compose up -d postgres redis

# Run database migrations
make migrate

# Start the API server (hot reload)
make run

# In a separate terminal — start Celery worker
celery -A agentflow.core.celery_app worker --loglevel=info

# In a separate terminal — start the frontend
cd frontend && npm install && npm run dev
```

### Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://agentflow:agentflow@localhost:5432/agentflow` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `CELERY_BROKER_URL` | Celery broker URL | `redis://localhost:6379/1` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `DEFAULT_MODEL` | Default LLM model | `gpt-4o` |
| `DEBUG` | Enable debug mode and API docs | `false` |
| `SECRET_KEY` | Application secret key | — |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full environment variable reference.

## API Documentation

With `DEBUG=true`, interactive Swagger UI is available at [`/docs`](http://localhost:8000/docs) and ReDoc at [`/redoc`](http://localhost:8000/redoc).

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/ready` | Readiness check (DB + Redis) |
| `POST` | `/api/v1/workflows` | Create a workflow |
| `GET` | `/api/v1/workflows` | List workflows (paginated) |
| `POST` | `/api/v1/workflows/{id}/execute` | Execute a workflow |
| `GET` | `/api/v1/agents/types` | List available agent types |
| `GET` | `/api/v1/dashboard/stats` | Dashboard statistics |
| `WS` | `/ws/{workflow_id}` | Real-time workflow events |

See [docs/API.md](docs/API.md) for the full API reference with request/response examples.

## Project Structure

```
agentflow/
├── src/agentflow/
│   ├── main.py                    # FastAPI application entrypoint
│   ├── config.py                  # Settings via pydantic-settings
│   ├── api/
│   │   ├── router.py              # Route aggregation
│   │   ├── health.py              # Health and readiness endpoints
│   │   ├── workflows.py           # Workflow CRUD + execution
│   │   ├── agents.py              # Agent management
│   │   ├── dashboard.py           # Dashboard statistics
│   │   ├── websocket.py           # WebSocket server + event bus
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── middleware.py          # Logging, metrics, tracing middleware
│   ├── core/
│   │   ├── engine.py              # Workflow execution engine
│   │   ├── orchestrator.py        # High-level orchestration interface
│   │   ├── state.py               # Workflow/step state machine
│   │   ├── resilient_engine.py    # Engine with fault tolerance wrappers
│   │   ├── exceptions.py          # Exception hierarchy
│   │   ├── celery_app.py          # Celery configuration
│   │   ├── fault_tolerance/
│   │   │   ├── retry.py           # Retry with exponential backoff
│   │   │   ├── circuit_breaker.py # Circuit breaker state machine
│   │   │   ├── bulkhead.py        # Concurrency isolation
│   │   │   └── dead_letter_queue.py # Failed task storage
│   │   └── tasks/
│   │       └── workflow_tasks.py  # Celery task definitions
│   ├── agents/
│   │   ├── base.py                # BaseAgent abstract class
│   │   ├── factory.py             # Agent factory + registry
│   │   ├── research_agent.py      # Research agent
│   │   ├── analysis_agent.py      # Analysis agent
│   │   ├── writer_agent.py        # Writer agent
│   │   └── code_agent.py          # Code generation agent
│   ├── models/
│   │   ├── base.py                # SQLAlchemy base model
│   │   ├── workflow.py            # Workflow + WorkflowStep models
│   │   ├── agent.py               # Agent + AgentExecution models
│   │   └── task.py                # Task model
│   ├── services/
│   │   └── llm_service.py         # LiteLLM wrapper with retries
│   └── observability/
│       ├── logging_config.py      # structlog configuration
│       ├── tracing.py             # OpenTelemetry setup + @traced decorator
│       └── metrics.py             # Prometheus metrics collector
├── tests/
│   ├── conftest.py                # Fixtures (app, client, MockLLMService)
│   ├── test_health.py             # Health endpoint tests
│   ├── test_engine.py             # Workflow engine tests
│   ├── test_agents.py             # Agent execution tests
│   ├── test_fault_tolerance.py    # Retry, circuit breaker, bulkhead, DLQ tests
│   └── test_observability.py      # Tracing and metrics tests
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main app with sidebar navigation
│   │   ├── api/client.ts          # Axios API client + TypeScript types
│   │   └── components/            # Dashboard, WorkflowList, AgentPanel, DLQPanel
│   ├── package.json
│   └── vite.config.ts
├── docker/
│   ├── Dockerfile                 # Multi-stage API image
│   ├── Dockerfile.worker          # Celery worker image
│   ├── prometheus.yml             # Prometheus scrape config
│   └── grafana/provisioning/      # Grafana datasources + dashboards
├── docker-compose.yml             # Full stack (7 services)
├── pyproject.toml                 # Project metadata + tool config
├── Makefile                       # Development commands
├── .env.example                   # Environment variable template
└── docs/
    ├── ARCHITECTURE.md            # System architecture
    ├── API.md                     # API reference
    ├── DEVELOPMENT.md             # Developer guide
    └── DEPLOYMENT.md              # Deployment guide
```

## Testing

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_engine.py -v

# Run with coverage report
pytest --cov=agentflow --cov-report=html
```

The test suite includes 119 tests covering:

- Workflow engine state machine and execution
- All four agent types (research, analysis, writer, code)
- Fault tolerance patterns (retry, circuit breaker, bulkhead, DLQ)
- Observability (tracing, metrics collection)
- Health check endpoints
- API schema validation

## Deployment

### Docker Compose (Production)

```bash
# Set production environment variables
export ENVIRONMENT=production
export DEBUG=false
export SECRET_KEY=$(openssl rand -hex 32)

# Deploy
docker compose up -d --build
```

All services include health checks, resource limits, and restart policies. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full production deployment guide.

### Kubernetes

AgentFlow is designed for horizontal scaling. Key considerations:

- API and worker pods scale independently
- Workers are stateless — scale based on queue depth
- PostgreSQL and Redis should use managed services (RDS, ElastiCache) in production
- Use Kubernetes HPA for auto-scaling based on Prometheus metrics

### AWS ECS

Deploy using ECS services with Fargate or EC2 launch types. Each component (API, worker, frontend) maps to a separate ECS service with its own task definition and scaling policy.

## Observability

AgentFlow provides three observability pillars:

**Tracing** — OpenTelemetry distributed traces across API requests, workflow execution, and agent calls. Every workflow step gets its own span with `workflow_id` and `step_id` attributes.

**Metrics** — Prometheus metrics exposed at `/metrics`:
- `workflows_total`, `workflow_duration_seconds`, `workflows_active`
- `agent_executions_total`, `agent_execution_duration_seconds`, `agent_tokens_total`
- `circuit_breaker_state`, `retry_attempts_total`, `dlq_entries_total`
- `request_duration_seconds`, `requests_total`

**Logging** — Structured JSON logs via structlog with OpenTelemetry trace correlation.

```bash
# Example Prometheus queries
rate(workflows_total{status="completed"}[5m])           # Workflow completion rate
histogram_quantile(0.95, workflow_duration_seconds)       # P95 workflow duration
sum(agent_tokens_total) by (agent_type)                   # Token usage by agent
```

Grafana dashboards are auto-provisioned at http://localhost:3001 when running with Docker Compose.

> Grafana dashboard screenshots will be added here.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to AgentFlow.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Vardan Malik**

- GitHub: [github.com/VardanMalik](https://github.com/VardanMalik)
- LinkedIn: [linkedin.com/in/vardanmalik](https://linkedin.com/in/vardanmalik)
