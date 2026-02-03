# AgentFlow

Distributed AI Agent Orchestration Platform — coordinate multiple AI agents to complete complex workflows with fault tolerance, observability, and scalability.

## Features

- **Workflow Engine** — define multi-step workflows with sequential and parallel execution
- **Agent Orchestration** — register, manage, and coordinate AI agents across tasks
- **Multi-Model Support** — use any LLM provider (OpenAI, Anthropic, etc.) via LiteLLM
- **Distributed Task Queue** — Celery + Redis for reliable, scalable task execution
- **Async-First** — built on FastAPI and async SQLAlchemy for high throughput
- **Observability** — structured logging with structlog and distributed tracing via OpenTelemetry
- **Type-Safe Configuration** — pydantic-settings with environment variable support

## Tech Stack

| Layer           | Technology                     |
|-----------------|--------------------------------|
| API Framework   | FastAPI                        |
| Task Queue      | Celery + Redis                 |
| Database        | PostgreSQL + SQLAlchemy (async)|
| LLM Gateway     | LiteLLM                        |
| Observability   | OpenTelemetry + structlog      |
| Validation      | Pydantic v2                    |
| Testing         | pytest + pytest-asyncio        |
| Linting         | Ruff + mypy                    |
| Containerization| Docker + Docker Compose        |

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/agentflow.git
cd agentflow

# Copy environment variables
cp .env.example .env

# Install in development mode
make install-dev

# Start infrastructure (PostgreSQL, Redis)
make docker-up

# Run database migrations
make migrate

# Start the development server
make run
```

The API will be available at `http://localhost:8000`. With `DEBUG=true`, interactive docs are at `/docs`.

### Running Tests

```bash
make test
```

### Linting & Formatting

```bash
make lint       # Check for issues
make format     # Auto-fix and format
make typecheck  # Run mypy
```

## Project Structure

```
agentflow/
├── src/agentflow/
│   ├── __init__.py          # Package version
│   ├── main.py              # FastAPI application entrypoint
│   ├── config.py            # Settings via pydantic-settings
│   ├── api/                 # HTTP route handlers
│   │   ├── routes.py        # Router aggregation
│   │   ├── health.py        # Health check endpoint
│   │   ├── workflows.py     # Workflow CRUD endpoints
│   │   └── agents.py        # Agent management endpoints
│   ├── core/                # Workflow engine & orchestration
│   │   ├── orchestrator.py  # Workflow lifecycle management
│   │   └── engine.py        # Step execution engine
│   ├── agents/              # AI agent definitions
│   │   └── base.py          # Abstract base agent class
│   ├── models/              # SQLAlchemy database models
│   │   └── base.py          # Base model & session factory
│   ├── services/            # Business logic layer
│   └── utils/               # Helpers & shared utilities
│       └── logging.py       # Structured logging setup
├── tests/                   # Test suite
│   ├── conftest.py          # Shared fixtures
│   ├── test_health.py       # Health endpoint tests
│   └── test_engine.py       # Workflow engine tests
├── docker/
│   ├── Dockerfile           # Multi-stage production image
│   └── docker-compose.yml   # Local development stack
├── docs/                    # Documentation
├── .env.example             # Environment variable template
├── pyproject.toml           # Project metadata & dependencies
├── Makefile                 # Common development commands
└── LICENSE
```

## License

MIT
