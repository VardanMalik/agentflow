# Development Guide

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Docker and Docker Compose
- PostgreSQL 16+ and Redis 7+ (or use Docker for these)

## Setting Up Local Development

### 1. Clone and install

```bash
git clone https://github.com/VardanMalik/agentflow.git
cd agentflow

# Install Python package with dev dependencies
make install-dev
```

This installs the package in editable mode, dev dependencies (pytest, ruff, mypy, pre-commit), and sets up pre-commit hooks.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` — Redis connection string
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — at least one LLM provider key
- `SECRET_KEY` — any random string for development

### 3. Start infrastructure

```bash
# Start PostgreSQL and Redis via Docker
docker compose up -d postgres redis
```

### 4. Run database migrations

```bash
make migrate
```

### 5. Start the API server

```bash
make run
# → uvicorn agentflow.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now running at http://localhost:8000 with hot reload. Interactive docs at http://localhost:8000/docs (requires `DEBUG=true`).

### 6. Start a Celery worker

In a separate terminal:

```bash
celery -A agentflow.core.celery_app worker --loglevel=info --concurrency=4
```

### 7. Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Running Tests

```bash
# Run all tests with coverage
make test

# Run a specific test file
pytest tests/test_engine.py -v

# Run a specific test
pytest tests/test_engine.py::test_execute_sequential_workflow -v

# Generate HTML coverage report
pytest --cov=agentflow --cov-report=html
open htmlcov/index.html
```

### Test structure

| File | Tests |
|---|---|
| `test_health.py` | Health and readiness endpoints |
| `test_engine.py` | Workflow engine state machine, sequential/parallel execution, retry |
| `test_agents.py` | All four agent types with mocked LLM |
| `test_fault_tolerance.py` | Retry, circuit breaker, bulkhead, dead letter queue |
| `test_observability.py` | Tracing setup, metrics collection |

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` — async test functions are detected automatically.

The `conftest.py` provides shared fixtures:

- `app` — Test FastAPI application
- `client` — `httpx.AsyncClient` configured for ASGI transport
- `MockLLMService` — Configurable test double that returns deterministic responses

---

## Code Style

AgentFlow uses **Ruff** for linting and formatting, and **mypy** in strict mode for type checking.

```bash
# Check for lint issues
make lint

# Auto-fix and format
make format

# Run type checker
make typecheck
```

### Ruff configuration

From `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "TCH"]
```

Enabled rules: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, flake8-simplify, flake8-type-checking.

### mypy configuration

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
plugins = ["pydantic.mypy"]
```

Pre-commit hooks run both linters automatically on each commit.

---

## Adding a New Agent

### Step 1: Create the agent class

Create `src/agentflow/agents/your_agent.py`:

```python
from agentflow.agents.base import AgentContext, AgentResult, BaseAgent
from agentflow.services.llm_service import LLMService


class YourAgent(BaseAgent):
    """Description of what this agent does."""

    def __init__(self, llm_service: LLMService) -> None:
        super().__init__(name="your-agent")
        self.llm_service = llm_service

    async def execute(self, context: AgentContext) -> AgentResult:
        # 1. Extract inputs
        your_input = context.inputs.get("your_field", "default")

        # 2. Build the prompt
        system_prompt = "You are a specialist in..."
        user_prompt = f"Do something with: {your_input}"

        # 3. Call the LLM
        response = await self.llm_service.completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1024,
        )

        if not response.success:
            return AgentResult(error=response.error)

        # 4. Parse and return results
        return AgentResult(
            output={"result": response.content},
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
```

### Step 2: Register in the factory

Edit `src/agentflow/agents/factory.py`:

```python
from agentflow.agents.your_agent import YourAgent

class AgentFactory:
    def _register_defaults(self) -> None:
        # ... existing agents ...
        self.register("your_type", YourAgent(self.llm_service))
```

### Step 3: Add tests

Create `tests/test_your_agent.py`:

```python
import pytest
from agentflow.agents.your_agent import YourAgent


@pytest.mark.asyncio
async def test_your_agent_executes(mock_llm_service):
    agent = YourAgent(mock_llm_service)
    context = AgentContext(
        run_id=uuid4(),
        step_id=uuid4(),
        inputs={"your_field": "test value"},
    )
    result = await agent.execute(context)
    assert result.output is not None
    assert result.error is None
```

---

## Adding a New API Endpoint

### Step 1: Define schemas

Add request/response models to `src/agentflow/api/schemas.py`:

```python
class YourRequest(BaseModel):
    field: str

class YourResponse(BaseModel):
    result: str
```

### Step 2: Create the route handler

Create `src/agentflow/api/your_routes.py`:

```python
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/your-resource", tags=["your-resource"])

@router.get("/")
async def list_resources() -> dict:
    return {"items": []}

@router.post("/", status_code=201)
async def create_resource(request: YourRequest) -> YourResponse:
    return YourResponse(result="created")
```

### Step 3: Register the router

Edit `src/agentflow/api/router.py`:

```python
from agentflow.api.your_routes import router as your_router

api_router.include_router(your_router)
```

### Step 4: Add tests

```python
@pytest.mark.asyncio
async def test_list_resources(client):
    response = await client.get("/api/v1/your-resource/")
    assert response.status_code == 200
```

---

## Database Migrations

AgentFlow uses **Alembic** for database migrations with async SQLAlchemy.

```bash
# Apply all pending migrations
make migrate

# Create a new migration after changing models
alembic revision --autogenerate -m "add your_table"

# Downgrade one revision
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

When adding a new model:

1. Create the model in `src/agentflow/models/`
2. Import it in `src/agentflow/models/__init__.py` (so Alembic detects it)
3. Run `alembic revision --autogenerate -m "description"`
4. Review the generated migration in `alembic/versions/`
5. Apply with `make migrate`

---

## Debugging Tips

### API debugging

- Set `DEBUG=true` and `LOG_LEVEL=DEBUG` in `.env`
- Use the Swagger UI at `/docs` to test endpoints interactively
- Check structured logs — every request logs method, path, status, and duration

### Workflow debugging

- Use `GET /api/v1/workflows/{id}` to inspect step-by-step results and errors
- Check the DLQ via `GET /api/v1/dashboard/stats` or the DLQ panel in the dashboard
- Workflow state transitions are logged with `workflow_id` for filtering

### Celery debugging

```bash
# Monitor tasks in real time
celery -A agentflow.core.celery_app events

# Inspect active tasks
celery -A agentflow.core.celery_app inspect active

# Inspect reserved (queued) tasks
celery -A agentflow.core.celery_app inspect reserved

# Check worker stats
celery -A agentflow.core.celery_app inspect stats
```

### Database debugging

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U agentflow -d agentflow

# Check workflow status
SELECT id, name, status, created_at FROM workflows ORDER BY created_at DESC LIMIT 10;
```

### Observability debugging

- Prometheus metrics at http://localhost:9090 — query `workflows_total` or `agent_executions_total`
- Grafana dashboards at http://localhost:3001 — pre-provisioned with AgentFlow dashboard
- Trace correlation: search for `trace_id` values across logs and traces
