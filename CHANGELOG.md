# Changelog

All notable changes to AgentFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-08

Initial release of AgentFlow.

### Added

#### Core Platform
- FastAPI application with async request handling and lifespan management
- Workflow engine with state machine (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- Sequential and parallel step execution within workflows
- Orchestrator interface for high-level workflow management (submit, execute, cancel, retry)
- Celery distributed task queue with three dedicated queues (default, workflows, agents)

#### AI Agents
- Base agent abstract class with standardized input/output contract
- Research agent — topic investigation with structured findings
- Analysis agent — data analysis with pattern detection and recommendations
- Writer agent — content generation with configurable tone and format
- Code agent — code generation with language and style options
- Agent factory with registry pattern for extensible agent registration
- LLM service with LiteLLM integration supporting OpenAI, Anthropic, and other providers

#### Fault Tolerance
- Retry pattern with exponential backoff and jitter
- Circuit breaker with CLOSED/OPEN/HALF_OPEN state machine
- Bulkhead pattern with semaphore-based concurrency isolation
- Dead letter queue for failed tasks with manual retry capability
- Resilient workflow engine composing all four patterns

#### Observability
- OpenTelemetry distributed tracing with OTLP gRPC export
- `@traced` decorator for automatic span creation
- Prometheus metrics: workflow counters, agent histograms, fault tolerance gauges, HTTP metrics
- Structured logging via structlog with OpenTelemetry trace correlation
- Grafana dashboard auto-provisioning with Prometheus datasource

#### API
- Workflow CRUD endpoints with pagination and status filtering
- Workflow execution, cancellation, and retry endpoints
- Agent management and testing endpoints
- Dashboard statistics endpoint
- Health and readiness checks (database + Redis)
- WebSocket server for real-time workflow and step events
- Request tracing, logging, and metrics middleware

#### Frontend
- React 18 + TypeScript dashboard with Vite and Tailwind CSS
- Dashboard overview with workflow metrics and charts (Recharts)
- Workflow list with pagination and status filters
- Workflow detail view with step timeline
- Agent panel with execution statistics
- Dead letter queue viewer with retry and purge actions
- Settings panel
- Real-time health status polling

#### Infrastructure
- Multi-stage Docker builds for API and Celery worker
- Docker Compose stack: API, worker, frontend, PostgreSQL, Redis, Prometheus, Grafana
- Health checks on all services
- OCI image labels
- Resource limits and reservations for all containers

#### Database
- SQLAlchemy 2.0 async models with PostgreSQL + asyncpg
- Workflow, WorkflowStep, Agent, AgentExecution, and Task models
- Alembic migration support

#### CI/CD
- GitHub Actions CI pipeline: lint, test, type-check, Docker build
- Release pipeline with GHCR publishing
- Dependabot configuration for automated dependency updates

#### Developer Experience
- Makefile with common development commands
- Pre-commit hooks for Ruff linting and formatting
- pytest test suite with 119 tests and coverage reporting
- Environment variable template (`.env.example`)

[0.1.0]: https://github.com/VardanMalik/agentflow/releases/tag/v0.1.0
