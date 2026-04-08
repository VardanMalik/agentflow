# Deployment Guide

## Docker Compose (Production)

### Prerequisites

- Docker Engine 24+
- Docker Compose v2
- At least 4 GB RAM available for containers

### Deployment

```bash
# Clone the repository
git clone https://github.com/VardanMalik/agentflow.git
cd agentflow

# Create production environment file
cp .env.example .env
```

Edit `.env` with production values:

```bash
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgresql+asyncpg://agentflow:STRONG_PASSWORD@postgres:5432/agentflow
REDIS_URL=redis://redis:6379/0
OPENAI_API_KEY=sk-...
```

Deploy:

```bash
docker compose up -d --build
```

Verify all services are healthy:

```bash
docker compose ps
docker compose exec api curl -s http://localhost:8000/api/v1/health/ready
```

### Services

| Service | Port | Health Check |
|---|---|---|
| API (FastAPI) | 8000 | `GET /api/v1/health` |
| Celery Worker | — | `celery inspect ping` |
| Frontend (React) | 3000 | `GET /` |
| PostgreSQL | 5432 | `pg_isready` |
| Redis | 6379 | `redis-cli ping` |
| Prometheus | 9090 | `GET /-/healthy` |
| Grafana | 3001 | `GET /api/health` |

### Resource Limits

Default resource limits in `docker-compose.yml`:

| Service | Memory Limit | CPU Limit | Memory Reserve | CPU Reserve |
|---|---|---|---|---|
| API | 512 MB | 1.0 | 256 MB | 0.25 |
| Worker | 1 GB | 2.0 | 512 MB | 0.50 |
| Frontend | 128 MB | 0.25 | 64 MB | 0.05 |
| PostgreSQL | 512 MB | 1.0 | 256 MB | 0.25 |
| Redis | 384 MB | 0.5 | 128 MB | 0.10 |

Adjust based on workload. Workers should be allocated more resources for LLM-heavy workflows.

---

## Environment Variables Reference

### Application

| Variable | Description | Default | Required |
|---|---|---|---|
| `ENVIRONMENT` | Runtime environment | `development` | No |
| `DEBUG` | Enable debug mode and API docs | `false` | No |
| `LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` | No |
| `SECRET_KEY` | Application secret key | — | **Yes** |

### Server

| Variable | Description | Default | Required |
|---|---|---|---|
| `HOST` | Bind host | `0.0.0.0` | No |
| `PORT` | Bind port | `8000` | No |

### Database

| Variable | Description | Default | Required |
|---|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | — | **Yes** |
| `DATABASE_POOL_SIZE` | Connection pool size | `20` | No |
| `DATABASE_MAX_OVERFLOW` | Max overflow connections | `10` | No |

### Redis

| Variable | Description | Default | Required |
|---|---|---|---|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | No |

### Celery

| Variable | Description | Default | Required |
|---|---|---|---|
| `CELERY_BROKER_URL` | Celery broker URL | `redis://localhost:6379/1` | No |
| `CELERY_RESULT_BACKEND` | Celery result backend | `redis://localhost:6379/2` | No |
| `CELERY_TASK_DEFAULT_RETRY_DELAY` | Default retry delay (seconds) | `60` | No |
| `CELERY_TASK_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `CELERY_TASK_SOFT_TIME_LIMIT` | Soft time limit (seconds) | `300` | No |
| `CELERY_TASK_TIME_LIMIT` | Hard time limit (seconds) | `600` | No |
| `CELERY_RESULT_EXPIRES` | Result expiry (seconds) | `86400` | No |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | Worker prefetch count | `1` | No |

### LLM Providers

| Variable | Description | Default | Required |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — | Conditional |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | Conditional |
| `DEFAULT_MODEL` | Default LLM model | `gpt-4o` | No |

At least one LLM provider API key is required for agent execution.

### Observability

| Variable | Description | Default | Required |
|---|---|---|---|
| `OTEL_SERVICE_NAME` | OpenTelemetry service name | `agentflow` | No |
| `OTEL_EXPORTER_ENDPOINT` | OTLP exporter endpoint | `http://localhost:4317` | No |

### Security

| Variable | Description | Default | Required |
|---|---|---|---|
| `API_KEY_HEADER` | API key header name | `X-API-Key` | No |

---

## Scaling Considerations

### Horizontal Scaling

**API servers**: Stateless — run multiple instances behind a load balancer (Nginx, ALB, Traefik). Each instance connects to the same PostgreSQL and Redis.

```bash
docker compose up -d --scale api=3
```

**Celery workers**: Stateless — scale based on queue depth. Each worker connects to the shared Redis broker.

```bash
docker compose up -d --scale worker=5
```

**Frontend**: Static assets can be served from a CDN. The Vite build produces a static bundle.

### Queue Isolation

Celery uses three dedicated queues for priority isolation:

| Queue | Purpose |
|---|---|
| `default` | General tasks |
| `workflows` | Workflow orchestration tasks |
| `agents` | Agent execution tasks |

For dedicated scaling, run workers that consume specific queues:

```bash
# Worker for workflow tasks only
celery -A agentflow.core.celery_app worker -Q workflows --concurrency=2

# Worker for agent tasks only (higher concurrency for I/O-bound LLM calls)
celery -A agentflow.core.celery_app worker -Q agents --concurrency=8
```

### Database Scaling

- **Connection pooling**: Default pool size is 20 with max overflow of 10. Increase for higher API concurrency.
- **Read replicas**: Route read queries (workflow listing, dashboard stats) to replicas.
- **Managed services**: Use Amazon RDS or Google Cloud SQL in production for automated backups, failover, and scaling.

### Redis Scaling

- **Separate instances**: Use separate Redis instances for Celery broker (DB 1), result backend (DB 2), and caching (DB 0) in high-throughput deployments.
- **Redis Cluster**: For high availability, deploy Redis in cluster mode.
- **Memory policy**: Default configuration uses `allkeys-lru` eviction with 256 MB max memory.

---

## Monitoring Setup

### Prometheus

Prometheus is pre-configured to scrape the API at `/metrics` every 15 seconds with 15-day retention.

Key metrics to monitor:

```promql
# Workflow throughput
rate(workflows_total[5m])

# Workflow success rate
rate(workflows_total{status="completed"}[5m]) / rate(workflows_total[5m])

# P95 workflow duration
histogram_quantile(0.95, rate(workflow_duration_seconds_bucket[5m]))

# Active workflows
workflows_active

# Agent token usage
sum(rate(agent_tokens_total[5m])) by (agent_type)

# Circuit breaker state (0=closed, 1=open, 2=half_open)
circuit_breaker_state

# DLQ size
dlq_entries_total

# API latency P99
histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m]))

# Error rate
rate(requests_total{status_code=~"5.."}[5m])
```

### Grafana

Grafana is auto-provisioned with a Prometheus datasource at http://localhost:3001 (default credentials: `admin` / `admin`).

Pre-configured dashboards include:
- Workflow overview (throughput, duration, success rate)
- Agent performance (executions, token usage, latency)
- Fault tolerance state (circuit breaker, bulkhead, DLQ)
- HTTP request metrics (latency, error rate, throughput)

### Alerting

Recommended Prometheus alerting rules (add to your alertmanager config):

```yaml
groups:
  - name: agentflow
    rules:
      - alert: HighWorkflowFailureRate
        expr: rate(workflows_total{status="failed"}[5m]) / rate(workflows_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "Workflow failure rate above 10%"

      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        annotations:
          summary: "Circuit breaker is open — agent calls are being rejected"

      - alert: DLQGrowing
        expr: dlq_entries_total > 50
        for: 10m
        annotations:
          summary: "Dead letter queue has more than 50 entries"

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) > 2
        for: 5m
        annotations:
          summary: "API P95 latency above 2 seconds"
```

---

## Backup and Recovery

### PostgreSQL

```bash
# Create a backup
docker compose exec postgres pg_dump -U agentflow agentflow > backup_$(date +%Y%m%d).sql

# Restore from backup
docker compose exec -T postgres psql -U agentflow agentflow < backup_20260408.sql
```

For production, use automated backup solutions:
- Amazon RDS: Automated snapshots with point-in-time recovery
- Google Cloud SQL: Automated backups with configurable retention
- Self-managed: `pg_basebackup` with WAL archiving for continuous backup

### Redis

Redis data is persisted to the `redisdata` Docker volume. For production:
- Enable AOF persistence (`appendonly yes`) for durability
- Use Redis Sentinel or Redis Cluster for high availability
- Celery task results are ephemeral (default TTL: 24 hours) — Redis data loss affects result retrieval but not task execution

### Application State

- **Workflow state**: Persisted in PostgreSQL. No application-level state to back up.
- **DLQ entries**: Currently in-memory. Entries are lost on API restart. For production persistence, consider migrating DLQ storage to PostgreSQL.
- **Prometheus data**: Persisted to the `prometheusdata` Docker volume. 15-day retention by default.
- **Grafana dashboards**: Provisioned from config files in `docker/grafana/provisioning/`. No backup needed — dashboards are version-controlled.

---

## Kubernetes Deployment

AgentFlow is designed for Kubernetes but does not include manifests in this release. Key considerations for a Kubernetes deployment:

- **API**: Deployment with 2+ replicas, HPA based on CPU/request rate, readiness probe on `/api/v1/health/ready`
- **Workers**: Deployment with HPA based on custom Prometheus metric (queue depth), no service needed
- **Frontend**: Deployment or static file serving via Nginx/CDN
- **PostgreSQL**: Use a managed service (RDS, Cloud SQL) or a StatefulSet with PersistentVolumeClaims
- **Redis**: Use a managed service (ElastiCache, Memorystore) or deploy with Redis Operator
- **Prometheus/Grafana**: Use the kube-prometheus-stack Helm chart

### AWS ECS

Each component maps to a separate ECS service:

| Component | Launch Type | Scaling |
|---|---|---|
| API | Fargate | Target tracking on CPU/request count |
| Worker | Fargate | Step scaling on SQS queue depth (if using SQS) or custom metric |
| Frontend | Fargate or S3 + CloudFront | Fixed or CDN |

Use AWS Secrets Manager for API keys and database credentials. Configure ECS task definitions with the environment variables listed above.
