"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "AgentFlow"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://agentflow:agentflow@localhost:5432/agentflow",
    )
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_default_queue: str = "default"
    celery_task_default_retry_delay: int = 60
    celery_task_max_retries: int = 3
    celery_task_soft_time_limit: int = 300
    celery_task_time_limit: int = 600
    celery_result_expires: int = 86400
    celery_worker_prefetch_multiplier: int = 1

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_model: str = "gpt-4o"

    # Observability
    otel_service_name: str = "agentflow"
    otel_exporter_endpoint: str = "http://localhost:4317"

    # Security
    secret_key: str = "change-me-in-production"
    api_key_header: str = "X-API-Key"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
