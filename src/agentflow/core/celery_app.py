"""Celery application configuration."""

from celery import Celery

from agentflow.config import get_settings


def create_celery_app() -> Celery:
    """Create and configure the Celery application."""
    settings = get_settings()

    app = Celery("agentflow")

    app.conf.update(
        # Broker & backend
        broker_url=settings.celery_broker_url,
        result_backend=settings.celery_result_backend,

        # Serialization
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        event_serializer="json",

        # Reliability — process tasks at-least-once
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,

        # Results
        result_expires=settings.celery_result_expires,

        # Time limits
        task_soft_time_limit=settings.celery_task_soft_time_limit,
        task_time_limit=settings.celery_task_time_limit,

        # Retry defaults
        task_default_retry_delay=settings.celery_task_default_retry_delay,

        # Queue routing
        task_default_queue=settings.celery_task_default_queue,
        task_routes={
            "agentflow.core.tasks.workflow_tasks.*": {"queue": "workflows"},
            "agentflow.core.tasks.agent_tasks.*": {"queue": "agents"},
        },
        task_queues={
            "default": {"exchange": "default", "routing_key": "default"},
            "workflows": {"exchange": "workflows", "routing_key": "workflows"},
            "agents": {"exchange": "agents", "routing_key": "agents"},
        },

        # Discovery
        task_packages=["agentflow.core.tasks"],

        # Misc
        timezone="UTC",
        enable_utc=True,
        worker_hijack_root_logger=False,
    )

    app.autodiscover_tasks(["agentflow.core.tasks"])

    return app


celery_app = create_celery_app()
