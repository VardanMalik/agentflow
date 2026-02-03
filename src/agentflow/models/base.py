"""SQLAlchemy base model and database session management."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry

from agentflow.config import get_settings

mapper_registry = registry()


class Base(DeclarativeBase):
    """Base model with common columns for all tables."""

    registry = mapper_registry

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id}>"


def get_engine():
    """Create the async database engine."""
    settings = get_settings()
    return create_async_engine(
        str(settings.database_url),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.debug,
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create an async session factory."""
    engine = get_engine()
    return async_sessionmaker(engine, expire_on_commit=False)
