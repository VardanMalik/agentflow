"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass
class AgentContext:
    """Execution context passed to an agent."""

    run_id: UUID
    step_id: UUID
    inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result returned by an agent after execution."""

    output: Any = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all AI agents.

    Subclass this to create agents with specific capabilities.
    Each agent receives a context with inputs and must return a result.
    """

    def __init__(self, agent_id: UUID | None = None, name: str = "") -> None:
        self.agent_id = agent_id or uuid4()
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task.

        Args:
            context: The execution context with inputs and metadata.

        Returns:
            The result of the agent's execution.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.name!r})"
