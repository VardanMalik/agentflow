"""Factory for creating and registering built-in agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agentflow.agents.analysis_agent import AnalysisAgent
from agentflow.agents.base import BaseAgent
from agentflow.agents.code_agent import CodeAgent
from agentflow.agents.research_agent import ResearchAgent
from agentflow.agents.writer_agent import WriterAgent

if TYPE_CHECKING:
    from agentflow.core.engine import AgentRegistry
    from agentflow.services.llm_service import LLMService

logger = structlog.get_logger(__name__)

# Mapping from string type keys to agent classes.
_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "research": ResearchAgent,
    "analysis": AnalysisAgent,
    "writer": WriterAgent,
    "code": CodeAgent,
}


class AgentFactory:
    """Factory that instantiates and registers built-in agents.

    Usage::

        factory = AgentFactory(llm_service)
        factory.register_agents(registry)

        # Or create a single agent on demand:
        agent = AgentFactory.create_agent("research", llm_service)

    The factory holds a default :class:`~agentflow.services.llm_service.LLMService`
    used by :meth:`register_agents`. Individual agents can be created with a
    different service via the static :meth:`create_agent` method.
    """

    def __init__(self, llm_service: LLMService) -> None:
        """Initialise the factory with a default LLM service.

        Args:
            llm_service: The :class:`~agentflow.services.llm_service.LLMService`
                instance used when :meth:`register_agents` is called.
        """
        self._llm = llm_service
        self._log = logger.bind(factory="AgentFactory")

    @staticmethod
    def create_agent(agent_type: str, llm_service: LLMService) -> BaseAgent:
        """Instantiate a single agent by type string.

        Args:
            agent_type: One of ``"research"``, ``"analysis"``, ``"writer"``,
                or ``"code"``.
            llm_service: The LLM service to inject into the agent.

        Returns:
            A new :class:`~agentflow.agents.base.BaseAgent` instance.

        Raises:
            ValueError: If *agent_type* is not a recognised built-in type.
        """
        agent_cls = _AGENT_CLASSES.get(agent_type)
        if agent_cls is None:
            available = ", ".join(sorted(_AGENT_CLASSES))
            raise ValueError(
                f"Unknown agent type {agent_type!r}. "
                f"Available types: {available}"
            )
        return agent_cls(llm_service=llm_service)

    def register_agents(self, registry: AgentRegistry) -> None:
        """Create all built-in agents and register them in *registry*.

        Uses the :class:`~agentflow.services.llm_service.LLMService` provided
        at construction time.

        Args:
            registry: The :class:`~agentflow.core.engine.AgentRegistry` to
                populate.
        """
        for agent_type in _AGENT_CLASSES:
            agent = self.create_agent(agent_type, self._llm)
            registry.register(agent_type, agent)
            self._log.debug("agent_registered", agent_type=agent_type, agent=repr(agent))

        self._log.info(
            "agents_registered",
            types=list(_AGENT_CLASSES.keys()),
            total=len(_AGENT_CLASSES),
        )
