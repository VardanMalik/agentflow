"""Shared test fixtures."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from agentflow.agents.analysis_agent import AnalysisAgent
from agentflow.agents.base import AgentContext
from agentflow.agents.code_agent import CodeAgent
from agentflow.agents.research_agent import ResearchAgent
from agentflow.agents.writer_agent import WriterAgent
from agentflow.main import create_app
from agentflow.services.llm_service import LLMResponse


# ---------------------------------------------------------------------------
# HTTP client fixtures (existing)
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a test application instance."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# MockLLMService
# ---------------------------------------------------------------------------


class MockLLMService:
    """Minimal test double for LLMService that returns preset LLMResponse objects.

    Records every call made to ``completion`` for assertion in tests.
    """

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _success_llm(content: str) -> MockLLMService:
    """Return a MockLLMService that yields a successful response with *content*."""
    return MockLLMService(
        LLMResponse(
            content=content,
            model="mock-model",
            input_tokens=50,
            output_tokens=100,
            latency_ms=200,
        )
    )


def _error_llm(error: str = "API Error: rate limit exceeded") -> MockLLMService:
    """Return a MockLLMService that always yields an error response."""
    return MockLLMService(
        LLMResponse(
            content="",
            model="mock-model",
            input_tokens=0,
            output_tokens=0,
            latency_ms=50,
            error=error,
        )
    )


# ---------------------------------------------------------------------------
# LLM service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_service() -> MockLLMService:
    """Generic mock LLM service that returns an empty-JSON success response."""
    return _success_llm("{}")


@pytest.fixture
def error_llm_service() -> MockLLMService:
    """Mock LLM service that always returns an error response."""
    return _error_llm()


@pytest.fixture
def make_mock_llm():
    """Factory fixture: call with an LLMResponse to get a configured MockLLMService."""

    def _factory(response: LLMResponse) -> MockLLMService:
        return MockLLMService(response)

    return _factory


# ---------------------------------------------------------------------------
# AgentContext factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_context():
    """Factory fixture for building AgentContext instances with arbitrary inputs."""

    def _factory(
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        return AgentContext(
            run_id=uuid4(),
            step_id=uuid4(),
            inputs=inputs or {},
            metadata=metadata or {},
        )

    return _factory


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def research_agent() -> ResearchAgent:
    """ResearchAgent backed by a mock LLM returning valid structured JSON."""
    llm = _success_llm(
        json.dumps(
            {
                "findings": "Python is a versatile programming language used across many domains.",
                "key_points": ["Easy to learn", "Large ecosystem", "Cross-platform"],
                "sources": ["Python documentation", "Stack Overflow Developer Survey"],
            }
        )
    )
    return ResearchAgent(llm_service=llm)


@pytest.fixture
def analysis_agent() -> AnalysisAgent:
    """AnalysisAgent backed by a mock LLM returning valid structured JSON."""
    llm = _success_llm(
        json.dumps(
            {
                "analysis": "Sales data shows consistent upward trends across all quarters.",
                "patterns": ["Growth in Q3", "Seasonal dip in Q1"],
                "recommendations": ["Increase Q3 inventory", "Plan Q1 promotions"],
                "confidence": 0.87,
            }
        )
    )
    return AnalysisAgent(llm_service=llm)


@pytest.fixture
def writer_agent() -> WriterAgent:
    """WriterAgent backed by a mock LLM returning valid structured JSON."""
    llm = _success_llm(
        json.dumps(
            {
                "content": "# Introduction\n\nThis article explores modern technology trends.",
                "format": "article",
            }
        )
    )
    return WriterAgent(llm_service=llm)


@pytest.fixture
def code_agent() -> CodeAgent:
    """CodeAgent backed by a mock LLM returning valid structured JSON."""
    llm = _success_llm(
        json.dumps(
            {
                "code": "def add(a: int, b: int) -> int:\n    return a + b",
                "language": "python",
                "explanation": "Adds two integers and returns the result.",
                "dependencies": [],
            }
        )
    )
    return CodeAgent(llm_service=llm)
