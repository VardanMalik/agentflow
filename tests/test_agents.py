"""Comprehensive tests for the agent system."""

from __future__ import annotations

import json

import pytest

from agentflow.agents.analysis_agent import AnalysisAgent
from agentflow.agents.code_agent import CodeAgent
from agentflow.agents.factory import AgentFactory
from agentflow.agents.research_agent import ResearchAgent
from agentflow.agents.writer_agent import WriterAgent
from agentflow.core.engine import AgentRegistry
from agentflow.services.llm_service import LLMResponse


# ===========================================================================
# ResearchAgent
# ===========================================================================


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_valid_topic_returns_structured_output(self, research_agent, make_context):
        context = make_context(inputs={"topic": "Python programming language"})
        result = await research_agent.execute(context)

        assert result.error is None
        assert result.output is not None
        assert "findings" in result.output
        assert "key_points" in result.output
        assert "sources" in result.output
        assert isinstance(result.output["findings"], str)
        assert isinstance(result.output["key_points"], list)
        assert isinstance(result.output["sources"], list)

    @pytest.mark.asyncio
    async def test_missing_topic_returns_error(self, research_agent, make_context):
        context = make_context(inputs={})
        result = await research_agent.execute(context)

        assert result.error is not None
        assert "topic" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_empty_topic_returns_error(self, research_agent, make_context):
        context = make_context(inputs={"topic": ""})
        result = await research_agent.execute(context)

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_depth_brief_sent_to_llm(self, research_agent, make_context):
        context = make_context(inputs={"topic": "Machine learning", "depth": "brief"})
        result = await research_agent.execute(context)

        assert result.error is None
        assert result.output is not None
        # Verify depth parameter was forwarded to the LLM prompt
        assert len(research_agent._llm.calls) == 1
        # depth="brief" maps to the concise overview instruction, not the literal word
        assert "concise overview" in research_agent._llm.calls[0]["user_prompt"].lower()

    @pytest.mark.asyncio
    async def test_depth_detailed_sent_to_llm(self, research_agent, make_context):
        context = make_context(inputs={"topic": "Deep learning", "depth": "detailed"})
        result = await research_agent.execute(context)

        assert result.error is None
        assert len(research_agent._llm.calls) == 1
        assert "detailed" in research_agent._llm.calls[0]["user_prompt"].lower()

    @pytest.mark.asyncio
    async def test_usage_populated_on_success(self, research_agent, make_context):
        context = make_context(inputs={"topic": "AI"})
        result = await research_agent.execute(context)

        assert result.error is None
        assert result.usage["input_tokens"] == 50
        assert result.usage["output_tokens"] == 100

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_error(self, error_llm_service, make_context):
        agent = ResearchAgent(llm_service=error_llm_service)
        context = make_context(inputs={"topic": "Python"})
        result = await agent.execute(context)

        assert result.error is not None
        assert "LLM error" in result.error
        assert result.output is None


# ===========================================================================
# AnalysisAgent
# ===========================================================================


class TestAnalysisAgent:
    @pytest.mark.asyncio
    async def test_valid_data_returns_structured_output(self, analysis_agent, make_context):
        context = make_context(inputs={"data": "Sales: Q1=100, Q2=150, Q3=200, Q4=180"})
        result = await analysis_agent.execute(context)

        assert result.error is None
        assert result.output is not None
        assert "analysis" in result.output
        assert "patterns" in result.output
        assert "recommendations" in result.output
        assert "confidence" in result.output
        assert isinstance(result.output["analysis"], str)
        assert isinstance(result.output["patterns"], list)
        assert isinstance(result.output["recommendations"], list)

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_unit_range(self, analysis_agent, make_context):
        context = make_context(inputs={"data": "some numbers"})
        result = await analysis_agent.execute(context)

        assert result.error is None
        assert 0.0 <= result.output["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_missing_data_returns_error(self, analysis_agent, make_context):
        context = make_context(inputs={})
        result = await analysis_agent.execute(context)

        assert result.error is not None
        assert "data" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_invalid_analysis_type_returns_error(self, analysis_agent, make_context):
        context = make_context(
            inputs={"data": "some data", "analysis_type": "invalid_type"}
        )
        result = await analysis_agent.execute(context)

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_sentiment_analysis_type(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "analysis": "Highly positive sentiment detected.",
                    "patterns": ["Positive language", "Enthusiastic tone"],
                    "recommendations": ["Maintain communication style"],
                    "confidence": 0.92,
                }
            ),
            model="mock-model",
            input_tokens=40,
            output_tokens=80,
            latency_ms=150,
        )
        agent = AnalysisAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={"data": "I love this product!", "analysis_type": "sentiment"}
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["confidence"] == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_comparative_analysis_type(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "analysis": "Product A outperforms Product B in three categories.",
                    "patterns": ["A leads on speed", "B leads on cost"],
                    "recommendations": ["Choose A for performance workloads"],
                    "confidence": 0.78,
                }
            ),
            model="mock-model",
            input_tokens=55,
            output_tokens=110,
            latency_ms=220,
        )
        agent = AnalysisAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={
                "data": "Product A: speed=fast, cost=high. Product B: speed=slow, cost=low.",
                "analysis_type": "comparative",
            }
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_error(self, error_llm_service, make_context):
        agent = AnalysisAgent(llm_service=error_llm_service)
        context = make_context(inputs={"data": "some data"})
        result = await agent.execute(context)

        assert result.error is not None
        assert "LLM error" in result.error
        assert result.output is None


# ===========================================================================
# WriterAgent
# ===========================================================================


class TestWriterAgent:
    @pytest.mark.asyncio
    async def test_valid_brief_returns_structured_output(self, writer_agent, make_context):
        context = make_context(inputs={"brief": "Write about the benefits of Python"})
        result = await writer_agent.execute(context)

        assert result.error is None
        assert result.output is not None
        assert "content" in result.output
        assert "word_count" in result.output
        assert "format" in result.output
        assert isinstance(result.output["content"], str)
        assert isinstance(result.output["word_count"], int)
        assert result.output["word_count"] > 0

    @pytest.mark.asyncio
    async def test_missing_brief_returns_error(self, writer_agent, make_context):
        context = make_context(inputs={})
        result = await writer_agent.execute(context)

        assert result.error is not None
        assert "brief" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_empty_brief_returns_error(self, writer_agent, make_context):
        context = make_context(inputs={"brief": ""})
        result = await writer_agent.execute(context)

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_invalid_tone_returns_error(self, writer_agent, make_context):
        context = make_context(inputs={"brief": "Write something", "tone": "angry"})
        result = await writer_agent.execute(context)

        assert result.error is not None
        assert "tone" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_invalid_format_returns_error(self, writer_agent, make_context):
        context = make_context(inputs={"brief": "Write something", "format": "poem"})
        result = await writer_agent.execute(context)

        assert result.error is not None
        assert "format" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_casual_tone_and_report_format(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "content": "Hey! Here's a quick rundown of our Q3 findings...",
                    "format": "report",
                }
            ),
            model="mock-model",
            input_tokens=60,
            output_tokens=120,
            latency_ms=250,
        )
        agent = WriterAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={
                "brief": "Summarise Q3 results for the team",
                "tone": "casual",
                "format": "report",
            }
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["format"] == "report"

    @pytest.mark.asyncio
    async def test_technical_tone_and_summary_format(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "content": "The algorithm achieves O(n log n) complexity via...",
                    "format": "summary",
                }
            ),
            model="mock-model",
            input_tokens=48,
            output_tokens=96,
            latency_ms=190,
        )
        agent = WriterAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={
                "brief": "Explain merge sort to engineers",
                "tone": "technical",
                "format": "summary",
            }
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["format"] == "summary"

    @pytest.mark.asyncio
    async def test_word_count_matches_content(self, make_context, make_mock_llm):
        content = "one two three four five"
        response = LLMResponse(
            content=json.dumps({"content": content, "format": "article"}),
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
        )
        agent = WriterAgent(llm_service=make_mock_llm(response))
        context = make_context(inputs={"brief": "Write something short"})
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["word_count"] == 5

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_error(self, error_llm_service, make_context):
        agent = WriterAgent(llm_service=error_llm_service)
        context = make_context(inputs={"brief": "Write something"})
        result = await agent.execute(context)

        assert result.error is not None
        assert "LLM error" in result.error
        assert result.output is None


# ===========================================================================
# CodeAgent
# ===========================================================================


class TestCodeAgent:
    @pytest.mark.asyncio
    async def test_valid_requirements_returns_structured_output(
        self, code_agent, make_context
    ):
        context = make_context(
            inputs={"requirements": "Write a function to add two numbers"}
        )
        result = await code_agent.execute(context)

        assert result.error is None
        assert result.output is not None
        assert "code" in result.output
        assert "language" in result.output
        assert "explanation" in result.output
        assert "dependencies" in result.output
        assert isinstance(result.output["code"], str)
        assert len(result.output["code"]) > 0
        assert isinstance(result.output["dependencies"], list)

    @pytest.mark.asyncio
    async def test_missing_requirements_returns_error(self, code_agent, make_context):
        context = make_context(inputs={})
        result = await code_agent.execute(context)

        assert result.error is not None
        assert "requirements" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_empty_requirements_returns_error(self, code_agent, make_context):
        context = make_context(inputs={"requirements": ""})
        result = await code_agent.execute(context)

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_language_parameter(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "code": "function add(a, b) { return a + b; }",
                    "language": "javascript",
                    "explanation": "Adds two numbers in JavaScript.",
                    "dependencies": [],
                }
            ),
            model="mock-model",
            input_tokens=45,
            output_tokens=90,
            latency_ms=180,
        )
        agent = CodeAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={"requirements": "Add two numbers", "language": "javascript"}
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["language"] == "javascript"

    @pytest.mark.asyncio
    async def test_dependencies_included_in_output(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "code": "import requests\n\ndef fetch(url): return requests.get(url)",
                    "language": "python",
                    "explanation": "Fetches a URL using requests.",
                    "dependencies": ["requests"],
                }
            ),
            model="mock-model",
            input_tokens=50,
            output_tokens=100,
            latency_ms=200,
        )
        agent = CodeAgent(llm_service=make_mock_llm(response))
        context = make_context(inputs={"requirements": "Fetch a URL"})
        result = await agent.execute(context)

        assert result.error is None
        assert result.output["dependencies"] == ["requests"]

    @pytest.mark.asyncio
    async def test_invalid_style_returns_error(self, code_agent, make_context):
        context = make_context(
            inputs={"requirements": "Write a sorter", "style": "experimental"}
        )
        result = await code_agent.execute(context)

        assert result.error is not None
        assert "style" in result.error.lower()
        assert result.output is None

    @pytest.mark.asyncio
    async def test_verbose_style_accepted(self, make_context, make_mock_llm):
        response = LLMResponse(
            content=json.dumps(
                {
                    "code": (
                        "def add(a: int, b: int) -> int:\n"
                        '    """Add two integers.\n\n'
                        "    Args:\n        a: First operand.\n"
                        "        b: Second operand.\n\n"
                        '    Returns:\n        Sum of a and b.\n    """\n'
                        "    return a + b"
                    ),
                    "language": "python",
                    "explanation": "Fully documented add function.",
                    "dependencies": [],
                }
            ),
            model="mock-model",
            input_tokens=70,
            output_tokens=140,
            latency_ms=280,
        )
        agent = CodeAgent(llm_service=make_mock_llm(response))
        context = make_context(
            inputs={"requirements": "Add two numbers", "style": "verbose"}
        )
        result = await agent.execute(context)

        assert result.error is None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_error(self, error_llm_service, make_context):
        agent = CodeAgent(llm_service=error_llm_service)
        context = make_context(
            inputs={"requirements": "Write a hello world function"}
        )
        result = await agent.execute(context)

        assert result.error is not None
        assert "LLM error" in result.error
        assert result.output is None


# ===========================================================================
# AgentFactory
# ===========================================================================


class TestAgentFactory:
    @pytest.mark.asyncio
    async def test_create_research_agent(self, mock_llm_service):
        agent = AgentFactory.create_agent("research", mock_llm_service)
        assert isinstance(agent, ResearchAgent)

    @pytest.mark.asyncio
    async def test_create_analysis_agent(self, mock_llm_service):
        agent = AgentFactory.create_agent("analysis", mock_llm_service)
        assert isinstance(agent, AnalysisAgent)

    @pytest.mark.asyncio
    async def test_create_writer_agent(self, mock_llm_service):
        agent = AgentFactory.create_agent("writer", mock_llm_service)
        assert isinstance(agent, WriterAgent)

    @pytest.mark.asyncio
    async def test_create_code_agent(self, mock_llm_service):
        agent = AgentFactory.create_agent("code", mock_llm_service)
        assert isinstance(agent, CodeAgent)

    @pytest.mark.asyncio
    async def test_create_unknown_type_raises_value_error(self, mock_llm_service):
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create_agent("unknown_type", mock_llm_service)

    @pytest.mark.asyncio
    async def test_create_empty_type_raises_value_error(self, mock_llm_service):
        with pytest.raises(ValueError):
            AgentFactory.create_agent("", mock_llm_service)

    @pytest.mark.asyncio
    async def test_register_agents_populates_registry(self, mock_llm_service):
        registry = AgentRegistry()
        factory = AgentFactory(llm_service=mock_llm_service)
        factory.register_agents(registry)

        assert registry.has("research")
        assert registry.has("analysis")
        assert registry.has("writer")
        assert registry.has("code")

    @pytest.mark.asyncio
    async def test_register_agents_correct_agent_types(self, mock_llm_service):
        registry = AgentRegistry()
        factory = AgentFactory(llm_service=mock_llm_service)
        factory.register_agents(registry)

        assert isinstance(registry.get("research"), ResearchAgent)
        assert isinstance(registry.get("analysis"), AnalysisAgent)
        assert isinstance(registry.get("writer"), WriterAgent)
        assert isinstance(registry.get("code"), CodeAgent)

    @pytest.mark.asyncio
    async def test_register_agents_all_types_present(self, mock_llm_service):
        registry = AgentRegistry()
        factory = AgentFactory(llm_service=mock_llm_service)
        factory.register_agents(registry)

        assert set(registry.available_types) == {"research", "analysis", "writer", "code"}

    @pytest.mark.asyncio
    async def test_registered_agents_are_executable(
        self, mock_llm_service, make_context, make_mock_llm
    ):
        """Agents registered by the factory can be retrieved and executed."""
        llm = make_mock_llm(
            LLMResponse(
                content=json.dumps(
                    {
                        "findings": "Test finding.",
                        "key_points": ["Point A"],
                        "sources": ["Source 1"],
                    }
                ),
                model="mock-model",
                input_tokens=10,
                output_tokens=20,
                latency_ms=100,
            )
        )
        registry = AgentRegistry()
        factory = AgentFactory(llm_service=llm)
        factory.register_agents(registry)

        agent = registry.get("research")
        assert agent is not None

        context = make_context(inputs={"topic": "Testing"})
        result = await agent.execute(context)

        assert result.error is None
        assert result.output is not None


# ===========================================================================
# Error handling — cross-agent
# ===========================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_research_agent_propagates_llm_error(
        self, error_llm_service, make_context
    ):
        agent = ResearchAgent(llm_service=error_llm_service)
        result = await agent.execute(make_context(inputs={"topic": "AI"}))

        assert result.error is not None
        assert result.output is None
        assert "rate limit" in result.error.lower() or "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_analysis_agent_propagates_llm_error(
        self, error_llm_service, make_context
    ):
        agent = AnalysisAgent(llm_service=error_llm_service)
        result = await agent.execute(make_context(inputs={"data": "numbers"}))

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_writer_agent_propagates_llm_error(
        self, error_llm_service, make_context
    ):
        agent = WriterAgent(llm_service=error_llm_service)
        result = await agent.execute(make_context(inputs={"brief": "Write content"}))

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_code_agent_propagates_llm_error(
        self, error_llm_service, make_context
    ):
        agent = CodeAgent(llm_service=error_llm_service)
        result = await agent.execute(
            make_context(inputs={"requirements": "Write code"})
        )

        assert result.error is not None
        assert result.output is None

    @pytest.mark.asyncio
    async def test_agent_result_has_no_usage_on_error(
        self, error_llm_service, make_context
    ):
        """On LLM failure, the agent should not expose misleading token counts."""
        agent = ResearchAgent(llm_service=error_llm_service)
        result = await agent.execute(make_context(inputs={"topic": "Python"}))

        assert result.error is not None
        # usage defaults to an empty dict when the agent returns early on error
        assert result.usage == {}
