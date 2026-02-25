"""Research agent that gathers and synthesises information on a topic."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

import structlog

from agentflow.agents.base import AgentContext, AgentResult, BaseAgent
from agentflow.services.llm_service import LLMService

logger = structlog.get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Return the first valid JSON object found in *text*, or ``None``.

    Attempts a direct parse first, then searches for a JSON fenced code block.
    """
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


class ResearchAgent(BaseAgent):
    """Agent that researches a topic and returns structured findings.

    Expected inputs:
        topic (str): The subject to research. Required.
        depth (str): ``"brief"`` or ``"detailed"``. Optional, defaults to
            ``"detailed"``.

    Output schema::

        {
            "findings":   str,          # Full research narrative
            "key_points": list[str],    # Concise bullet-point insights
            "sources":    list[str],    # References or source types cited
        }
    """

    SYSTEM_PROMPT = (
        "You are a research specialist. Analyze the given topic and provide "
        "comprehensive findings with sources and key insights."
    )

    def __init__(
        self,
        llm_service: LLMService,
        agent_id: UUID | None = None,
        name: str = "",
    ) -> None:
        super().__init__(agent_id=agent_id, name=name)
        self._llm = llm_service
        self._log = logger.bind(agent=self.name, agent_id=str(self.agent_id))

    async def execute(self, context: AgentContext) -> AgentResult:
        """Research a topic and return structured findings.

        Args:
            context: Execution context. ``context.inputs`` must contain
                ``"topic"`` and may optionally contain ``"depth"``
                (``"brief"`` or ``"detailed"``).

        Returns:
            :class:`~agentflow.agents.base.AgentResult` with ``output`` dict
            containing ``findings``, ``key_points``, and ``sources``. On
            failure ``error`` is set and ``output`` is ``None``.
        """
        topic: str | None = context.inputs.get("topic")
        if not topic:
            return AgentResult(error="Missing required input: 'topic'")

        depth: str = context.inputs.get("depth", "detailed")

        self._log.info(
            "research_agent_start",
            topic=topic,
            depth=depth,
            run_id=str(context.run_id),
            step_id=str(context.step_id),
        )

        depth_instruction = (
            "Provide a concise overview with 3–5 key points."
            if depth == "brief"
            else "Provide a thorough, detailed analysis covering all significant aspects."
        )

        user_prompt = (
            f"Research the following topic: {topic}\n\n"
            f"{depth_instruction}\n\n"
            "Respond with a JSON object containing exactly these keys:\n"
            '  "findings"   – a detailed narrative string of your research\n'
            '  "key_points" – an array of concise insight strings\n'
            '  "sources"    – an array of relevant source names or types\n\n'
            "Return only the JSON object with no surrounding text."
        )

        llm_response = await self._llm.completion(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )

        if not llm_response.success:
            self._log.error("research_agent_llm_error", error=llm_response.error)
            return AgentResult(error=f"LLM error: {llm_response.error}")

        parsed = _try_parse_json(llm_response.content)
        if parsed is not None:
            output: dict[str, Any] = {
                "findings": str(parsed.get("findings", "")),
                "key_points": [str(p) for p in parsed.get("key_points", [])],
                "sources": [str(s) for s in parsed.get("sources", [])],
            }
        else:
            self._log.warning("research_agent_json_parse_failed", topic=topic)
            output = {
                "findings": llm_response.content,
                "key_points": [],
                "sources": [],
            }

        self._log.info(
            "research_agent_complete",
            topic=topic,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
        )

        return AgentResult(
            output=output,
            usage={
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
            },
        )
