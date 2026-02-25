"""Analysis agent that extracts patterns and insights from provided data."""

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

_VALID_ANALYSIS_TYPES = {"sentiment", "statistical", "comparative"}


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


def _clamp_confidence(value: Any) -> float:
    """Coerce *value* to a float clamped to ``[0.0, 1.0]``."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class AnalysisAgent(BaseAgent):
    """Agent that analyses data and extracts actionable insights.

    Expected inputs:
        data (str | Any): The data or text to analyse. Required.
        analysis_type (str): ``"sentiment"``, ``"statistical"``, or
            ``"comparative"``. Optional, defaults to ``"statistical"``.

    Output schema::

        {
            "analysis":        str,          # Narrative describing findings
            "patterns":        list[str],    # Detected patterns or trends
            "recommendations": list[str],    # Actionable next steps
            "confidence":      float,        # Analyst confidence in [0.0, 1.0]
        }
    """

    SYSTEM_PROMPT = (
        "You are a data analysis specialist. Analyze the provided data and "
        "extract meaningful patterns, trends, and actionable insights."
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
        """Analyse the data from context and return structured insights.

        Args:
            context: Execution context. ``context.inputs`` must contain
                ``"data"`` and may optionally contain ``"analysis_type"``
                (``"sentiment"``, ``"statistical"``, or ``"comparative"``).

        Returns:
            :class:`~agentflow.agents.base.AgentResult` with ``output`` dict
            containing ``analysis``, ``patterns``, ``recommendations``, and
            ``confidence``. On failure ``error`` is set and ``output`` is
            ``None``.
        """
        data = context.inputs.get("data")
        if data is None:
            return AgentResult(error="Missing required input: 'data'")

        analysis_type: str = context.inputs.get("analysis_type", "statistical")
        if analysis_type not in _VALID_ANALYSIS_TYPES:
            return AgentResult(
                error=(
                    f"Invalid analysis_type {analysis_type!r}. "
                    f"Must be one of: {', '.join(sorted(_VALID_ANALYSIS_TYPES))}"
                )
            )

        self._log.info(
            "analysis_agent_start",
            analysis_type=analysis_type,
            run_id=str(context.run_id),
            step_id=str(context.step_id),
        )

        type_guidance = {
            "sentiment": (
                "Focus on emotional tone, sentiment polarity, and subjective "
                "language patterns. Express confidence as a probability."
            ),
            "statistical": (
                "Focus on numerical distributions, frequencies, outliers, and "
                "statistical trends present in the data."
            ),
            "comparative": (
                "Compare and contrast the elements present, highlighting "
                "similarities, differences, and relative strengths."
            ),
        }[analysis_type]

        user_prompt = (
            f"Perform a {analysis_type} analysis on the following data:\n\n"
            f"{data}\n\n"
            f"Analysis guidance: {type_guidance}\n\n"
            "Respond with a JSON object containing exactly these keys:\n"
            '  "analysis"        – a detailed narrative of your findings\n'
            '  "patterns"        – an array of pattern/trend strings detected\n'
            '  "recommendations" – an array of actionable recommendation strings\n'
            '  "confidence"      – a float between 0.0 and 1.0 indicating your '
            "confidence in the analysis\n\n"
            "Return only the JSON object with no surrounding text."
        )

        llm_response = await self._llm.completion(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2048,
        )

        if not llm_response.success:
            self._log.error("analysis_agent_llm_error", error=llm_response.error)
            return AgentResult(error=f"LLM error: {llm_response.error}")

        parsed = _try_parse_json(llm_response.content)
        if parsed is not None:
            output: dict[str, Any] = {
                "analysis": str(parsed.get("analysis", "")),
                "patterns": [str(p) for p in parsed.get("patterns", [])],
                "recommendations": [str(r) for r in parsed.get("recommendations", [])],
                "confidence": _clamp_confidence(parsed.get("confidence", 0.0)),
            }
        else:
            self._log.warning(
                "analysis_agent_json_parse_failed",
                analysis_type=analysis_type,
            )
            output = {
                "analysis": llm_response.content,
                "patterns": [],
                "recommendations": [],
                "confidence": 0.0,
            }

        self._log.info(
            "analysis_agent_complete",
            analysis_type=analysis_type,
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
