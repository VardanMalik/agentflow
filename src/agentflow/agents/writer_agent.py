"""Writer agent that generates structured content from a brief."""

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

_VALID_TONES = {"formal", "casual", "technical"}
_VALID_FORMATS = {"article", "report", "summary"}


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


def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


class WriterAgent(BaseAgent):
    """Agent that generates well-structured written content from a brief.

    Expected inputs:
        brief (str): Description of the content to write. Required.
        tone (str): ``"formal"``, ``"casual"``, or ``"technical"``. Optional,
            defaults to ``"formal"``.
        format (str): ``"article"``, ``"report"``, or ``"summary"``. Optional,
            defaults to ``"article"``.

    Output schema::

        {
            "content":    str,   # The generated written content
            "word_count": int,   # Number of words in the generated content
            "format":     str,   # The format that was applied
        }
    """

    SYSTEM_PROMPT = (
        "You are a professional content writer. Create well-structured, "
        "engaging content based on the provided brief and research."
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
        """Generate written content from the brief in context inputs.

        Args:
            context: Execution context. ``context.inputs`` must contain
                ``"brief"`` and may optionally contain ``"tone"``
                (``"formal"``, ``"casual"``, or ``"technical"``) and
                ``"format"`` (``"article"``, ``"report"``, or ``"summary"``).

        Returns:
            :class:`~agentflow.agents.base.AgentResult` with ``output`` dict
            containing ``content``, ``word_count``, and ``format``. On
            failure ``error`` is set and ``output`` is ``None``.
        """
        brief: str | None = context.inputs.get("brief")
        if not brief:
            return AgentResult(error="Missing required input: 'brief'")

        tone: str = context.inputs.get("tone", "formal")
        if tone not in _VALID_TONES:
            return AgentResult(
                error=(
                    f"Invalid tone {tone!r}. "
                    f"Must be one of: {', '.join(sorted(_VALID_TONES))}"
                )
            )

        fmt: str = context.inputs.get("format", "article")
        if fmt not in _VALID_FORMATS:
            return AgentResult(
                error=(
                    f"Invalid format {fmt!r}. "
                    f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}"
                )
            )

        self._log.info(
            "writer_agent_start",
            tone=tone,
            format=fmt,
            run_id=str(context.run_id),
            step_id=str(context.step_id),
        )

        format_guidance = {
            "article": (
                "Write a full article with an introduction, body sections, "
                "and a conclusion. Use markdown headings where appropriate."
            ),
            "report": (
                "Write a structured report with an executive summary, "
                "detailed findings, and recommendations."
            ),
            "summary": (
                "Write a concise summary that captures the most important "
                "points in 2–4 paragraphs."
            ),
        }[fmt]

        tone_guidance = {
            "formal": "Use formal, professional language appropriate for business or academic audiences.",
            "casual": "Use approachable, conversational language that is easy to read.",
            "technical": "Use precise, technical language appropriate for a specialist audience.",
        }[tone]

        user_prompt = (
            f"Write {fmt} content based on the following brief:\n\n"
            f"{brief}\n\n"
            f"Tone: {tone_guidance}\n"
            f"Format guidance: {format_guidance}\n\n"
            "Respond with a JSON object containing exactly these keys:\n"
            '  "content" – the full generated text as a single string\n'
            '  "format"  – the format name you used (e.g. "article")\n\n'
            "Return only the JSON object with no surrounding text."
        )

        llm_response = await self._llm.completion(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=3000,
        )

        if not llm_response.success:
            self._log.error("writer_agent_llm_error", error=llm_response.error)
            return AgentResult(error=f"LLM error: {llm_response.error}")

        parsed = _try_parse_json(llm_response.content)
        if parsed is not None:
            content = str(parsed.get("content", ""))
            applied_format = str(parsed.get("format", fmt))
        else:
            self._log.warning("writer_agent_json_parse_failed", tone=tone, format=fmt)
            content = llm_response.content
            applied_format = fmt

        output: dict[str, Any] = {
            "content": content,
            "word_count": _word_count(content),
            "format": applied_format,
        }

        self._log.info(
            "writer_agent_complete",
            tone=tone,
            format=fmt,
            word_count=output["word_count"],
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
