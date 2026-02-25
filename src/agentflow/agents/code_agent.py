"""Code agent that generates production-quality code from requirements."""

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

_VALID_STYLES = {"concise", "verbose"}


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


class CodeAgent(BaseAgent):
    """Agent that generates clean, production-quality code from requirements.

    Expected inputs:
        requirements (str): Description of what the code should do. Required.
        language (str): Target programming language. Optional, defaults to
            ``"python"``.
        style (str): ``"concise"`` (minimal, focused) or ``"verbose"``
            (fully documented). Optional, defaults to ``"concise"``.

    Output schema::

        {
            "code":         str,        # The generated source code
            "language":     str,        # Language of the generated code
            "explanation":  str,        # Narrative explanation of the code
            "dependencies": list[str],  # Required packages / imports
        }
    """

    SYSTEM_PROMPT = (
        "You are an expert software engineer. Write clean, well-documented, "
        "production-quality code based on the requirements."
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
        """Generate code from the requirements in context inputs.

        Args:
            context: Execution context. ``context.inputs`` must contain
                ``"requirements"`` and may optionally contain ``"language"``
                (default ``"python"``) and ``"style"``
                (``"concise"`` or ``"verbose"``).

        Returns:
            :class:`~agentflow.agents.base.AgentResult` with ``output`` dict
            containing ``code``, ``language``, ``explanation``, and
            ``dependencies``. On failure ``error`` is set and ``output`` is
            ``None``.
        """
        requirements: str | None = context.inputs.get("requirements")
        if not requirements:
            return AgentResult(error="Missing required input: 'requirements'")

        language: str = context.inputs.get("language", "python")
        style: str = context.inputs.get("style", "concise")

        if style not in _VALID_STYLES:
            return AgentResult(
                error=(
                    f"Invalid style {style!r}. "
                    f"Must be one of: {', '.join(sorted(_VALID_STYLES))}"
                )
            )

        self._log.info(
            "code_agent_start",
            language=language,
            style=style,
            run_id=str(context.run_id),
            step_id=str(context.step_id),
        )

        style_guidance = {
            "concise": (
                "Write minimal, focused code. Avoid boilerplate. Use inline "
                "comments only where the logic is non-obvious."
            ),
            "verbose": (
                "Write fully documented code with docstrings for every "
                "function/class, type annotations, and explanatory comments."
            ),
        }[style]

        user_prompt = (
            f"Write {language} code that satisfies the following requirements:\n\n"
            f"{requirements}\n\n"
            f"Style: {style_guidance}\n\n"
            "Respond with a JSON object containing exactly these keys:\n"
            '  "code"         – the complete source code as a single string\n'
            '  "language"     – the programming language used\n'
            '  "explanation"  – a concise explanation of how the code works\n'
            '  "dependencies" – an array of external package names required '
            "(empty array if none)\n\n"
            "Return only the JSON object with no surrounding text."
        )

        llm_response = await self._llm.completion(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=3000,
        )

        if not llm_response.success:
            self._log.error("code_agent_llm_error", error=llm_response.error)
            return AgentResult(error=f"LLM error: {llm_response.error}")

        parsed = _try_parse_json(llm_response.content)
        if parsed is not None:
            output: dict[str, Any] = {
                "code": str(parsed.get("code", "")),
                "language": str(parsed.get("language", language)),
                "explanation": str(parsed.get("explanation", "")),
                "dependencies": [str(d) for d in parsed.get("dependencies", [])],
            }
        else:
            self._log.warning(
                "code_agent_json_parse_failed",
                language=language,
                style=style,
            )
            output = {
                "code": llm_response.content,
                "language": language,
                "explanation": "",
                "dependencies": [],
            }

        self._log.info(
            "code_agent_complete",
            language=language,
            style=style,
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
