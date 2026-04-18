"""LLM-backed summary / terms extraction agents.

:class:`TermsAgent` calls an :class:`LLMEngine` to extract domain terminology
and summary metadata (topic/title/description) from a block of source text.
Shared by :class:`PreloadableTerms` and :class:`OneShotTerms` in
:mod:`llm_ops.providers`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .protocol import LLMEngine

logger = logging.getLogger(__name__)


_TERMS_SYSTEM_PROMPT = """\
You are a terminology-extraction assistant. Given source-language text,
extract structured metadata and technical terminology useful for subtitle
translation from {src} to {tgt}.

Return ONLY a JSON object with these keys:
- "topic":       domain / field (one short phrase, e.g. "machine learning")
- "title":       one-line summary of what the text is about
- "description": 1-2 sentence overview
- "terms":       {{src → tgt}} mapping of technical terminology only
                 (skip common words; keep proper nouns, jargon, acronyms)

Example:
{{
  "topic": "deep learning",
  "title": "Lecture on gradient descent",
  "description": "An introductory explanation of gradient descent and backpropagation.",
  "terms": {{"gradient descent": "梯度下降", "backpropagation": "反向传播"}}
}}
"""


@dataclass(frozen=True)
class TermsAgentResult:
    """Output of :meth:`TermsAgent.extract`."""

    terms: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "TermsAgentResult":
        return cls(terms={}, metadata={})


class TermsAgent:
    """Extract terminology + summary metadata from a text block via an LLM."""

    __slots__ = ("_engine", "_src", "_tgt", "_max_input_chars")

    def __init__(
        self,
        engine: LLMEngine,
        source_lang: str,
        target_lang: str,
        *,
        max_input_chars: int = 8000,
    ):
        self._engine = engine
        self._src = source_lang
        self._tgt = target_lang
        self._max_input_chars = max_input_chars

    async def extract(self, texts: list[str]) -> TermsAgentResult:
        """Call the LLM and parse a :class:`TermsAgentResult`.

        Raises the engine's exception on failure; callers are expected to
        wrap in their own retry / degradation logic.
        """
        body = "\n".join(t.strip() for t in texts if t and t.strip())
        if len(body) > self._max_input_chars:
            body = body[: self._max_input_chars]

        system = _TERMS_SYSTEM_PROMPT.format(src=self._src, tgt=self._tgt)
        response = await self._engine.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ])
        return parse_terms_response(response.text)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_terms_response(text: str) -> TermsAgentResult:
    """Parse an LLM response into a :class:`TermsAgentResult`.

    Tolerant of ``<think>`` tags and surrounding prose; returns
    :meth:`TermsAgentResult.empty` if no valid JSON object is found.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    match = _JSON_RE.search(cleaned)
    if not match:
        logger.warning("TermsAgent: no JSON object in response")
        return TermsAgentResult.empty()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("TermsAgent: JSON decode failed: %s", exc)
        return TermsAgentResult.empty()

    raw_terms = data.get("terms") or {}
    terms: dict[str, str] = {}
    if isinstance(raw_terms, dict):
        for k, v in raw_terms.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                terms[k.strip()] = v.strip()

    metadata: dict[str, str] = {}
    for key in ("topic", "title", "description"):
        value = data.get(key)
        if isinstance(value, str):
            metadata[key] = value.strip()

    return TermsAgentResult(terms=terms, metadata=metadata)
