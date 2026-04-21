"""AlignAgent — LLM-driven subtitle alignment agent.

Splits a single translated string into ``N`` sub-strings, one per source
segment, such that concatenating the outputs reproduces the input
translation (modulo whitespace / trailing punctuation).

Mirrors the legacy ``translate.AlignAgent`` from the v1 codebase:
JSON-mode system prompt, validation that ``len(output) == len(segments)``,
and a concat-equality check.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from domain.lang import LangOps
from ports.engine import LLMEngine
from ports.retries import AttemptOutcome, retry_until_valid

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a subtitle-splitting expert. The user provides a list of source-language
segments and a single target-language translation. Split the translation into
pieces that, when concatenated in order, reproduce the target-language input
exactly (whitespace and punctuation may be adjusted). Output strictly the JSON
object below; no prose, no markdown.

Input JSON:
{{
  "source_segments": ["<segment 1>", "<segment 2>", ...],
  "target_text":     "<single string>"
}}

Output JSON:
{{
  "mapping": [
    {{"source": "<segment 1>", "target": "<target piece 1>"}},
    {{"source": "<segment 2>", "target": "<target piece 2>"}}
  ]
}}

Rules:
1. ``len(mapping) == len(source_segments)``.
2. Keep natural reading flow in the target language; do not force
   word-level alignment.
3. A target piece may be empty only if the source segment has no
   meaningful content to map.
4. Preserve the order of the source segments.
"""


_JSON_RE = re.compile(r"\{[\s\S]*\}")
_FENCE_RE = re.compile(r"^\s*```(?:json)?|```\s*$", re.IGNORECASE | re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AlignResult:
    """Output of :meth:`AlignAgent.align`."""

    pieces: list[str]
    accepted: bool
    attempts: int
    reason: str = ""


class AlignAgent:
    """LLM agent that splits a translated string to match ``N`` source segments.

    Args:
        engine: :class:`LLMEngine` used for chat completions.
        target_lang: Target language code; used by :class:`LangOps` for
            length / concat-equality normalization.
        max_retries: Retry budget (0 = single attempt).
        tolerate_ratio: Accept results when the concat-equality check
            rejects but the length ratio between joined output and the
            expected text stays within ``(1 - r, 1 + r)``. Mirrors the
            legacy ``norm_ratio`` / ``accept_ratio`` behavior.
    """

    def __init__(
        self,
        engine: LLMEngine,
        target_lang: str,
        *,
        max_retries: int = 2,
        tolerate_ratio: float = 0.1,
        ops: LangOps | None = None,
    ) -> None:
        self._engine = engine
        self._target_lang = target_lang
        self._max_retries = max(0, int(max_retries))
        self._tolerate_ratio = max(0.0, float(tolerate_ratio))
        self._ops = ops or LangOps.for_language(target_lang)

    async def align(self, source_segments: list[str], target_text: str) -> AlignResult:
        """Split ``target_text`` into ``len(source_segments)`` pieces."""
        if not source_segments:
            return AlignResult(pieces=[], accepted=True, attempts=0)

        n = len(source_segments)
        target_text = target_text or ""
        if n == 1:
            return AlignResult(pieces=[target_text], accepted=True, attempts=0)

        if not target_text.strip():
            return AlignResult(pieces=[""] * n, accepted=True, attempts=0)

        expected = self._normalize(target_text)

        async def call(_attempt: int) -> list[str]:
            return await self._invoke(source_segments, target_text)

        def validate(pieces: list[str]) -> tuple[bool, list[str] | None, str]:
            if len(pieces) != n:
                return False, None, f"length mismatch ({len(pieces)} != {n})"
            joined = self._normalize("".join(pieces))
            if joined == expected:
                return True, pieces, ""
            if expected and self._within_ratio(joined, expected):
                return True, pieces, ""
            return False, None, "concat mismatch"

        outcome: AttemptOutcome[list[str]] = await retry_until_valid(
            call,
            validate=validate,
            max_retries=self._max_retries,
            on_reject=lambda i, r: logger.debug("align reject attempt=%d reason=%s", i, r),
            on_exception=lambda i, exc: logger.warning("align exception attempt=%d: %r", i, exc),
        )

        if outcome.accepted and outcome.value is not None:
            return AlignResult(pieces=outcome.value, accepted=True, attempts=outcome.attempts)

        logger.warning(
            "AlignAgent giving up after %d attempts (reason=%s); falling back to target_text at idx 0.",
            outcome.attempts,
            outcome.last_reason,
        )
        fallback = [target_text] + [""] * (n - 1)
        return AlignResult(
            pieces=fallback,
            accepted=False,
            attempts=outcome.attempts,
            reason=outcome.last_reason,
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _invoke(self, source_segments: list[str], target_text: str) -> list[str]:
        payload = {
            "source_segments": source_segments,
            "target_text": target_text,
        }
        user = json.dumps(payload, ensure_ascii=False)

        response = await self._engine.complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        return _parse_align_response(response.text, expected_n=len(source_segments))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text).strip()

    def _within_ratio(self, a: str, b: str) -> bool:
        ops = self._ops
        la = ops.length(a)
        lb = ops.length(b)
        if lb == 0:
            return la == 0
        return abs(la - lb) / lb <= self._tolerate_ratio


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_align_response(text: str, *, expected_n: int) -> list[str]:
    cleaned = _THINK_RE.sub("", text or "")
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError("AlignAgent: no JSON object in response")

    data: Any = json.loads(match.group(0))
    mapping = data.get("mapping") if isinstance(data, dict) else None
    if not isinstance(mapping, list):
        raise ValueError("AlignAgent: missing 'mapping' array")

    pieces: list[str] = []
    for item in mapping:
        if isinstance(item, dict):
            piece = item.get("target") or item.get("zh") or item.get("tgt") or ""
        elif isinstance(item, str):
            piece = item
        else:
            piece = ""
        pieces.append(str(piece))

    if len(pieces) != expected_n:
        logger.debug("AlignAgent parsed %d pieces (expected %d)", len(pieces), expected_n)
    return pieces


__all__ = ["AlignAgent", "AlignResult"]
