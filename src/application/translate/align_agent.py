"""AlignAgent — LLM-driven binary-split alignment agent.

Ports the legacy ``translate.AlignAgent`` semantics:

- **Binary only.** Each call splits a translation into **exactly two** pieces
  against **exactly two** source half-texts. N>2 is handled by the caller
  (see :class:`AlignProcessor`) via recursive bisection.
- **Dual mode.**

  - ``use_json=True`` — stable JSON schema, 1 retry, lax ratio thresholds
    (norm=accept=5). Used for outer recursion.
  - ``use_json=False`` — two-line text output, 6 retries, strict ratio
    (norm=accept=3), prompt degradation + target-text trimming after two
    failed attempts. Used for the rearrange pass.

- **Validation.** After parsing, the result must
    1. contain exactly two non-empty pieces,
    2. concatenate back to the full translation (via
       :meth:`LangOps.check_and_correct_split_sentence`),
    3. produce a cross-language ratio within the accept threshold.
- **Rearrange hint.** When the ratio is ≥ ``norm_ratio`` but < ``accept_ratio``
  the result is accepted but flagged ``need_rearrange=True`` — the caller may
  then rebalance source-side word boundaries (see
  :func:`domain.subtitle.rebalance_segment_words`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from application.translate.align_ratio import cross_ratio
from domain.lang import LangOps
from ports.engine import LLMEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts (ported verbatim from legacy AlignAgent, language-neutralized where
# possible but keeping the effective Chinese-text output constraint — the
# legacy system only ever ran with target=zh. Callers should pass the
# target-language LangOps so ratio/concat checks work for any language.)
# ---------------------------------------------------------------------------


_JSON_SYSTEM_PROMPT = """\
你是一名专业的字幕分割专家，用户会提供两个源语言片段和对应的目标语言字幕，你负责将目标语言字幕切分为能够与源语言片段同步播放的两个片段。

用户输入格式如下：
{
    "src_chunks": ["<源语言片段1>", "<源语言片段2>"],
    "tgt_text": "<对应的目标语言字幕>"
}

你需要输出的 JSON 格式如下：
{
    "mapping": [
        {"src": "<源语言片段1>", "tgt": "<对应的目标语言字幕片段1>"},
        {"src": "<源语言片段2>", "tgt": "<对应的目标语言字幕片段2>"}
    ]
}

注意：两个目标语言字幕片段拼接起来必须与输入的 tgt_text 字段完全一致。
"""


_TEXT_SYSTEM_PROMPT = """\
你是一名语言分析专家，擅长根据语法结构将给出的句子分割成两部分。
必须分割为两部分，同时只需按行输出结果，不包含任何解释或额外信息。
"""


_TEXT_DEGRADED_SYSTEM_PROMPT = _TEXT_SYSTEM_PROMPT


_FENCE_RE = re.compile(r"^\s*```(?:json)?|```\s*$", re.IGNORECASE | re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BisectResult:
    """Outcome of a single :meth:`AlignAgent.bisect` call."""

    pieces: list[str]
    """Two target-language pieces (or empty strings on giving up)."""

    accepted: bool
    """True if the pieces passed concat + ratio validation."""

    need_rearrange: bool
    """True if ratio ≥ norm but < accept (caller may rebalance source words)."""

    attempts: int
    reason: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AlignAgent:
    """LLM agent that binary-splits a single translation into two pieces.

    Args:
        engine: :class:`LLMEngine` for chat completions.
        target_lang: Target-language code (drives concat / ratio validation).
        use_json: ``True`` for JSON mode (1 retry, relaxed ratios, no
            rearrange), ``False`` for text mode (6 retries, strict ratios,
            optional rearrange hint, prompt degradation on retry ≥ 2).
        max_retries: Override the retry budget. ``None`` selects the
            mode-appropriate default (JSON=1, text=6).
        source_ops: Source-language ``LangOps`` (defaults to English).
        target_ops: Target-language ``LangOps`` (defaults to ``target_lang``).
    """

    def __init__(
        self,
        engine: LLMEngine,
        target_lang: str,
        *,
        use_json: bool = True,
        max_retries: int | None = None,
        source_lang: str = "en",
        source_ops: LangOps | None = None,
        target_ops: LangOps | None = None,
    ) -> None:
        self._engine = engine
        self._target_lang = target_lang
        self._source_lang = source_lang
        self._use_json = bool(use_json)
        self._max_retries = max_retries if max_retries is not None else (1 if use_json else 6)
        self._max_retries = max(0, int(self._max_retries))
        self._src_ops = source_ops or LangOps.for_language(source_lang)
        self._tgt_ops = target_ops or LangOps.for_language(target_lang)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def use_json(self) -> bool:
        return self._use_json

    async def bisect(
        self,
        src_texts: list[str],
        translation: str,
        *,
        norm_ratio: float,
        accept_ratio: float,
    ) -> BisectResult:
        """Split *translation* into two pieces aligning with *src_texts* (length=2).

        Args:
            src_texts: Exactly two source-language half-texts.
            translation: Full target-language translation to split.
            norm_ratio: Below this cross-ratio, accept without rearrange hint.
            accept_ratio: Below this cross-ratio, accept with rearrange hint.
        """
        if len(src_texts) != 2:
            raise ValueError(f"bisect expects exactly 2 source texts, got {len(src_texts)}")
        if not translation or not translation.strip():
            return BisectResult(pieces=["", ""], accepted=False, need_rearrange=False, attempts=0, reason="empty translation")

        system_prompt = _JSON_SYSTEM_PROMPT if self._use_json else _TEXT_SYSTEM_PROMPT
        runtime_translation = self._tgt_ops.rstrip_punc(translation)
        last_reason = ""
        current_ratio = norm_ratio / 2

        for attempt in range(self._max_retries + 1):
            # Prompt degradation for text mode from retry ≥ 2: identical prompt
            # body but trim 3 target chars to force the LLM to re-plan the split.
            if not self._use_json and attempt >= 2:
                system_prompt = _TEXT_DEGRADED_SYSTEM_PROMPT
                if self._tgt_ops.length(runtime_translation) > 10:
                    runtime_translation = runtime_translation[:-3]

            try:
                raw = await self._call_llm(system_prompt, src_texts, runtime_translation)
            except Exception as exc:  # network / engine error
                last_reason = f"llm error: {exc!r}"
                logger.debug("align attempt=%d exception: %r", attempt, exc)
                continue

            try:
                pieces = self._parse(raw)
            except ValueError as exc:
                last_reason = f"parse: {exc}"
                logger.debug("align attempt=%d parse failed: %s", attempt, exc)
                continue

            if len(pieces) != 2 or any(not p.strip() for p in pieces):
                last_reason = f"bad shape: {pieces}"
                continue

            # Concat validation + CJK-aware reversal fixup.
            good, fixed = self._tgt_ops.check_and_correct_split_sentence(pieces, translation, can_reverse=True)
            if not good:
                last_reason = "concat mismatch"
                continue

            ratio = cross_ratio(src_texts, fixed, self._src_ops, self._tgt_ops)
            if ratio <= norm_ratio:
                return BisectResult(pieces=fixed, accepted=True, need_rearrange=False, attempts=attempt + 1)
            if ratio < accept_ratio:
                current_ratio = ratio
                return BisectResult(pieces=fixed, accepted=True, need_rearrange=True, attempts=attempt + 1)
            last_reason = f"ratio too high ({ratio:.2f})"

        logger.warning(
            "AlignAgent exhausted retries (%d) use_json=%s reason=%s; giving up.",
            self._max_retries + 1,
            self._use_json,
            last_reason,
        )
        return BisectResult(
            pieces=["", ""],
            accepted=False,
            need_rearrange=False,
            attempts=self._max_retries + 1,
            reason=last_reason,
        )

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    async def _call_llm(self, system_prompt: str, src_texts: list[str], translation: str) -> str:
        if self._use_json:
            user = json.dumps({"src_chunks": src_texts, "tgt_text": translation}, ensure_ascii=False)
        else:
            # Legacy text-mode prompt: only the target translation is shown,
            # the LLM must split it on its own grammatical intuition.
            user = f"用户的给出句子为：“{translation}”"

        response = await self._engine.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ]
        )
        return response.text or ""

    def _parse(self, raw: str) -> list[str]:
        cleaned = _THINK_RE.sub("", raw or "").strip()
        if self._use_json:
            return _parse_json_mapping(cleaned)
        return _parse_text_lines(cleaned)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_mapping(raw: str) -> list[str]:
    """Parse the JSON-mode ``{"mapping":[{src,tgt},{src,tgt}]}`` response."""
    # Defense-in-depth: strip code fences, trailing commas, fallback to body scan.
    body = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(body)
    except Exception:
        body2 = _TRAILING_COMMA_RE.sub(r"\1", body)
        try:
            data = json.loads(body2)
        except Exception:
            match = _JSON_OBJ_RE.search(body2)
            if not match:
                raise ValueError("no JSON object in response") from None
            data = json.loads(_TRAILING_COMMA_RE.sub(r"\1", match.group(0)))

    mapping = data.get("mapping") if isinstance(data, dict) else None
    if not isinstance(mapping, list):
        raise ValueError("missing 'mapping' array")

    pieces: list[str] = []
    for item in mapping:
        if isinstance(item, dict):
            # Accept legacy and new key names.
            piece = item.get("tgt") or item.get("zh") or item.get("target") or ""
        elif isinstance(item, str):
            piece = item
        else:
            piece = ""
        pieces.append(str(piece))
    return pieces


def _parse_text_lines(raw: str) -> list[str]:
    """Parse the text-mode two-line response."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Trim potential LLM preamble or trailing commentary.
    lines = lines[:2]
    if len(lines) != 2:
        raise ValueError(f"text mode expects 2 lines, got {len(lines)}")
    return lines


__all__ = ["AlignAgent", "BisectResult"]
