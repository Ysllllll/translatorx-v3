"""LLM-based chunker using recursive binary splitting.

Ported from the old system's ``ChunkAgent``: an LLM is asked to split a
sentence into N grammatically coherent parts (default 2).  The process
recurses (up to *max_depth* times) until every chunk fits within *chunk_len*.
On repeated LLM failure, the configured *on_failure* policy decides the
fallback (rule-based split, keep as-is, or raise).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from domain.lang._core._base_ops import _BaseOps
    from application.translate import LLMEngine

from ports.retries import resolve_on_failure, retry_until_valid

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = (
    "你是一名语言分析专家，擅长根据语法结构将给出的句子分割成{n}部分。\n"
    "必须拆分为{n}部分，同时只需按行输出分割结果，不包含任何解释或额外信息。"
)

# Matches leading punctuation in LLM output lines.
_STRIP_LEADING_NUM = re.compile(r"^\d+[.)]\s*")

#: Chunker extends the shared :data:`llm_ops.retries.OnFailure` vocabulary
#: with a domain-specific ``"rule"`` option that falls back to a
#: deterministic length-based splitter.
OnFailure = Literal["rule", "keep", "raise"]


def _chunks_match_source(parts: list[str], source: str) -> bool:
    """Verify that joining chunks reconstructs the source text.

    Tries space-join (for space-delimited languages) and no-space-join
    (for CJK), then falls back to alphanumeric-only comparison.
    """
    if " ".join(parts) == source:
        return True
    if "".join(parts) == source:
        return True
    # Alphanumeric fallback (handles minor whitespace differences)
    src_alnum = "".join(ch for ch in source.lower() if ch.isalnum())
    parts_alnum = "".join(ch for p in parts for ch in p.lower() if ch.isalnum())
    return src_alnum == parts_alnum


class LlmChunker:
    """Recursive LLM chunker conforming to :data:`ApplyFn`.

    Parameters
    ----------
    engine:
        LLM engine to use for chunking (can differ from the translation engine).
    chunk_len:
        Maximum length per chunk (default 90, matching old TINYS_CHUNK). Compared
        against ``length_fn(text)`` if provided, else ``ops.length(text)`` if
        *ops* is given, else ``len(text)``.
    max_depth:
        Maximum recursion depth (default 4).
    ops:
        Optional language ops. When provided, ``ops.length`` becomes the default
        length metric (correct for CJK/mixed text) and ``ops.split_by_length``
        powers the rule-based fallback.
    length_fn:
        Optional custom length callable — overrides ``ops.length`` / ``len`` for
        every length measurement. Use when the default metric doesn't match your
        budget semantics (e.g. display-width budgeting)::

            chunker = LlmChunker(
                engine,
                ops=ops,
                length_fn=lambda t: ops.length(t, cjk_width=2),
            )
    max_concurrent:
        Maximum concurrent LLM calls within a batch (default 8).
    max_retries:
        How many times to retry the LLM per split attempt before giving up on
        this node (default 2). The initial call counts as attempt #1, so
        ``max_retries=2`` means at most 3 total calls.
    on_failure:
        What to do when the LLM ultimately fails at a given node:

        - ``"rule"`` (default): fall back to rule-based split.
        - ``"keep"``: stop recursing and return the text unchanged.
        - ``"raise"``: raise :class:`RuntimeError`.
    split_parts:
        How many parts the LLM is asked to produce per call (default 2 —
        classic binary recursion). Any ``N >= 2`` is allowed.
    """

    def __init__(
        self,
        engine: "LLMEngine",
        *,
        chunk_len: int = 90,
        max_depth: int = 4,
        ops: "_BaseOps | None" = None,
        max_concurrent: int = 8,
        max_retries: int = 2,
        on_failure: OnFailure = "rule",
        split_parts: int = 2,
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        if split_parts < 2:
            raise ValueError("split_parts must be >= 2")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if on_failure not in ("rule", "keep", "raise"):
            raise ValueError(f"invalid on_failure: {on_failure!r}")

        self._engine = engine
        self._chunk_len = chunk_len
        self._max_depth = max_depth
        self._ops = ops
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._on_failure = on_failure
        self._split_parts = split_parts

        if length_fn is not None:
            self._length: Callable[[str], int] = length_fn
        elif ops is not None:
            self._length = ops.length
        else:
            self._length = len
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(n=_cn_num(split_parts))

    # -- helpers ----------------------------------------------------------

    # -- ApplyFn interface ------------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Synchronous ApplyFn — runs the async pipeline internally.

        Two separate concurrency mechanisms are at play and they solve
        *different* problems:

        * ``ThreadPoolExecutor`` only exists to escape a caller that already
          owns a running event loop (e.g. inside a Jupyter cell or another
          async framework). ``asyncio.run`` cannot be nested, so we hand off
          to a worker thread which owns its own loop.
        * ``asyncio.Semaphore`` inside :meth:`_process_batch` throttles how
          many LLM calls run in parallel *within* that loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self._process_batch(texts))).result()
        return asyncio.run(self._process_batch(texts))

    async def _process_batch(self, texts: list[str]) -> list[list[str]]:
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _chunk_one(text: str) -> list[str]:
            if self._length(text) <= self._chunk_len:
                return [text]
            async with sem:
                parts = await self._chunk_recursive(text, depth=0)
            if not _chunks_match_source(parts, text):
                logger.warning(
                    "Chunk result does not reconstruct source, applying on_failure=%s: %r",
                    self._on_failure,
                    text[:80],
                )
                parts = self._handle_failure(text)
            return parts

        return list(await asyncio.gather(*(_chunk_one(t) for t in texts)))

    async def _chunk_recursive(self, text: str, depth: int) -> list[str]:
        """Recursively split *text* into chunks ≤ chunk_len."""
        if self._length(text) <= self._chunk_len or depth >= self._max_depth:
            return [text]

        parts = await self._llm_split(text)
        if parts is None:
            parts = self._handle_failure(text)
            # If on_failure=="keep", _handle_failure returns [text]; no further recursion.
            if parts == [text]:
                return parts

        result: list[str] = []
        for part in parts:
            if self._length(part) > self._chunk_len:
                result.extend(await self._chunk_recursive(part, depth + 1))
            else:
                result.append(part)
        return result

    def _handle_failure(self, text: str) -> list[str]:
        """Resolve an unrecoverable LLM failure according to *on_failure*."""
        if self._on_failure == "rule":
            return self._rule_split(text)
        # "keep" / "raise" → delegate to shared helper.
        return resolve_on_failure(
            self._on_failure,
            keep_value=[text],
            reason=f"LLM chunk failed for text: {text[:80]!r}",
        )

    async def _llm_split(self, text: str) -> list[str] | None:
        """Ask the LLM to split *text* into N parts. Returns None after exhausting retries."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]

        async def _call(_attempt: int):
            return await self._engine.complete(messages)

        def _validate(completion):
            raw = completion.text.strip()
            lines = [_STRIP_LEADING_NUM.sub("", line).strip() for line in raw.splitlines() if line.strip()]
            if len(lines) != self._split_parts:
                return False, None, f"expected {self._split_parts} lines, got {len(lines)}"
            if not _chunks_match_source(lines, text):
                return False, None, "reconstruction mismatch"
            return True, lines, ""

        def _on_reject(attempt: int, reason: str) -> None:
            logger.debug(
                "LLM chunk rejected (attempt %d/%d, %s) for text: %s",
                attempt + 1,
                self._max_retries + 1,
                reason,
                text[:80],
            )

        def _on_exception(attempt: int, exc: Exception) -> None:
            logger.debug(
                "LLM chunk call failed (attempt %d/%d, %r) for text: %s",
                attempt + 1,
                self._max_retries + 1,
                exc,
                text[:60],
            )

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=self._max_retries,
            on_reject=_on_reject,
            on_exception=_on_exception,
        )
        if not outcome.accepted:
            logger.debug(
                "LLM chunk giving up after %d attempts (%s)",
                outcome.attempts,
                outcome.last_reason,
            )
            return None
        return outcome.value

    def _rule_split(self, text: str) -> list[str]:
        """Fall back to rule-based split at word boundaries."""
        if self._ops is not None:
            return self._ops.split_by_length(text, self._chunk_len)
        # Simple fallback: split at midpoint word boundary.
        mid = len(text) // 2
        # Find nearest space.
        left = text.rfind(" ", 0, mid + 20)
        if left == -1:
            left = mid
        return [text[:left].strip(), text[left:].strip()]


_CN_DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _cn_num(n: int) -> str:
    """Render small integers in Chinese (for the prompt template)."""
    if 0 <= n <= 10:
        return _CN_DIGITS[n]
    return str(n)


__all__ = ["LlmChunker", "OnFailure"]
