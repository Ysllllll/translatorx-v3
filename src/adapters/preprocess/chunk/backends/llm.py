"""LLM chunk backend — registered as ``"llm"``.

Recursive splitter backed by an LLM. Produces a closure that plugs into
the registry dispatch like every other chunk backend.

Per-text flow:

1. If the text is already within ``max_len`` (measured by
   :meth:`LangOps.length`), pass it through untouched.
2. Otherwise prompt the LLM to cut it into ``split_parts`` pieces,
   validate reconstruction via :func:`chunks_match_source`, and retry
   the prompt up to ``max_retries`` times.

   When ``split_parts == 2`` the validator additionally tries a
   deterministic 2-piece recovery via
   :func:`~adapters.preprocess.chunk.reconstruct.recover_pair` before
   rejecting a response — it can salvage LLM outputs where one half is
   correct and the other drifted on spacing/punctuation, or where the
   model returned only one half or swapped the order.
3. Recurse into any produced part still over budget, up to
   ``max_depth`` levels.
4. When the LLM exhausts retries, apply the backend's own
   ``on_failure`` — ``"rule"`` falls back to
   :meth:`LangOps.split_by_length`, ``"keep"`` returns ``[text]``,
   ``"raise"`` re-raises.

A batch is fanned out with ``asyncio.gather`` under a
:class:`asyncio.Semaphore` throttle. The orchestrator
:class:`~adapters.preprocess.chunk.chunker.Chunker` still performs a
final :func:`chunks_match_source` check via ``_finalize`` on top of all
of the above.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Callable, Literal

from domain.lang import LangOps, normalize_language
from ports.retries import resolve_on_failure, retry_until_valid

from adapters.preprocess._common import run_async_in_sync
from adapters.preprocess.chunk.reconstruct import chunks_match_source, recover_pair
from adapters.preprocess.chunk.registry import Backend, ChunkBackendRegistry

if TYPE_CHECKING:
    from domain.lang._core._base_ops import _BaseOps
    from ports.engine import LLMEngine

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_TEMPLATE = (
    "你是一名语言分析专家，擅长根据语法结构将给出的句子分割成{n}部分。\n"
    "必须拆分为{n}部分，同时只需按行输出分割结果，不包含任何解释或额外信息。"
)

_STRIP_LEADING_NUM = re.compile(r"^\d+[.)]\s*")

#: LLM-specific extension of :data:`~ports.retries.OnFailure` — adds
#: ``"rule"`` which falls back to :meth:`LangOps.split_by_length`.
OnFailure = Literal["rule", "keep", "raise"]


_CN_DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _cn_num(n: int) -> str:
    if 0 <= n <= 10:
        return _CN_DIGITS[n]
    return str(n)


@ChunkBackendRegistry.register("llm")
def llm_backend(
    *,
    engine: "LLMEngine",
    language: str,
    max_len: int = 90,
    max_depth: int = 4,
    max_concurrent: int = 8,
    max_retries: int = 2,
    on_failure: OnFailure = "rule",
    split_parts: int = 2,
) -> Backend:
    """Build an LLM recursive-chunking backend.

    The factory returns a synchronous :data:`Backend` that internally
    runs ``asyncio.gather`` with a :class:`asyncio.Semaphore` throttle.

    Parameters
    ----------
    engine:
        An :class:`LLMEngine` (``async .complete(messages)``).
    language:
        BCP-47 / ISO code; drives :class:`LangOps` selection for length
        measurement and the rule-based fallback.
    max_len:
        Maximum per-chunk length (measured by :meth:`LangOps.length`).
        Anything at or below this is passed through untouched. When
        embedded in a :class:`~adapters.preprocess.chunk.chunker.Chunker`
        without an explicit value, the orchestrator's own ``max_len``
        is injected automatically.
    max_depth:
        Upper bound on recursive splits before giving up.
    max_concurrent:
        Upper bound on concurrent LLM calls within one batch.
    max_retries:
        Per-call retries on invalid LLM output; total attempts =
        ``max_retries + 1``.
    on_failure:
        Backend-local policy when a single split fails after all
        retries:

        * ``"rule"`` — fall back to :meth:`LangOps.split_by_length`.
        * ``"keep"`` — return ``[text]`` unchanged.
        * ``"raise"`` — raise :class:`RuntimeError`.
    split_parts:
        Number of pieces requested per split.

    Raises
    ------
    ValueError
        If ``split_parts < 2``, ``max_retries < 0``, ``max_len <= 0``,
        or ``on_failure`` is not one of the accepted values.
    """
    if split_parts < 2:
        raise ValueError("split_parts must be >= 2")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if on_failure not in ("rule", "keep", "raise"):
        raise ValueError(f"invalid on_failure: {on_failure!r}")
    if max_len <= 0:
        raise ValueError("max_len must be > 0")

    ops = LangOps.for_language(normalize_language(language))
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(n=_cn_num(split_parts))

    def _handle_failure(text: str) -> list[str]:
        if on_failure == "rule":
            return ops.split_by_length(text, max_len)
        return resolve_on_failure(
            on_failure,
            keep_value=[text],
            reason=f"LLM chunk failed for text: {text[:80]!r}",
        )

    async def _llm_split(text: str) -> list[str] | None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        async def _call(_attempt: int):
            return await engine.complete(messages)

        def _validate(completion):
            raw = completion.text.strip()
            lines = [_STRIP_LEADING_NUM.sub("", line).strip() for line in raw.splitlines() if line.strip()]
            if len(lines) == split_parts and chunks_match_source(lines, text, language=language, strict=True):
                return True, lines, ""
            # 2-piece recovery: salvage LLM output that is almost correct.
            if split_parts == 2 and 1 <= len(lines) <= 2:
                padded = lines + [""] * (2 - len(lines))
                recovered = recover_pair(padded, text, language=language)
                if recovered is not None and chunks_match_source(recovered, text, language=language, strict=True):
                    return True, recovered, ""
            if len(lines) != split_parts:
                return False, None, f"expected {split_parts} lines, got {len(lines)}"
            return False, None, "reconstruction mismatch"

        def _on_reject(attempt: int, reason: str) -> None:
            logger.debug(
                "LLM chunk rejected (attempt %d/%d, %s) for text: %s",
                attempt + 1,
                max_retries + 1,
                reason,
                text[:80],
            )

        def _on_exception(attempt: int, exc: Exception) -> None:
            logger.debug(
                "LLM chunk call failed (attempt %d/%d, %r) for text: %s",
                attempt + 1,
                max_retries + 1,
                exc,
                text[:60],
            )

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=max_retries,
            on_reject=_on_reject,
            on_exception=_on_exception,
        )
        if not outcome.accepted:
            return None
        return outcome.value

    async def _chunk_recursive(text: str, depth: int, length: Callable[[str], int]) -> list[str]:
        if length(text) <= max_len or depth >= max_depth:
            return [text]
        parts = await _llm_split(text)
        if parts is None:
            parts = _handle_failure(text)
            if parts == [text]:
                return parts
        result: list[str] = []
        for part in parts:
            if length(part) > max_len:
                result.extend(await _chunk_recursive(part, depth + 1, length))
            else:
                result.append(part)
        return result

    async def _process_batch(texts: list[str]) -> list[list[str]]:
        sem = asyncio.Semaphore(max_concurrent)
        length = ops.length

        async def _chunk_one(text: str) -> list[str]:
            if length(text) <= max_len:
                return [text]
            async with sem:
                parts = await _chunk_recursive(text, depth=0, length=length)
            if not chunks_match_source(parts, text, language=language):
                logger.warning(
                    "Chunk result does not reconstruct source, applying on_failure=%s: %r",
                    on_failure,
                    text[:80],
                )
                parts = _handle_failure(text)
            return parts

        results = await asyncio.gather(*(_chunk_one(t) for t in texts), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return list(results)  # type: ignore[arg-type]

    def _backend(texts: list[str]) -> list[list[str]]:
        return run_async_in_sync(lambda: _process_batch(texts))

    return _backend


__all__ = ["llm_backend", "OnFailure"]
