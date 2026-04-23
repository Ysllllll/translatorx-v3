"""Reconstruction validation and 2-piece recovery helpers.

Used by :class:`~adapters.preprocess.chunk.chunker.Chunker` and by the
``llm`` chunk backend to:

* verify a split reconstructs its source text
  (:func:`chunks_match_source`); and
* recover a ``split_parts == 2`` LLM split when one half drifted —
  mirrors the ``check_and_correct_split_sentence`` mechanism from the
  legacy translatorx subtitle handler.

The recovery path is deterministic; it never calls back to the LLM.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.lang import LangOps, normalize_language
from domain.lang._core._punctuation import strip_punct

if TYPE_CHECKING:
    from domain.lang._core._base_ops import _BaseOps


# Whitespace normalizer for "same up to spacing" equality checks.
_WS = re.compile(r"\s+")


def _collapse_ws(s: str) -> str:
    return _WS.sub("", s)


def chunks_match_source(
    parts: list[str],
    source: str,
    *,
    language: str | None = None,
) -> bool:
    """Verify that joining *parts* reconstructs *source*.

    Parameters
    ----------
    parts:
        Candidate chunks produced by a backend.
    source:
        Original text the backend was asked to split.
    language:
        Optional language code. When provided, the per-language
        :meth:`LangOps.join` is tried first, which gives an accurate
        match for CJK and space-delimited languages alike.

    Fallbacks (applied in order) to tolerate common LLM output jitter:

    1. ``" ".join(parts) == source`` — space-delimited languages.
    2. ``"".join(parts) == source`` — CJK / no-space languages.
    3. Alphanumeric-only equality — tolerates whitespace /
       punctuation-spacing drift across the whole string.
    """
    if language is not None:
        ops = LangOps.for_language(normalize_language(language))
        if ops.join(parts) == source:
            return True
    if " ".join(parts) == source:
        return True
    if "".join(parts) == source:
        return True
    src_alnum = "".join(ch for ch in source.lower() if ch.isalnum())
    parts_alnum = "".join(ch for p in parts for ch in p.lower() if ch.isalnum())
    return src_alnum == parts_alnum


# ---------------------------------------------------------------------
# 2-piece recovery (mirrors legacy check_and_correct_split_sentence)
# ---------------------------------------------------------------------


def _align_to_source(fragment: str, pool: list[str], ops: "_BaseOps") -> str:
    """Locate *fragment* inside *pool* by punctuation-stripped match.

    Returns the original pool slice rejoined via ``ops.join`` (so
    original source punctuation is preserved) and **consumes** that
    slice from *pool* in-place to prevent overlapping re-matches on a
    subsequent call.

    Raises :class:`ValueError` if the fragment cannot be anchored.

    Mirrors ``LanguageHandler.correct_sentence_punctuation_by_words_list``
    from the legacy translatorx subtitle module.
    """
    frag_normalised = " ".join(line.strip() for line in fragment.split("\n") if line.strip())
    frag_words = ops.split(frag_normalised, mode="word")
    n = len(frag_words)
    if n == 0:
        raise ValueError("empty fragment")
    if n > len(pool):
        raise ValueError("fragment longer than remaining pool")
    for i in range(len(pool) - n + 1):
        window = pool[i : i + n]
        if all(strip_punct(window[j]) == strip_punct(frag_words[j]) for j in range(n)):
            del pool[i : i + n]
            return ops.join(window)
    raise ValueError(f"fragment not found in pool: {fragment[:60]!r}")


def _swap_trailing_punct(first: str, second: str) -> tuple[str, str]:
    """Swap each piece's trailing punctuation run.

    Given ``("a,", "b.")`` returns ``("a.", "b,")``. Anything without a
    trailing punctuation run is left unchanged. Used by
    :func:`recover_pair` when ``can_reverse=True`` to recover natural
    reading order from an LLM response where texts and their trailing
    punctuation both migrated to the other piece.
    """
    m1 = re.search(r"[^\w\s]+$", first, flags=re.UNICODE)
    m2 = re.search(r"[^\w\s]+$", second, flags=re.UNICODE)
    if not m1 or not m2:
        return first, second
    head1, tail1 = first[: m1.start()], first[m1.start() :]
    head2, tail2 = second[: m2.start()], second[m2.start() :]
    return head1 + tail2, head2 + tail1


def recover_pair(
    parts: list[str],
    source: str,
    *,
    language: str,
    can_reverse: bool = False,
) -> list[str] | None:
    """Recover a 2-piece split when ``parts`` do not reconstruct ``source``.

    Only handles ``len(parts) == 2``. Returns a new ``[first, second]``
    that matches the desired output according to *can_reverse* (see
    below), or ``None`` if the pair is unrecoverable.

    Strategy (mirrors legacy ``check_and_correct_split_sentence``):

    1. Anchor each fragment against the source word pool via
       :func:`_align_to_source`; fragments that don't match become
       empty strings.
    2. If both are empty → give up.
    3. If exactly one matched, derive the other by stripping the
       matched half from the source.
    4. Classify whether ``join([first, second])`` reconstructs the
       source (``reversed_order=False``) or whether only
       ``join([second, first])`` does (``reversed_order=True``).

    ``can_reverse`` controls the treatment of the reversed case:

    * ``can_reverse=False`` (default, strict) — the source order is
       **restored**. Input ``["b.", "a,"]`` for source ``"a, b."``
       recovers to ``["a,", "b."]``. The output still reconstructs the
       source when joined.
    * ``can_reverse=True`` — the reversed order is **accepted** and
       trailing punctuation runs are **swapped** so they match the new
       positional role of each piece. Input ``["b.", "a,"]`` for source
       ``"a, b."`` recovers to ``["b,", "a."]``. The output does **not**
       reconstruct the source by concatenation; callers that need
       strict reconstruction must use ``can_reverse=False``.

    The non-reversed branches (``reversed_order=False``) are identical
    for both settings — ``can_reverse`` only gates the behaviour when
    the pair is classified as reversed.

    Parameters
    ----------
    parts:
        The 2-piece candidate from the LLM.
    source:
        The original text that was asked to be split.
    language:
        BCP-47 / ISO code.
    can_reverse:
        If ``True`` accept the reversed order and swap tail punctuation.
        If ``False`` (default) always restore source order — safe for
        consumers that verify reconstruction (e.g. the chunk pipeline).
    """
    if len(parts) != 2:
        return None

    ops = LangOps.for_language(normalize_language(language))
    pool = ops.split(source, mode="word")

    fixed: list[str] = []
    for frag in parts:
        try:
            fixed.append(_align_to_source(frag, pool, ops))
        except ValueError:
            fixed.append("")

    if not any(fixed):
        return None

    first, second = fixed
    if first and not second:
        second = re.sub(re.escape(first), "", source)
    elif second and not first:
        first = re.sub(re.escape(second), "", source)

    sent_nospace = _collapse_ws(source)

    def _join_nospace(a: str, b: str) -> str:
        return _collapse_ws(ops.join([a, b]))

    good = False
    reversed_order = False
    if _join_nospace(first, second) == sent_nospace:
        good = True
    elif _join_nospace(second, first) == sent_nospace:
        good = True
        reversed_order = True
    else:
        if first:
            candidate_second = re.sub(re.escape(first), "", source)
            if _join_nospace(first, candidate_second) == sent_nospace:
                second = candidate_second
                good = True
        if not good and second:
            candidate_first = re.sub(re.escape(second), "", source)
            if _join_nospace(candidate_first, second) == sent_nospace:
                first = candidate_first
                good = True

    if not good:
        return None

    if reversed_order:
        if can_reverse:
            # Keep the reversed order; swap the trailing punctuation runs
            # so mid/end punctuation matches the new positional role.
            first, second = _swap_trailing_punct(first, second)
        else:
            # Restore original source order; no tail-punct rearrangement.
            first, second = second, first

    return [first.strip(), second.strip()]


__all__ = ["chunks_match_source", "recover_pair"]
