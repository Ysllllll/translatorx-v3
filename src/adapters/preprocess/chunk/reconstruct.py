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


def recover_pair(
    parts: list[str],
    source: str,
    *,
    language: str,
    can_reverse: bool = True,
) -> list[str] | None:
    """Recover a 2-piece split when ``parts`` do not reconstruct ``source``.

    Only handles ``len(parts) == 2``. Returns a new ``[first, second]``
    that does reconstruct the source, or ``None`` if unrecoverable.

    Strategy (mirrors legacy ``check_and_correct_split_sentence``):

    1. Anchor each fragment against the source word pool via
       :func:`_align_to_source`; fragments that don't match become
       empty strings.
    2. If both are empty → give up.
    3. If exactly one matched, derive the other by stripping the
       matched half from the source.
    4. Verify ``join([first, second]) == source`` (ignoring spacing).
       If only the reversed order ``join([second, first])``
       reconstructs, return the reversed order so the result is
       always a valid reconstruction. ``can_reverse`` is retained as a
       legacy parameter; the tail-punctuation swap optimisation it
       used to gate is intentionally omitted because it breaks strict
       reconstruction (and v3's ``Chunker._finalize`` requires that
       invariant).
    5. Final fallback: re-derive each half by subtracting the other
       from the source and re-verify.

    Parameters
    ----------
    parts:
        The 2-piece candidate from the LLM.
    source:
        The original text that was asked to be split.
    language:
        BCP-47 / ISO code.
    can_reverse:
        Legacy parameter retained for API compatibility; currently
        ignored.  See note in step 4.
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
    # Step 3: derive the missing half.
    if first and not second:
        second = re.sub(re.escape(first), "", source)
    elif second and not first:
        first = re.sub(re.escape(second), "", source)

    # Step 4: verify (whitespace-insensitive).
    sent_nospace = _collapse_ws(source)

    def _join_nospace(a: str, b: str) -> str:
        return _collapse_ws(ops.join([a, b]))

    good = False
    if _join_nospace(first, second) == sent_nospace:
        good = True
        reversed_order = False
    elif _join_nospace(second, first) == sent_nospace:
        good = True
        reversed_order = True
    else:
        reversed_order = False
        # Step 5: re-derive each half by subtraction, re-verify.
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

    # When the LLM returned parts in reversed order, simply restore the
    # order that reconstructs the source. The legacy implementation also
    # offered a tail-punctuation swap optimisation when ``can_reverse``
    # was ``True`` — that path deliberately traded strict reconstruction
    # for nicer mid/end punctuation placement, which is incompatible
    # with the v3 ``Chunker._finalize`` invariant that parts must join
    # back to the source.
    if reversed_order:
        first, second = second, first
    _ = can_reverse  # legacy parameter retained for API compatibility

    return [first.strip(), second.strip()]


__all__ = ["chunks_match_source", "recover_pair"]
