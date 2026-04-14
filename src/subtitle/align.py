"""Word-level timing utilities for subtitle segments.

Workflow::

    seg = fill_words(segment)                          # ensure words exist
    groups = distribute_words(seg.words, sentences)    # match text → words
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .model import Segment, Word

from lang_ops._core._punctuation import (
    OPENING_PUNCT as _OPENING,
    TRAILING_PUNCT as _TRAILING,
    CLOSING_PUNCT as _CLOSING,
    DASHES as _DASHES,
)

# Closing/trailing/dash — attaches to the *previous* word.
_CLOSE = _TRAILING | _CLOSING | _DASHES


def _all_in(s: str, charset: frozenset[str]) -> bool:
    return bool(s) and all(ch in charset for ch in s)


# ------------------------------------------------------------------
# attach_punct_words
# ------------------------------------------------------------------

def attach_punct_words(words: list[Word]) -> list[Word]:
    """Merge standalone punctuation Words into adjacent text Words.

    Closing punct → previous word, opening punct → next word.
    Time ranges are extended to cover merged punctuation.
    Returns original list if no merging needed.
    """
    if not words or not any(not w.content for w in words):
        return words

    # Phase 1: closing/trailing punct → previous word
    merged: list[Word] = []
    for w in words:
        if merged and not w.content and _all_in(w.word.strip(), _CLOSE):
            p = merged[-1]
            merged[-1] = replace(p, word=p.word + w.word.lstrip(),
                                 end=max(p.end, w.end))
        else:
            merged.append(w)

    # Phase 2: opening punct → next word
    result: list[Word] = []
    pending: Word | None = None
    for w in merged:
        if not w.content and _all_in(w.word.strip(), _OPENING):
            if pending:
                pending = replace(pending,
                                  word=pending.word + w.word.lstrip(),
                                  end=max(pending.end, w.end))
            else:
                pending = w
        elif pending:
            result.append(replace(w,
                                  word=pending.word + w.word.lstrip(),
                                  start=min(pending.start, w.start)))
            pending = None
        else:
            result.append(w)

    if pending:
        result.append(pending)
    return result


# ------------------------------------------------------------------
# normalize_words / fill_words
# ------------------------------------------------------------------

def _synthesize_words(
    tokens: list[str], start: float, end: float,
) -> list[Word]:
    """Create evenly-spaced Words from text tokens."""
    duration = end - start
    total = sum(len(t) for t in tokens) or 1
    words: list[Word] = []
    t = start
    for tok in tokens:
        d = duration * len(tok) / total
        words.append(Word(word=tok, start=t, end=t + d))
        t += d
    return words


def normalize_words(
    text: str | None,
    words: list[Word],
    split_fn: Callable[[str], list[str]] | None = None,
    start: float = 0.0,
    end: float = 0.0,
) -> tuple[str, list[Word]]:
    """Reconcile *text* and *words* into a consistent ``(text, words)`` pair.

    Three cases:
    1. Only text → synthesize words with proportional timing.
    2. Only words → derive text by joining ``w.word``.
    3. Both → keep text, attach punctuation on words.
    """
    has_text = bool(text and text.strip())
    has_words = bool(words)

    if has_text and has_words:
        return text, attach_punct_words(words)  # type: ignore[return-value]
    if has_text:
        assert text is not None
        tokens = split_fn(text) if split_fn else text.split()
        if not tokens:
            return text, []
        return text, attach_punct_words(_synthesize_words(tokens, start, end))
    if has_words:
        return "".join(w.word for w in words), attach_punct_words(words)
    return text or "", []


def fill_words(
    segment: Segment,
    split_fn: Callable[[str], list[str]] | None = None,
) -> Segment:
    """Ensure *segment* has consistent text and word-level timing.

    Thin wrapper around :func:`normalize_words` — handles the three
    cases (only-text, only-words, both) and returns a new Segment.
    """
    text, words = normalize_words(
        segment.text, segment.words, split_fn,
        start=segment.start, end=segment.end,
    )
    if text == segment.text and words is segment.words:
        return segment
    return replace(segment, text=text, words=words)


# ------------------------------------------------------------------
# find_words / distribute_words / align_segments
# ------------------------------------------------------------------

def find_words(
    words: list[Word],
    sub_text: str,
    start: int = 0,
) -> tuple[int, int]:
    """Find the contiguous slice of *words* covering *sub_text*.

    Three-level tolerant matching (exact → content → case-insensitive).
    Returns ``(start_idx, end_idx)`` or ``(start, start)`` if not found.
    """
    if not sub_text or not sub_text.strip() or start >= len(words):
        return (start, start)

    sub_lower = sub_text.lower()
    first: int | None = None
    last = start
    pos = 0
    pos_low = 0

    for i in range(start, len(words)):
        w = words[i]
        if not w.content:
            if first is not None:
                last = i + 1
            continue

        # Three-level tolerant matching: exact → content → case-insensitive
        idx = sub_text.find(w.word, pos)
        if idx >= 0:
            needle_len = len(w.word)
        else:
            idx = sub_text.find(w.content, pos)
            if idx >= 0:
                needle_len = len(w.content)
            else:
                idx = sub_lower.find(w.content.lower(), pos_low)
                if idx >= 0:
                    needle_len = len(w.content)
                else:
                    break  # no match at any level → stop

        if first is None:
            first = i
        last = i + 1
        pos = pos_low = idx + needle_len

    return (first, last) if first is not None else (start, start)


def distribute_words(
    words: list[Word], texts: list[str],
) -> list[list[Word]]:
    """Assign *words* to consecutive *texts* via :func:`find_words`."""
    result: list[list[Word]] = []
    idx = 0
    for text in texts:
        s, e = find_words(words, text, start=idx)
        result.append(list(words[s:e]))
        idx = e
    return result


def align_segments(
    chunks: list[str], words: list[Word],
) -> list[Segment]:
    """Align text chunks with timed words → list of Segments."""
    groups = distribute_words(words, chunks)
    segs: list[Segment] = []
    for text, grp in zip(chunks, groups):
        s = grp[0].start if grp else 0.0
        e = grp[-1].end if grp else 0.0
        segs.append(Segment(start=s, end=e, text=text, words=grp))
    return segs
