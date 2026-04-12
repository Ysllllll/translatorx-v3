"""Word-level timing utilities for subtitle segments.

Two-step workflow:
    1. ``fill_words`` — ensure every Segment has word-level timing
    2. ``find_words`` / ``distribute_words`` — match text pieces back to words

Example::

    seg = fill_words(segment)
    sentences = ops.split_sentences(seg.text)
    for words in distribute_words(seg.words, sentences):
        start_time = words[0].start
        end_time = words[-1].end
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ._types import Segment, Word

# Punctuation characters for tolerant matching.
_PUNCT = frozenset(
    ".,!?;:，。！？；：、"
    "\"\"\"'''"
    "()[]{}（）《》「」『』【】〈〉"
    "—–‐-…·"
    "/\\@#$%^&*+=|~"
)


def _strip_punct(s: str) -> str:
    """Strip leading and trailing punctuation."""
    i = 0
    while i < len(s) and s[i] in _PUNCT:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in _PUNCT:
        j -= 1
    return s[i:j]


# ------------------------------------------------------------------
# fill_words
# ------------------------------------------------------------------

def fill_words(
    segment: Segment,
    split_fn: Callable[[str], list[str]] | None = None,
) -> Segment:
    """Ensure *segment* has word-level timing.

    If ``segment.words`` is already populated, returns the segment unchanged.
    Otherwise, splits the text into tokens (via *split_fn* or ``str.split``)
    and distributes the segment's time range proportionally by token length.

    Args:
        segment: Input segment.
        split_fn: Optional tokenizer ``(text) -> list[str]``.
                  Defaults to ``str.split()`` (whitespace).
                  For CJK, pass ``ops.split`` to get proper tokenization.

    Returns:
        Segment with ``words`` populated.
    """
    if segment.words:
        return segment

    tokens = split_fn(segment.text) if split_fn else segment.text.split()
    if not tokens:
        return segment

    duration = segment.end - segment.start
    total_len = sum(len(t) for t in tokens) or 1
    words: list[Word] = []
    t = segment.start

    for token in tokens:
        w_dur = duration * len(token) / total_len
        words.append(Word(word=token, start=t, end=t + w_dur))
        t += w_dur

    return replace(segment, words=words)


# ------------------------------------------------------------------
# find_words
# ------------------------------------------------------------------

def find_words(
    words: list[Word],
    sub_text: str,
    start: int = 0,
) -> tuple[int, int]:
    """Find the contiguous slice of *words* that covers *sub_text*.

    Uses punctuation-tolerant matching: strips punctuation from both
    word tokens and the search text when exact matching fails.

    Args:
        words: Full word list (e.g. ``segment.words``).
        sub_text: A substring of the original text (e.g. one sentence).
        start: Index in *words* to begin searching from.

    Returns:
        ``(start_idx, end_idx)`` such that ``words[start_idx:end_idx]``
        covers *sub_text*.  If no match is found, returns ``(start, start)``.
    """
    if not sub_text or not sub_text.strip() or start >= len(words):
        return (start, start)

    first: int | None = None
    last = start
    pos = 0  # scan position in sub_text

    for i in range(start, len(words)):
        w_content = _strip_punct(words[i].word)
        if not w_content:
            # Pure-punctuation word — absorb into current match if started
            if first is not None:
                last = i + 1
            continue

        # Try exact word match first
        idx = sub_text.find(words[i].word, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(words[i].word)
            continue

        # Fallback: content-only match (punctuation stripped)
        idx = sub_text.find(w_content, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w_content)
            continue

        # Word doesn't appear in remaining sub_text → stop
        break

    if first is None:
        return (start, start)
    return (first, last)


# ------------------------------------------------------------------
# distribute_words
# ------------------------------------------------------------------

def distribute_words(
    words: list[Word],
    texts: list[str],
) -> list[list[Word]]:
    """Assign *words* to consecutive *texts* pieces.

    Calls :func:`find_words` sequentially, advancing through the word list.

    Args:
        words: Full word list from a segment.
        texts: Consecutive text pieces (e.g. from ``split_sentences``).

    Returns:
        A list of word slices, one per text piece.

    Example::

        sentences = ops.split_sentences(segment.text)
        groups = distribute_words(segment.words, sentences)
        for group in groups:
            if group:
                print(group[0].start, group[-1].end)
    """
    result: list[list[Word]] = []
    idx = 0
    for text in texts:
        start_i, end_i = find_words(words, text, start=idx)
        result.append(list(words[start_i:end_i]))
        idx = end_i
    return result
