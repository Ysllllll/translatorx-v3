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

def _normalize_word(w: str) -> str:
    """Strip punctuation and whitespace for tolerant matching."""
    return _strip_punct(w.strip())


def find_words(
    words: list[Word],
    sub_text: str,
    start: int = 0,
) -> tuple[int, int]:
    """Find the contiguous slice of *words* that covers *sub_text*.

    Uses multi-level tolerant matching:
        1. Exact match
        2. Punctuation-stripped match
        3. Case-insensitive + punctuation/whitespace-stripped match

    This handles real-world word lists where tokens may have leading
    spaces (Whisper-style), different casing, or missing punctuation.

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

    sub_lower = sub_text.lower()
    first: int | None = None
    last = start
    pos = 0       # scan position in sub_text
    pos_lower = 0  # scan position in sub_lower (for case-insensitive)

    for i in range(start, len(words)):
        w_raw = words[i].word
        w_content = _normalize_word(w_raw)
        if not w_content:
            # Pure-punctuation / whitespace word — absorb if match started
            if first is not None:
                last = i + 1
            continue

        # Level 1: exact match
        idx = sub_text.find(w_raw, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w_raw)
            pos_lower = pos
            continue

        # Level 2: content-only match (punctuation + whitespace stripped)
        idx = sub_text.find(w_content, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w_content)
            pos_lower = pos
            continue

        # Level 3: case-insensitive content match
        w_lower = w_content.lower()
        idx = sub_lower.find(w_lower, pos_lower)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w_lower)
            pos_lower = pos
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


# ------------------------------------------------------------------
# align_segments
# ------------------------------------------------------------------

def align_segments(
    chunks: list[str],
    words: list[Word],
) -> list[Segment]:
    """Align text chunks with timed words to produce :class:`Segment` objects.

    Typical usage with :class:`~lang_ops.splitter.ChunkPipeline`::

        spans = ops.chunk(seg.text).sentences().result()
        segments = align_segments(spans, seg.words)

    Each returned Segment carries the text of one chunk, the subset of
    *words* that fall within it, and timing derived from those words.

    Args:
        chunks: Consecutive text pieces (e.g. from a pipeline ``.result()``).
        words: Full word list with timing (e.g. ``segment.words``).

    Returns:
        A list of :class:`Segment`, one per chunk.  If a chunk has no
        matching words its ``start`` and ``end`` are both ``0.0``.
    """
    groups = distribute_words(words, chunks)
    segments: list[Segment] = []
    for text, group in zip(chunks, groups):
        if group:
            start = group[0].start
            end = group[-1].end
        else:
            start = end = 0.0
        segments.append(Segment(start=start, end=end, text=text, words=group))
    return segments
