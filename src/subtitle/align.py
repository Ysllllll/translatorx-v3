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

from .model import Segment, Word

# Punctuation constants — single source of truth in _punctuation.py.
from lang_ops._core._punctuation import (
    ALL_PUNCT as _PUNCT,
    OPENING_PUNCT as _OPENING_PUNCT,
    TRAILING_PUNCT as _TRAILING,
    CLOSING_PUNCT as _CLOSING,
    DASHES as _DASHES,
    strip_punct as _strip_punct,
)

# Trailing / closing punctuation — attaches to the *previous* word.
_CLOSING_PUNCT = _TRAILING | _CLOSING | _DASHES


def _is_punct_only(s: str) -> bool:
    """Return True if the string consists entirely of punctuation."""
    return bool(s) and all(ch in _PUNCT for ch in s)


def _is_opening_punct_word(s: str) -> bool:
    """Return True if the string consists entirely of opening punctuation."""
    return bool(s) and all(ch in _OPENING_PUNCT for ch in s)


def _is_closing_punct_word(s: str) -> bool:
    """Return True if the string consists entirely of closing/trailing punctuation."""
    return bool(s) and all(ch in _CLOSING_PUNCT for ch in s)


# ------------------------------------------------------------------
# attach_punct_words
# ------------------------------------------------------------------

def attach_punct_words(words: list[Word]) -> list[Word]:
    """Merge standalone punctuation Words into adjacent text Words.

    Closing/trailing punctuation (e.g. ``","`` ``"."`` ``")"`` ``"。"``)
    attaches to the **previous** word.  Opening punctuation (e.g.
    ``"("`` ``"「"``) attaches to the **next** word.  Time ranges are
    extended to cover the merged punctuation.

    Words that already contain text (not pure punctuation) are left
    unchanged.  If the entire list is punctuation, it is returned as-is.

    Args:
        words: Word list, possibly with standalone punctuation Words.

    Returns:
        New list with punctuation merged into adjacent words.
        Returns the original list unchanged if no merging was needed.
    """
    if not words:
        return words

    # Fast path: if no standalone punctuation words, return as-is
    if not any(_is_punct_only(w.word.strip()) for w in words):
        return words

    # Phase 1: attach closing/trailing punct to previous word
    merged: list[Word] = []
    for w in words:
        text = w.word.strip()
        if merged and _is_closing_punct_word(text):
            prev = merged[-1]
            merged[-1] = replace(
                prev,
                word=prev.word + w.word.lstrip(),
                end=max(prev.end, w.end),
            )
        else:
            merged.append(w)

    # Phase 2: attach opening punct to next word (left-to-right)
    result: list[Word] = []
    pending_open: Word | None = None
    for w in merged:
        text = w.word.strip()
        if _is_opening_punct_word(text):
            if pending_open is not None:
                # Chain of opening puncts — merge them
                pending_open = replace(
                    pending_open,
                    word=pending_open.word + w.word.lstrip(),
                    end=max(pending_open.end, w.end),
                )
            else:
                pending_open = w
        elif pending_open is not None:
            # Attach pending opening punct to this word
            combined = pending_open.word + w.word.lstrip()
            result.append(replace(
                w,
                word=combined,
                start=min(pending_open.start, w.start),
            ))
            pending_open = None
        else:
            result.append(w)

    if pending_open is not None:
        result.append(pending_open)

    return result


# ------------------------------------------------------------------
# fill_words
# ------------------------------------------------------------------

def fill_words(
    segment: Segment,
    split_fn: Callable[[str], list[str]] | None = None,
) -> Segment:
    """Ensure *segment* has consistent text and word-level timing.

    Delegates to :func:`normalize_words` which handles three cases:

    - **Words present**: attaches standalone punctuation words to neighbors.
    - **No words**: synthesizes words from text with proportional timing.
    - **No text**: derives text by joining words.

    Args:
        segment: Input segment.
        split_fn: Optional tokenizer ``(text) -> list[str]``.
                  Defaults to ``str.split()`` (whitespace).
                  For CJK, pass ``ops.split`` to get proper tokenization.

    Returns:
        Segment with ``words`` populated (punctuation attached) and ``text``
        guaranteed non-empty when the segment has content.
    """
    text, words = normalize_words(
        segment.text, segment.words, split_fn,
        start=segment.start, end=segment.end,
    )
    if text == segment.text and words is segment.words:
        return segment
    return replace(segment, text=text, words=words)


# ------------------------------------------------------------------
# find_words
# ------------------------------------------------------------------

def find_words(
    words: list[Word],
    sub_text: str,
    start: int = 0,
) -> tuple[int, int]:
    """Find the contiguous slice of *words* that covers *sub_text*.

    Uses multi-level tolerant matching via ``Word.content``
    (punctuation-stripped form, computed once at construction):

        1. Exact match on ``word.word``
        2. Match on ``word.content`` (punct/whitespace stripped)
        3. Case-insensitive match on ``word.content``

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
        w = words[i]
        if not w.content:
            # Pure-punctuation / whitespace word — absorb if match started
            if first is not None:
                last = i + 1
            continue

        # Level 1: exact match
        idx = sub_text.find(w.word, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w.word)
            pos_lower = pos
            continue

        # Level 2: content-only match (punctuation + whitespace stripped)
        idx = sub_text.find(w.content, pos)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(w.content)
            pos_lower = pos
            continue

        # Level 3: case-insensitive content match
        c_lower = w.content.lower()
        idx = sub_lower.find(c_lower, pos_lower)
        if idx >= 0:
            if first is None:
                first = i
            last = i + 1
            pos = idx + len(c_lower)
            pos_lower = pos
            continue

        # Word doesn't appear in remaining sub_text → stop
        break

    if first is None:
        return (start, start)
    return (first, last)


# ------------------------------------------------------------------
# normalize_words
# ------------------------------------------------------------------

def normalize_words(
    text: str | None,
    words: list[Word],
    split_fn: Callable[[str], list[str]] | None = None,
    start: float = 0.0,
    end: float = 0.0,
) -> tuple[str, list[Word]]:
    """Reconcile *text* and *words* into a consistent pair.

    Handles three real-world scenarios:

    1. **Only text** (``words`` empty): split text into tokens and create
       evenly-spaced Word objects spanning ``[start, end]``.
    2. **Only words** (``text`` empty/None): derive text by joining words.
    3. **Both present but inconsistent** (post-correction): keep the new
       *text* as-is and attach punctuation on *words*.

    Args:
        text: Segment text (may be ``None`` or empty).
        words: Word list (may be empty).
        split_fn: Tokenizer ``(text) -> list[str]``.  Defaults to ``str.split()``.
        start: Segment start time (used when generating words from text).
        end: Segment end time.

    Returns:
        ``(text, words)`` — both guaranteed non-empty when input has content.
    """
    has_text = bool(text and text.strip())
    has_words = bool(words)

    if has_text and has_words:
        # Case 3: both present — just attach punctuation
        return text, attach_punct_words(words)  # type: ignore[return-value]

    if has_text:
        # Case 1: only text — synthesize words
        assert text is not None
        tokens = split_fn(text) if split_fn else text.split()
        if not tokens:
            return text, []
        duration = end - start
        total_len = sum(len(t) for t in tokens) or 1
        new_words: list[Word] = []
        t = start
        for token in tokens:
            w_dur = duration * len(token) / total_len
            new_words.append(Word(word=token, start=t, end=t + w_dur))
            t += w_dur
        return text, attach_punct_words(new_words)

    if has_words:
        # Case 2: only words — derive text
        derived = "".join(w.word for w in words)
        return derived, attach_punct_words(words)

    # Neither text nor words
    return text or "", []


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
