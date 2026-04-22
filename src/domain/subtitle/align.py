"""Word-level timing utilities for subtitle segments.

Workflow::

    seg = fill_words(segment)                          # ensure words exist
    groups = distribute_words(seg.words, sentences)    # match text → words
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from domain.model import Segment, Word

from domain.lang._core._punctuation import (
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
            merged[-1] = replace(p, word=p.word + w.word.lstrip(), end=max(p.end, w.end))
        else:
            merged.append(w)

    # Phase 2: opening punct → next word
    result: list[Word] = []
    pending: Word | None = None
    for w in merged:
        if not w.content and _all_in(w.word.strip(), _OPENING):
            if pending:
                pending = replace(
                    pending,
                    word=pending.word + w.word.lstrip(),
                    end=max(pending.end, w.end),
                )
            else:
                pending = w
        elif pending:
            result.append(
                replace(
                    w,
                    word=pending.word + w.word.lstrip(),
                    start=min(pending.start, w.start),
                )
            )
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
    tokens: list[str],
    start: float,
    end: float,
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
        segment.text,
        segment.words,
        split_fn,
        start=segment.start,
        end=segment.end,
    )
    if text == segment.text and words is segment.words:
        return segment
    return replace(segment, text=text, words=words)


# ------------------------------------------------------------------
# find_words / distribute_words / align_segments
# ------------------------------------------------------------------


def _find_word_boundary(
    text: str,
    needle: str,
    start: int,
) -> int:
    """Find *needle* in *text* at a word boundary.

    A match is valid only when the character before it (if any) and the
    character after it (if any) are NOT alphanumeric.  This prevents
    matching ``"you"`` inside ``"your"``.

    Falls back to ordinary ``str.find`` if no boundary-respecting match
    exists (handles edge cases where the word really is embedded, e.g.
    CJK characters).
    """
    pos = start
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            return -1
        # Left boundary: start of string or non-alnum before match
        left_ok = idx == 0 or not text[idx - 1].isalnum()
        # Right boundary: end of string or non-alnum after match
        end = idx + len(needle)
        right_ok = end >= len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            return idx
        pos = idx + 1


def find_words(
    words: list[Word],
    sub_text: str,
    start: int = 0,
) -> tuple[int, int]:
    """Find the contiguous slice of *words* covering *sub_text*.

    Four-level tolerant matching:
      1. exact word → 2. stripped content → 3. case-insensitive →
      4. alphanumeric-only (handles punctuation changes from punc
         restoration, e.g. ``dont`` matching ``don't``).

    Levels 1–3 require word-boundary matches to avoid matching short
    words inside longer ones (e.g. ``"you"`` inside ``"your"``).

    If alphanumeric matching also fails, the text and words are
    fundamentally misaligned — break immediately with no tolerance.

    Returns ``(start_idx, end_idx)`` or ``(start, start)`` if not found.
    """
    if not sub_text or not sub_text.strip() or start >= len(words):
        return (start, start)

    sub_lower = sub_text.lower()

    # Pre-compute alphanumeric mapping for level 4:
    # alnum_chars[i] → the i-th alnum character (lowered)
    # alnum_to_orig[i] → its position in the original sub_text
    alnum_chars: list[str] = []
    alnum_to_orig: list[int] = []
    for orig_i, ch in enumerate(sub_text):
        if ch.isalnum():
            alnum_chars.append(ch.lower())
            alnum_to_orig.append(orig_i)
    sub_alnum = "".join(alnum_chars)

    first: int | None = None
    last = start
    pos = 0
    pos_low = 0
    pos_alnum = 0

    for i in range(start, len(words)):
        w = words[i]
        if not w.content:
            if first is not None:
                last = i + 1
            continue

        # Level 1: exact word match (word-boundary)
        idx = _find_word_boundary(sub_text, w.word, pos)
        if idx >= 0:
            pos = pos_low = idx + len(w.word)
        else:
            # Level 2: content (punctuation-stripped word, word-boundary)
            idx = _find_word_boundary(sub_text, w.content, pos)
            if idx >= 0:
                pos = pos_low = idx + len(w.content)
            else:
                # Level 3: case-insensitive content (word-boundary)
                idx = _find_word_boundary(sub_lower, w.content.lower(), pos_low)
                if idx >= 0:
                    pos = pos_low = idx + len(w.content)
                else:
                    # Level 4: alphanumeric-only — strip all non-alnum
                    # from both word and text, then match.
                    w_alnum = "".join(ch for ch in w.content.lower() if ch.isalnum())
                    if not w_alnum:
                        break
                    idx_a = sub_alnum.find(w_alnum, pos_alnum)
                    if idx_a < 0:
                        break  # text and words are misaligned
                    # Map alnum position back to original text position
                    end_alnum = idx_a + len(w_alnum)
                    orig_end = alnum_to_orig[end_alnum - 1] + 1
                    pos = pos_low = orig_end
                    pos_alnum = end_alnum
                    if first is None:
                        first = i
                    last = i + 1
                    continue

        # Advance alnum position to stay in sync
        while pos_alnum < len(alnum_to_orig) and alnum_to_orig[pos_alnum] < pos:
            pos_alnum += 1

        if first is None:
            first = i
        last = i + 1

    return (first, last) if first is not None else (start, start)


def distribute_words(
    words: list[Word],
    texts: list[str],
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
    chunks: list[str],
    words: list[Word],
) -> list[Segment]:
    """Align text chunks with timed words → list of Segments."""
    groups = distribute_words(words, chunks)
    segs: list[Segment] = []
    for text, grp in zip(chunks, groups):
        s = grp[0].start if grp else 0.0
        e = grp[-1].end if grp else 0.0
        segs.append(Segment(start=s, end=e, text=text, words=grp))
    return segs


def rebalance_segment_words(
    seg_a: Segment,
    seg_b: Segment,
    target_ratio: float,
    max_chunk_len: int,
    *,
    ops,
) -> tuple[Segment, Segment]:
    """Redistribute the combined words of ``seg_a + seg_b`` to match *target_ratio*.

    Ports the legacy ``AlignAgent.process_elements``: merges the two segments'
    words, then picks the word-boundary ``i`` that minimizes
    ``|target_ratio - length(left_text) / length(right_text)|`` subject to
    ``length(left) <= max_chunk_len`` and ``length(right) <= max_chunk_len``.

    Returns two new :class:`Segment`\\s whose ``text`` is rebuilt from the
    rebalanced word content and whose ``start``/``end`` wrap the split.
    If *target_ratio* cannot be achieved without exceeding *max_chunk_len*
    on either side, returns ``(seg_a, seg_b)`` unchanged.
    """
    words = list(seg_a.words) + list(seg_b.words)
    if len(words) < 2:
        return seg_a, seg_b

    def _join(ws: list[Word]) -> str:
        # Rebuild text from surface forms, respecting ops tokenization.
        return ops.join([w.word for w in ws])

    best_i: int | None = None
    best_score = float("inf")
    for i in range(1, len(words)):
        left_text = _join(words[:i])
        right_text = _join(words[i:])
        left_len = ops.length(left_text)
        right_len = ops.length(right_text)
        if left_len > max_chunk_len or right_len > max_chunk_len:
            continue
        if right_len <= 0:
            continue
        score = abs(target_ratio - (left_len / right_len))
        if score < best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return seg_a, seg_b

    left_words = words[:best_i]
    right_words = words[best_i:]
    new_a = replace(
        seg_a,
        start=left_words[0].start if left_words else seg_a.start,
        end=left_words[-1].end if left_words else seg_a.end,
        text=_join(left_words),
        words=left_words,
    )
    new_b = replace(
        seg_b,
        start=right_words[0].start if right_words else seg_b.start,
        end=right_words[-1].end if right_words else seg_b.end,
        text=_join(right_words),
        words=right_words,
    )
    return new_a, new_b
