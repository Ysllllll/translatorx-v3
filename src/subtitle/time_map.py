"""TimeMap — maps character offsets to timestamps via piecewise linear interpolation.

Core concept: Span works in "character space", Segment works in "time space".
TimeMap is the coordinate transform between them.
"""

from __future__ import annotations

import bisect

from ._types import Segment, Word

# Punctuation characters for tolerant word matching.
_PUNCT = frozenset(
    '.,!?;:，。！？；：、'
    '"""\'\''
    '()[]{}（）《》「」『』【】〈〉'
    '—–‐-…·'
    '/\\@#$%^&*+=|~'
)


def _strip_punct(s: str) -> str:
    """Strip leading and trailing punctuation for content-based matching."""
    i = 0
    while i < len(s) and s[i] in _PUNCT:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in _PUNCT:
        j -= 1
    return s[i:j]


def _match_words(
    text: str, words: list[Word],
) -> list[tuple[int, int, float, float]]:
    """Match Word objects to character positions in text.

    Uses punctuation-tolerant matching: tries exact match first,
    then falls back to content-only match (punctuation stripped).

    Returns:
        List of (char_start, char_end, time_start, time_end) anchors.
    """
    anchors: list[tuple[int, int, float, float]] = []
    pos = 0
    for word in words:
        w = word.word
        if not w or w.isspace():
            continue

        # Exact match
        idx = text.find(w, pos)
        if idx >= 0:
            anchors.append((idx, idx + len(w), word.start, word.end))
            pos = idx + len(w)
            continue

        # Fallback: strip punctuation, match content
        content = _strip_punct(w)
        if content:
            idx = text.find(content, pos)
            if idx >= 0:
                anchors.append((idx, idx + len(content), word.start, word.end))
                pos = idx + len(content)

    return anchors


def _piecewise_lerp(
    knot_pos: list[int], knot_time: list[float], n: int,
) -> list[float]:
    """Piecewise linear interpolation producing n+1 boundary values."""
    result = [0.0] * (n + 1)
    for i in range(n + 1):
        j = bisect.bisect_right(knot_pos, i) - 1
        if j < 0:
            result[i] = knot_time[0]
        elif j >= len(knot_pos) - 1:
            result[i] = knot_time[-1]
        elif knot_pos[j] == i:
            result[i] = knot_time[j]
        else:
            p0, p1 = knot_pos[j], knot_pos[j + 1]
            t0, t1 = knot_time[j], knot_time[j + 1]
            result[i] = t0 + (t1 - t0) * (i - p0) / (p1 - p0)
    return result


class TimeMap:
    """Maps character offsets in concatenated segment text to timestamps.

    Uses word-level timing when available (with punctuation-tolerant matching),
    falls back to per-segment linear interpolation otherwise.

    Boundary model: ``boundaries[i]`` is the time at the edge *before*
    character ``i``.  ``boundaries[len(text)]`` is the time after the last
    character.  This matches Span's ``[start, end)`` convention::

        span = Span("Hello", 0, 5)
        start_time, end_time = time_map.time_range(span.start, span.end)
    """

    __slots__ = ("_text", "_boundaries")

    def __init__(self, text: str, boundaries: list[float]) -> None:
        self._text = text
        self._boundaries = boundaries

    @classmethod
    def from_segments(
        cls,
        segments: list[Segment],
        separator: str = " ",
    ) -> TimeMap:
        """Build a TimeMap by concatenating segment texts.

        Args:
            segments: Ordered segments with timing (and optional word-level timing).
            separator: Inserted between segment texts (default ``" "``).
                       Use a non-empty separator for multi-segment input to avoid
                       ambiguous time boundaries at segment junctions.
        """
        if not segments:
            return cls("", [0.0])

        full_text = separator.join(s.text for s in segments)
        n = len(full_text)

        # Collect control points: (char_position, time)
        raw_knots: list[tuple[int, float]] = [(0, segments[0].start)]
        offset = 0

        for i, seg in enumerate(segments):
            if i > 0:
                offset += len(separator)

            if seg.words:
                for cs, ce, ts, te in _match_words(seg.text, seg.words):
                    raw_knots.append((offset + cs, ts))
                    raw_knots.append((offset + ce, te))
            else:
                raw_knots.append((offset, seg.start))
                raw_knots.append((offset + len(seg.text), seg.end))

            offset += len(seg.text)

        raw_knots.append((n, segments[-1].end))

        # Deduplicate by position (last write wins — later segment's start
        # takes precedence over earlier segment's end at junction points)
        by_pos: dict[int, float] = {}
        for p, t in raw_knots:
            by_pos[p] = t
        items = sorted(by_pos.items())

        boundaries = _piecewise_lerp(
            [p for p, _ in items],
            [t for _, t in items],
            n,
        )

        return cls(full_text, boundaries)

    @property
    def text(self) -> str:
        """The concatenated text."""
        return self._text

    def time_at(self, char_offset: int) -> float:
        """Time at the given character offset."""
        return self._boundaries[char_offset]

    def time_range(self, char_start: int, char_end: int) -> tuple[float, float]:
        """Time interval for character range ``[char_start, char_end)``.

        Designed to work directly with Span offsets::

            spans = ChunkPipeline(time_map.text, language="en").sentences().spans()
            for span in spans:
                start_t, end_t = time_map.time_range(span.start, span.end)
        """
        return (self._boundaries[char_start], self._boundaries[char_end])
