"""Shared base class for per-language SegmentProcessor tests.

Subclasses set LANGUAGE plus test-data factory methods; the base
provides structural assertions that every language must pass.

Convention: base class name intentionally omits the ``Test`` prefix so
pytest does *not* try to collect it directly.
"""

from __future__ import annotations

from subtitle import Segment, Word, SentenceRecord, SegmentProcessor
from lang_ops import LangOps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def W(text: str, start: float, end: float,
      speaker: str | None = None) -> Word:
    """Shorthand Word constructor."""
    return Word(word=text, start=start, end=end, speaker=speaker)


def S(text: str, start: float, end: float,
      words: list[Word] | None = None) -> Segment:
    """Shorthand Segment constructor."""
    return Segment(start=start, end=end, text=text, words=words or [])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BuilderTestBase:
    """SegmentProcessor test base.

    Required class attributes
    -------------------------
    LANGUAGE : str
        ISO language code (e.g. ``"en"``, ``"zh"``).
    """

    LANGUAGE: str = ""

    # Cached ops — set once per subclass
    _ops: LangOps | None = None

    @classmethod
    def ops(cls) -> LangOps:
        if cls._ops is None:
            cls._ops = LangOps.for_language(cls.LANGUAGE)
        return cls._ops

    # ------------------------------------------------------------------
    # Structural invariants — every language must satisfy these
    # ------------------------------------------------------------------

    def test_empty_input(self) -> None:
        """Processor on empty segment list produces empty result."""
        assert SegmentProcessor([], self.ops()).build() == []
        assert SegmentProcessor([], self.ops()).sentences().build() == []
        assert SegmentProcessor([], self.ops()).records() == []

    def test_immutability(self) -> None:
        """Chaining creates new processors; originals are unchanged."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        proc = SegmentProcessor([seg], self.ops())
        p1 = proc.sentences()
        p2 = proc.clauses()

        r0 = proc.build()
        r1 = p1.build()
        r2 = p2.build()
        # All three return valid results independently
        assert len(r0) >= 1
        assert len(r1) >= 1
        assert len(r2) >= 1
        # Original didn't change
        assert r0[0].text == "Hello."

    def test_build_preserves_all_text(self) -> None:
        """join(build_results) reproduces the merged text."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = SegmentProcessor([seg], self.ops()).build()
        assert "".join(s.text for s in result) == "Hello."

    def test_build_segments_have_words(self) -> None:
        """Every output segment has at least one word."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = SegmentProcessor([seg], self.ops()).sentences().build()
        for s in result:
            assert len(s.words) >= 1, f"Segment {s.text!r} has no words"

    def test_timing_monotonic(self) -> None:
        """Output segment timings are non-decreasing."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = SegmentProcessor([seg], self.ops()).sentences().build()
        for i in range(1, len(result)):
            assert result[i].start >= result[i - 1].start

    def test_records_structure(self) -> None:
        """Each SentenceRecord has valid src_text, timing, and segments."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        records = SegmentProcessor([seg], self.ops()).records()
        for rec in records:
            assert isinstance(rec, SentenceRecord)
            assert rec.src_text
            assert rec.start <= rec.end
            assert len(rec.segments) >= 1
