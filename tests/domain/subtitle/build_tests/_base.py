"""Shared base class for per-language Subtitle tests.

Subclasses set LANGUAGE plus test-data factory methods; the base
provides structural assertions that every language must pass.

Convention: base class name intentionally omits the ``Test`` prefix so
pytest does *not* try to collect it directly.
"""

from __future__ import annotations

from domain.subtitle import Segment, Word, SentenceRecord, Subtitle
from domain.lang import LangOps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def W(text: str, start: float, end: float, speaker: str | None = None) -> Word:
    """Shorthand Word constructor."""
    return Word(word=text, start=start, end=end, speaker=speaker)


def S(text: str, start: float, end: float, words: list[Word] | None = None) -> Segment:
    """Shorthand Segment constructor."""
    return Segment(start=start, end=end, text=text, words=words or [])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BuilderTestBase:
    """Subtitle test base.

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
        """Subtitle on empty segment list produces empty result."""
        assert Subtitle([], self.ops()).build() == []
        assert Subtitle([], self.ops()).sentences().build() == []
        assert Subtitle([], self.ops()).records() == []

    def test_immutability(self) -> None:
        """Chaining creates new instances; originals are unchanged."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        sub = Subtitle([seg], self.ops())
        s1 = sub.sentences()
        s2 = sub.clauses()

        r0 = sub.build()
        r1 = s1.build()
        r2 = s2.build()
        # All three pipelines independently round-trip the single input
        # segment (no sentence/clause boundary inside "Hello.").
        assert [s.text for s in r0] == ["Hello."]
        assert [s.text for s in r1] == ["Hello."]
        assert [s.text for s in r2] == ["Hello."]
        # Original didn't change
        assert r0[0].text == "Hello."

    def test_build_preserves_all_text(self) -> None:
        """join(build_results) reproduces the merged text."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = Subtitle([seg], self.ops()).build()
        assert "".join(s.text for s in result) == "Hello."

    def test_build_segments_have_words(self) -> None:
        """Every output segment has at least one word."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = Subtitle([seg], self.ops()).sentences().build()
        actual = [(s.text, [w.word for w in s.words]) for s in result]
        expected = [("Hello.", ["Hello."])]
        assert actual == expected

    def test_timing_monotonic(self) -> None:
        """Output segment timings are non-decreasing."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        result = Subtitle([seg], self.ops()).sentences().build()
        for i in range(1, len(result)):
            assert result[i].start >= result[i - 1].start

    def test_records_structure(self) -> None:
        """Each SentenceRecord has valid src_text, timing, and segments."""
        seg = S("Hello.", 0.0, 1.0, words=[W("Hello.", 0.0, 1.0)])
        records = Subtitle([seg], self.ops()).records()
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, SentenceRecord)
        assert rec.src_text == "Hello."
        assert rec.start == 0.0 and rec.end == 1.0
        assert [s.text for s in rec.segments] == ["Hello."]
