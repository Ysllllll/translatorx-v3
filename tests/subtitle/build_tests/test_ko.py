"""Korean (ko) Subtitle tests.

Test data simulates real ASR output: eojeol-level words (space-separated
morpheme groups), punctuation as separate tokens, sentences split across
segment boundaries.
"""

from __future__ import annotations

from subtitle import Segment, Subtitle
from lang_ops import LangOps
from ._base import BuilderTestBase, S, W


_ops = LangOps.for_language("ko")


# ---------------------------------------------------------------------------
# Realistic test data — eojeol-level ASR words
# ---------------------------------------------------------------------------

def _news_segments() -> list[Segment]:
    """Simulates a Korean news broadcast ASR result.

    Eojeol-level words (typical of Korean ASR).
    Sentences span across segment boundaries.

    Full text: "최근 인공지능 기술이 빠르게 발전하고 있습니다.
    전문가들은 이 추세가 계속될 것이라고 전망합니다.
    그러나 우려의 목소리도 있습니다."
    """
    return [
        S("최근 인공지능 기술이 빠르게 발전하고 있습니다.", 0.0, 5.0, words=[
            W("최근", 0.0, 0.5), W("인공지능", 0.5, 1.2),
            W("기술이", 1.2, 1.8), W("빠르게", 1.8, 2.4),
            W("발전하고", 2.4, 3.2), W("있습니다", 3.2, 4.5),
            W(".", 4.5, 5.0),
        ]),
        S("전문가들은 이 추세가 계속될 것이라고 전망합니다. 그러나", 5.0, 10.0, words=[
            W("전문가들은", 5.0, 5.8), W("이", 5.8, 6.0),
            W("추세가", 6.0, 6.6), W("계속될", 6.6, 7.2),
            W("것이라고", 7.2, 7.8), W("전망합니다", 7.8, 8.8),
            W(".", 8.8, 9.0),
            W("그러나", 9.0, 10.0),
        ]),
        S("우려의 목소리도 있습니다.", 10.0, 13.0, words=[
            W("우려의", 10.0, 10.6), W("목소리도", 10.6, 11.4),
            W("있습니다", 11.4, 12.5),
            W(".", 12.5, 13.0),
        ]),
    ]


def _short_segments() -> list[Segment]:
    """Two simple Korean segments."""
    return [
        S("안녕하세요.", 0.0, 2.0, words=[
            W("안녕하세요", 0.0, 1.8), W(".", 1.8, 2.0),
        ]),
        S("반갑습니다!", 2.0, 4.0, words=[
            W("반갑습니다", 2.0, 3.8), W("!", 3.8, 4.0),
        ]),
    ]


def _clause_segments() -> list[Segment]:
    """Korean segment with clause separators.

    "사과, 바나나, 오렌지는 과일이고; 우유, 빵은 아침식사입니다."
    """
    return [
        S("사과, 바나나, 오렌지는 과일이고; 우유, 빵은 아침식사입니다.", 0.0, 8.0, words=[
            W("사과", 0.0, 0.5), W(",", 0.5, 0.6),
            W("바나나", 0.6, 1.2), W(",", 1.2, 1.3),
            W("오렌지는", 1.3, 2.0), W("과일이고", 2.0, 3.0),
            W(";", 3.0, 3.2),
            W("우유", 3.2, 3.8), W(",", 3.8, 3.9),
            W("빵은", 3.9, 4.5), W("아침식사입니다", 4.5, 7.5),
            W(".", 7.5, 8.0),
        ]),
    ]


def _multi_speaker_segments() -> list[Segment]:
    """Korean dialogue with speaker changes."""
    return [
        S("어떻게 생각하세요? 아주 좋다고 생각합니다!", 0.0, 5.0, words=[
            W("어떻게", 0.0, 0.6, speaker="A"),
            W("생각하세요", 0.6, 1.5, speaker="A"),
            W("?", 1.5, 2.0, speaker="A"),
            W("아주", 2.0, 2.5, speaker="B"),
            W("좋다고", 2.5, 3.2, speaker="B"),
            W("생각합니다", 3.2, 4.5, speaker="B"),
            W("!", 4.5, 5.0, speaker="B"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Inherits structural invariants
# ---------------------------------------------------------------------------

class TestKoreanBuilder(BuilderTestBase):
    LANGUAGE = "ko"


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

class TestKoreanSentences:

    def test_two_sentences(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        assert [s.text for s in result] == ["안녕하세요.", "반갑습니다!"]

    def test_sentences_across_boundaries(self) -> None:
        result = Subtitle(_news_segments(), _ops).sentences().build()
        texts = [s.text for s in result]
        assert len(texts) == 3
        assert "인공지능" in texts[0]
        assert "전망합니다" in texts[1]
        assert "있습니다" in texts[2]

    def test_sentence_timing(self) -> None:
        result = Subtitle(_news_segments(), _ops).sentences().build()
        assert result[0].start == 0.0
        assert result[-1].end == 13.0

    def test_words_preserved(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        for seg in result:
            assert len(seg.words) >= 1


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------

class TestKoreanClauses:

    def test_clause_split(self) -> None:
        result = Subtitle(_clause_segments(), _ops).clauses().build()
        assert len(result) >= 3  # at least comma + semicolon splits

    def test_clause_timing(self) -> None:
        result = Subtitle(_clause_segments(), _ops).clauses().build()
        assert result[0].start == 0.0
        assert result[-1].end == 8.0

    def test_sentences_then_clauses(self) -> None:
        result = (Subtitle(_news_segments(), _ops)
                  .sentences()
                  .clauses()
                  .build())
        # More clauses than sentences
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# By length
# ---------------------------------------------------------------------------

class TestKoreanByLength:

    def test_sentences_then_max_length(self) -> None:
        result = (Subtitle(_news_segments(), _ops)
                  .sentences()
                  .max_length(15)
                  .build())
        for seg in result:
            # Allow slight overshoot for indivisible eojeols
            assert _ops.length(seg.text) <= 20

    def test_text_preserved(self) -> None:
        result = (Subtitle(_news_segments(), _ops)
                  .sentences()
                  .max_length(15)
                  .build())
        joined = " ".join(s.text for s in result)
        # All content words present
        assert "인공지능" in joined
        assert "전망합니다" in joined


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestKoreanMerge:

    def test_merge_clauses_back(self) -> None:
        clauses = Subtitle(_clause_segments(), _ops).clauses().build()
        merged = Subtitle(_clause_segments(), _ops).clauses().merge(30).build()
        assert len(merged) <= len(clauses)

    def test_merge_preserves_text(self) -> None:
        result = Subtitle(_clause_segments(), _ops).clauses().merge(30).build()
        joined = " ".join(s.text for s in result)
        assert "사과" in joined
        assert "아침식사입니다" in joined

    def test_merge_words_timing(self) -> None:
        result = Subtitle(_clause_segments(), _ops).clauses().merge(30).build()
        for seg in result:
            assert len(seg.words) >= 1
            assert seg.start <= seg.end


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestKoreanRecords:

    def test_records_structure(self) -> None:
        records = Subtitle(_news_segments(), _ops).records()
        assert len(records) >= 1
        for rec in records:
            assert rec.src_text
            assert rec.start <= rec.end
            assert len(rec.segments) >= 1

    def test_records_with_max_length(self) -> None:
        records = Subtitle(_news_segments(), _ops).records(max_length=12)
        for rec in records:
            for seg in rec.segments:
                assert _ops.length(seg.text) <= 18  # some tolerance


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------

class TestKoreanSpeaker:

    def test_speaker_change_creates_boundary(self) -> None:
        result = (Subtitle(_multi_speaker_segments(), _ops,
                                 split_by_speaker=True)
                  .sentences()
                  .build())
        # Speaker change should create at least 2 groups
        assert len(result) >= 2

    def test_same_speaker_no_extra_splits(self) -> None:
        single = [S("안녕하세요. 반갑습니다.", 0.0, 3.0, words=[
            W("안녕하세요", 0.0, 1.0, speaker="A"),
            W(".", 1.0, 1.2, speaker="A"),
            W("반갑습니다", 1.2, 2.8, speaker="A"),
            W(".", 2.8, 3.0, speaker="A"),
        ])]
        result = (Subtitle(single, _ops, split_by_speaker=True)
                  .sentences()
                  .build())
        assert len(result) == 2  # two sentences, no extra speaker splits


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class TestKoreanStream:

    def test_stream_incremental(self) -> None:
        stream = Subtitle.stream(_ops)
        segs = _news_segments()
        all_done: list[Segment] = []
        for seg in segs:
            all_done.extend(stream.feed(seg))
        all_done.extend(stream.flush())
        assert len(all_done) >= 1
        joined = " ".join(s.text for s in all_done)
        assert "인공지능" in joined

    def test_stream_flush_empty(self) -> None:
        stream = Subtitle.stream(_ops)
        assert stream.flush() == []
