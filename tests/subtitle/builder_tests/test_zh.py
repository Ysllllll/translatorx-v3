"""Chinese (zh) SegmentBuilder tests.

Test data simulates real ASR output: single-character words (Whisper-style),
punctuation as separate tokens, sentences split across segment boundaries.
"""

from __future__ import annotations

import pytest
from subtitle import Segment, Word, SentenceRecord, SegmentBuilder
from lang_ops import TextOps
from ._base import BuilderTestBase, W, S


_ops = TextOps.for_language("zh")


# ---------------------------------------------------------------------------
# Realistic test data — single-character ASR words
# ---------------------------------------------------------------------------

def _asr_news_segments() -> list[Segment]:
    """Simulates a Chinese news broadcast ASR result.

    Single-character words (typical of Whisper zh output).
    Sentences span across segment boundaries.

    Full text: "近年来，人工智能技术蓬勃发展。专家认为，这一趋势将持续加速；
    然而，也有学者表达了担忧。我们需要审慎评估新技术的风险。"
    """
    return [
        S("近年来，人工智能技术蓬勃发展。", 0.0, 5.0, words=[
            W("近", 0.0, 0.3), W("年", 0.3, 0.6), W("来", 0.6, 0.9),
            W("，", 0.9, 1.0),
            W("人", 1.0, 1.3), W("工", 1.3, 1.6), W("智", 1.6, 1.9),
            W("能", 1.9, 2.2), W("技", 2.2, 2.5), W("术", 2.5, 2.8),
            W("蓬", 2.8, 3.1), W("勃", 3.1, 3.4),
            W("发", 3.4, 3.8), W("展", 3.8, 4.5),
            W("。", 4.5, 5.0),
        ]),
        S("专家认为，这一趋势将持续加速；然而，", 5.0, 10.0, words=[
            W("专", 5.0, 5.3), W("家", 5.3, 5.5),
            W("认", 5.5, 5.8), W("为", 5.8, 6.0),
            W("，", 6.0, 6.2),
            W("这", 6.2, 6.5), W("一", 6.5, 6.7),
            W("趋", 6.7, 7.0), W("势", 7.0, 7.3),
            W("将", 7.3, 7.6),
            W("持", 7.6, 7.9), W("续", 7.9, 8.2),
            W("加", 8.2, 8.5), W("速", 8.5, 8.8),
            W("；", 8.8, 9.0),
            W("然", 9.0, 9.3), W("而", 9.3, 9.5),
            W("，", 9.5, 10.0),
        ]),
        S("也有学者表达了担忧。我们需要审慎评估新技术的风险。", 10.0, 18.0, words=[
            W("也", 10.0, 10.2), W("有", 10.2, 10.5),
            W("学", 10.5, 10.8), W("者", 10.8, 11.0),
            W("表", 11.0, 11.3), W("达", 11.3, 11.5),
            W("了", 11.5, 11.7),
            W("担", 11.7, 12.0), W("忧", 12.0, 12.5),
            W("。", 12.5, 13.0),
            W("我", 13.0, 13.3), W("们", 13.3, 13.5),
            W("需", 13.5, 13.8), W("要", 13.8, 14.0),
            W("审", 14.0, 14.3), W("慎", 14.3, 14.6),
            W("评", 14.6, 14.9), W("估", 14.9, 15.2),
            W("新", 15.2, 15.5), W("技", 15.5, 15.8),
            W("术", 15.8, 16.0), W("的", 16.0, 16.3),
            W("风", 16.3, 16.6), W("险", 16.6, 17.5),
            W("。", 17.5, 18.0),
        ]),
    ]


def _short_segments() -> list[Segment]:
    """Two simple Chinese segments."""
    return [
        S("你好世界。", 0.0, 2.0, words=[
            W("你", 0.0, 0.4), W("好", 0.4, 0.8),
            W("世", 0.8, 1.3), W("界", 1.3, 1.8),
            W("。", 1.8, 2.0),
        ]),
        S("今天天气不错！", 2.0, 5.0, words=[
            W("今", 2.0, 2.4), W("天", 2.4, 2.8),
            W("天", 2.8, 3.1), W("气", 3.1, 3.5),
            W("不", 3.5, 4.0), W("错", 4.0, 4.5),
            W("！", 4.5, 5.0),
        ]),
    ]


def _clause_rich_segment() -> list[Segment]:
    """One segment with many clause separators."""
    # "苹果、香蕉、橘子，都是水果；牛奶、面包，都是早餐。"
    return [
        S("苹果、香蕉、橘子，都是水果；牛奶、面包，都是早餐。", 0.0, 10.0, words=[
            W("苹", 0.0, 0.3), W("果", 0.3, 0.5),
            W("、", 0.5, 0.6),
            W("香", 0.6, 0.9), W("蕉", 0.9, 1.2),
            W("、", 1.2, 1.3),
            W("橘", 1.3, 1.6), W("子", 1.6, 1.9),
            W("，", 1.9, 2.0),
            W("都", 2.0, 2.3), W("是", 2.3, 2.6),
            W("水", 2.6, 2.9), W("果", 2.9, 3.5),
            W("；", 3.5, 3.8),
            W("牛", 3.8, 4.1), W("奶", 4.1, 4.5),
            W("、", 4.5, 4.6),
            W("面", 4.6, 4.9), W("包", 4.9, 5.5),
            W("，", 5.5, 5.7),
            W("都", 5.7, 6.0), W("是", 6.0, 6.3),
            W("早", 6.3, 6.6), W("餐", 6.6, 7.5),
            W("。", 7.5, 10.0),
        ]),
    ]


def _multi_speaker_segments() -> list[Segment]:
    """Chinese dialogue with speaker changes.

    Full text: "你觉得怎么样？我觉得非常好！真的吗？当然是真的。"
    """
    return [
        S("你觉得怎么样？我觉得非常好！", 0.0, 5.0, words=[
            W("你", 0.0, 0.3, speaker="A"),
            W("觉", 0.3, 0.5, speaker="A"),
            W("得", 0.5, 0.7, speaker="A"),
            W("怎", 0.7, 0.9, speaker="A"),
            W("么", 0.9, 1.1, speaker="A"),
            W("样", 1.1, 1.5, speaker="A"),
            W("？", 1.5, 2.0, speaker="A"),
            W("我", 2.0, 2.3, speaker="B"),
            W("觉", 2.3, 2.5, speaker="B"),
            W("得", 2.5, 2.7, speaker="B"),
            W("非", 2.7, 3.0, speaker="B"),
            W("常", 3.0, 3.3, speaker="B"),
            W("好", 3.3, 4.5, speaker="B"),
            W("！", 4.5, 5.0, speaker="B"),
        ]),
        S("真的吗？当然是真的。", 5.0, 9.0, words=[
            W("真", 5.0, 5.3, speaker="A"),
            W("的", 5.3, 5.5, speaker="A"),
            W("吗", 5.5, 5.8, speaker="A"),
            W("？", 5.8, 6.0, speaker="A"),
            W("当", 6.0, 6.3, speaker="B"),
            W("然", 6.3, 6.6, speaker="B"),
            W("是", 6.6, 6.9, speaker="B"),
            W("真", 6.9, 7.2, speaker="B"),
            W("的", 7.2, 8.0, speaker="B"),
            W("。", 8.0, 9.0, speaker="B"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Inherits structural invariants
# ---------------------------------------------------------------------------

class TestChineseBuilder(BuilderTestBase):
    LANGUAGE = "zh"


# ---------------------------------------------------------------------------
# CJK join — no spaces
# ---------------------------------------------------------------------------

class TestChineseJoin:

    def test_segments_joined_without_spaces(self) -> None:
        """CJK segments are joined with empty string, not spaces."""
        result = SegmentBuilder(_short_segments(), _ops).build()
        assert len(result) == 1
        assert result[0].text == "你好世界。今天天气不错！"

    def test_three_segments_joined(self) -> None:
        segments = [
            S("你好", 0.0, 1.0, words=[W("你", 0.0, 0.5), W("好", 0.5, 1.0)]),
            S("世界", 1.0, 2.0, words=[W("世", 1.0, 1.5), W("界", 1.5, 2.0)]),
            S("再见", 2.0, 3.0, words=[W("再", 2.0, 2.5), W("见", 2.5, 3.0)]),
        ]
        result = SegmentBuilder(segments, _ops).build()
        assert result[0].text == "你好世界再见"
        assert result[0].start == 0.0
        assert result[0].end == 3.0


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

class TestChineseSentences:

    def test_two_sentences(self) -> None:
        result = SegmentBuilder(_short_segments(), _ops).sentences().build()
        assert [s.text for s in result] == ["你好世界。", "今天天气不错！"]
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 2.0
        assert result[1].end == 5.0

    def test_sentences_across_boundaries(self) -> None:
        result = SegmentBuilder(_asr_news_segments(), _ops).sentences().build()
        texts = [s.text for s in result]
        assert texts == [
            "近年来，人工智能技术蓬勃发展。",
            "专家认为，这一趋势将持续加速；然而，也有学者表达了担忧。",
            "我们需要审慎评估新技术的风险。",
        ]

    def test_sentence_timing(self) -> None:
        result = SegmentBuilder(_asr_news_segments(), _ops).sentences().build()
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[-1].start == 13.0
        assert result[-1].end == 18.0

    def test_char_level_words_preserved(self) -> None:
        """Single-character ASR words are correctly distributed."""
        result = SegmentBuilder(_short_segments(), _ops).sentences().build()
        # "你好世界。" → 5 words (你/好/世/界/。)
        assert len(result[0].words) == 5
        assert [w.word for w in result[0].words] == ["你", "好", "世", "界", "。"]
        assert result[0].words[0].start == 0.0
        assert result[0].words[-1].end == 2.0


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------

class TestChineseClauses:

    def test_clause_split(self) -> None:
        result = SegmentBuilder(_clause_rich_segment(), _ops).clauses().build()
        texts = [s.text for s in result]
        assert texts == [
            "苹果、", "香蕉、", "橘子，",
            "都是水果；",
            "牛奶、", "面包，",
            "都是早餐。",
        ]

    def test_clause_timing(self) -> None:
        result = SegmentBuilder(_clause_rich_segment(), _ops).clauses().build()
        # "苹果、" — starts at 0.0 (苹), ends at 0.6 (、)
        assert result[0].start == 0.0
        assert result[0].end == 0.6
        # "都是早餐。" — starts at 5.7 (都), ends at 10.0 (。)
        assert result[-1].start == 5.7
        assert result[-1].end == 10.0

    def test_sentences_then_clauses(self) -> None:
        result = (SegmentBuilder(_asr_news_segments(), _ops)
                  .sentences()
                  .clauses()
                  .build())
        texts = [s.text for s in result]
        assert texts == [
            "近年来，",
            "人工智能技术蓬勃发展。",
            "专家认为，",
            "这一趋势将持续加速；",
            "然而，",
            "也有学者表达了担忧。",
            "我们需要审慎评估新技术的风险。",
        ]


# ---------------------------------------------------------------------------
# By length
# ---------------------------------------------------------------------------

class TestChineseByLength:

    def test_length_constraint(self) -> None:
        result = (SegmentBuilder(_asr_news_segments(), _ops)
                  .sentences()
                  .by_length(10)
                  .build())
        for seg in result:
            assert _ops.length(seg.text) <= 10 or len(_ops.split(seg.text)) == 1, \
                f"Segment too long: {seg.text!r} ({_ops.length(seg.text)} chars)"

    def test_text_preserved(self) -> None:
        result = (SegmentBuilder(_asr_news_segments(), _ops)
                  .sentences()
                  .by_length(10)
                  .build())
        merged = "".join(s.text for s in result)
        original = "".join(s.text for s in _asr_news_segments())
        assert merged == original

    def test_full_chain(self) -> None:
        """sentences → clauses → by_length."""
        result = (SegmentBuilder(_asr_news_segments(), _ops)
                  .sentences()
                  .clauses()
                  .by_length(8)
                  .build())
        for seg in result:
            assert _ops.length(seg.text) <= 8 or len(_ops.split(seg.text)) == 1, \
                f"Segment too long: {seg.text!r}"

    def test_short_text_no_split(self) -> None:
        result = SegmentBuilder(_short_segments(), _ops).by_length(50).build()
        assert len(result) == 1
        assert result[0].text == "你好世界。今天天气不错！"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestChineseRecords:

    def test_records_structure(self) -> None:
        records = SegmentBuilder(_short_segments(), _ops).records()
        assert len(records) == 2
        assert records[0].src_text == "你好世界。"
        assert records[1].src_text == "今天天气不错！"
        assert records[0].start == 0.0
        assert records[0].end == 2.0

    def test_records_with_max_length(self) -> None:
        records = SegmentBuilder(_asr_news_segments(), _ops).records(max_length=8)
        for rec in records:
            for seg in rec.segments:
                assert _ops.length(seg.text) <= 8 or len(_ops.split(seg.text)) == 1, \
                    f"Sub-segment too long: {seg.text!r}"
            # Sub-segments have words
            for seg in rec.segments:
                assert len(seg.words) >= 1

    def test_records_sub_segments_cover_sentence(self) -> None:
        records = SegmentBuilder(_asr_news_segments(), _ops).records(max_length=8)
        for rec in records:
            merged = "".join(s.text for s in rec.segments)
            assert merged == rec.src_text, \
                f"Sub-segments don't cover sentence: {merged!r} != {rec.src_text!r}"


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------

class TestChineseSpeaker:

    def test_speaker_change_creates_boundary(self) -> None:
        result = (SegmentBuilder(_multi_speaker_segments(), _ops,
                                 split_by_speaker=True)
                  .sentences()
                  .build())
        texts = [s.text for s in result]
        # Speaker A: "你觉得怎么样？", Speaker B: "我觉得非常好！",
        # Speaker A: "真的吗？", Speaker B: "当然是真的。"
        assert any("你觉得怎么样？" in t for t in texts)
        assert any("我觉得非常好！" in t for t in texts)

    def test_same_speaker_no_extra_splits(self) -> None:
        segments = [S("你好世界。今天不错！", 0.0, 5.0, words=[
            W("你", 0.0, 0.3, speaker="A"), W("好", 0.3, 0.6, speaker="A"),
            W("世", 0.6, 0.9, speaker="A"), W("界", 0.9, 1.5, speaker="A"),
            W("。", 1.5, 2.0, speaker="A"),
            W("今", 2.0, 2.3, speaker="A"), W("天", 2.3, 2.6, speaker="A"),
            W("不", 2.6, 3.0, speaker="A"), W("错", 3.0, 4.0, speaker="A"),
            W("！", 4.0, 5.0, speaker="A"),
        ])]
        r_with = SegmentBuilder(segments, _ops, split_by_speaker=True).sentences().build()
        r_without = SegmentBuilder(segments, _ops).sentences().build()
        assert [s.text for s in r_with] == [s.text for s in r_without]


# ---------------------------------------------------------------------------
# Auto-fill words
# ---------------------------------------------------------------------------

class TestChineseAutoFill:

    def test_no_words_auto_filled(self) -> None:
        segments = [
            S("你好世界。", 0.0, 2.0),
            S("今天天气好。", 2.0, 5.0),
        ]
        result = SegmentBuilder(segments, _ops).sentences().build()
        assert [s.text for s in result] == ["你好世界。", "今天天气好。"]
        assert len(result[0].words) >= 1
        assert len(result[1].words) >= 1


# ---------------------------------------------------------------------------
# Stream mode
# ---------------------------------------------------------------------------

class TestChineseStream:

    def test_stream_incremental(self) -> None:
        stream = SegmentBuilder.stream(_ops)
        all_done: list[Segment] = []
        for seg in _asr_news_segments():
            all_done.extend(stream.feed(seg))
        all_done.extend(stream.flush())

        merged = "".join(s.text for s in all_done)
        original = "".join(s.text for s in _asr_news_segments())
        assert merged == original

    def test_stream_flush_empty(self) -> None:
        stream = SegmentBuilder.stream(_ops)
        assert stream.flush() == []

    def test_stream_single_segment(self) -> None:
        stream = SegmentBuilder.stream(_ops)
        done = stream.feed(S("你好世界。", 0.0, 2.0, words=[
            W("你", 0.0, 0.4), W("好", 0.4, 0.8),
            W("世", 0.8, 1.3), W("界", 1.3, 1.8),
            W("。", 1.8, 2.0),
        ]))
        assert done == []

        rest = stream.flush()
        assert len(rest) == 1
        assert rest[0].text == "你好世界。"
