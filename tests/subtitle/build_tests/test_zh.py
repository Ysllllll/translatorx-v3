"""Chinese (zh) Subtitle tests.

Test data simulates real ASR output: single-character words (Whisper-style),
punctuation as separate tokens, sentences split across segment boundaries.
"""

from __future__ import annotations

from subtitle import Segment, Subtitle
from lang_ops import LangOps
from ._base import BuilderTestBase, S, W


_ops = LangOps.for_language("zh")


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


def _mixed_language_segments() -> list[Segment]:
    """Chinese segment mixed with English words."""
    return [
        S("我正在学习Python编程。", 0.0, 3.0, words=[
            W("我", 0.0, 0.3), W("正", 0.3, 0.6), W("在", 0.6, 0.9),
            W("学", 0.9, 1.2), W("习", 1.2, 1.5),
            W("Python", 1.5, 2.2),
            W("编", 2.2, 2.5), W("程", 2.5, 2.8),
            W("。", 2.8, 3.0),
        ]),
        S("This是混合测试！", 3.0, 5.0, words=[
            W("This", 3.0, 3.5), W("是", 3.5, 3.8),
            W("混", 3.8, 4.1), W("合", 4.1, 4.4),
            W("测", 4.4, 4.7), W("试", 4.7, 4.9),
            W("！", 4.9, 5.0),
        ])
    ]


def _extreme_length_segment() -> list[Segment]:
    """Segment with extremely long word that exceeds max_length."""
    return [
        S("这是一个超级长的英文单词supercalifragilisticexpialidocious测试", 0.0, 5.0, words=[
            W("这", 0.0, 0.2), W("是", 0.2, 0.4), W("一", 0.4, 0.6), W("个", 0.6, 0.8),
            W("超", 0.8, 1.0), W("级", 1.0, 1.2), W("长", 1.2, 1.4), W("的", 1.4, 1.6),
            W("英", 1.6, 1.8), W("文", 1.8, 2.0), W("单", 2.0, 2.2), W("词", 2.2, 2.5),
            W("supercalifragilisticexpialidocious", 2.5, 4.5),
            W("测", 4.5, 4.7), W("试", 4.7, 5.0),
        ])
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


def _abnormal_punctuation_segments() -> list[Segment]:
    """Segments with multiple punctuations, missing punctuation, or spaces."""
    return [
        S("等等！！！你确定吗？？？", 0.0, 3.0, words=[
            W("等", 0.0, 0.3), W("等", 0.3, 0.6),
            W("！", 0.6, 0.8), W("！", 0.8, 1.0), W("！", 1.0, 1.2),
            W("你", 1.2, 1.5), W("确", 1.5, 1.8), W("定", 1.8, 2.1), W("吗", 2.1, 2.4),
            W("？", 2.4, 2.6), W("？", 2.6, 2.8), W("？", 2.8, 3.0),
        ]),
        S("这  是  空 格 测试", 3.0, 6.0, words=[
            W("这", 3.0, 3.3), W(" ", 3.3, 3.4), W(" ", 3.4, 3.5),
            W("是", 3.5, 3.8), W(" ", 3.8, 3.9), W(" ", 3.9, 4.0),
            W("空", 4.0, 4.3), W(" ", 4.3, 4.5), W("格", 4.5, 4.8),
            W(" ", 4.8, 5.0), W("测", 5.0, 5.5), W("试", 5.5, 6.0),
        ])
    ]


def _extreme_short_segment() -> list[Segment]:
    """Extremely short segments, e.g., single character or just punctuation."""
    return [
        S("啊", 0.0, 0.5, words=[W("啊", 0.0, 0.5)]),
        S("？", 0.5, 1.0, words=[W("？", 0.5, 1.0)]),
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
        result = Subtitle(_short_segments(), _ops).build()
        assert [s.text for s in result] == ["你好世界。今天天气不错！"]

    def test_three_segments_joined(self) -> None:
        segments = [
            S("你好", 0.0, 1.0, words=[W("你", 0.0, 0.5), W("好", 0.5, 1.0)]),
            S("世界", 1.0, 2.0, words=[W("世", 1.0, 1.5), W("界", 1.5, 2.0)]),
            S("再见", 2.0, 3.0, words=[W("再", 2.0, 2.5), W("见", 2.5, 3.0)]),
        ]
        result = Subtitle(segments, _ops).build()
        assert [s.text for s in result] == ["你好世界再见"]


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

class TestChineseSentences:

    def test_two_sentences(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        assert [s.text for s in result] == ["你好世界。", "今天天气不错！"]

    def test_sentences_across_boundaries(self) -> None:
        result = Subtitle(_asr_news_segments(), _ops).sentences().build()
        assert [s.text for s in result] == [
            "近年来，人工智能技术蓬勃发展。",
            "专家认为，这一趋势将持续加速；然而，也有学者表达了担忧。",
            "我们需要审慎评估新技术的风险。",
        ]

    def test_sentence_timing(self) -> None:
        result = Subtitle(_asr_news_segments(), _ops).sentences().build()
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[-1].start == 13.0
        assert result[-1].end == 18.0

    def test_char_level_words_preserved(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        assert [w.word for w in result[0].words] == ["你", "好", "世", "界", "。"]

    def test_abnormal_punctuation_sentences(self) -> None:
        result = Subtitle(_abnormal_punctuation_segments(), _ops).sentences().build()
        assert [s.text for s in result] == ["等等！！！", "你确定吗？？？", "这是空格测试"]

    def test_extreme_short_sentences(self) -> None:
        result = Subtitle(_extreme_short_segment(), _ops).sentences().build()
        assert [s.text for s in result] == ["啊？"]


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------

class TestChineseClauses:

    def test_clause_split(self) -> None:
        result = Subtitle(_clause_rich_segment(), _ops).clauses().build()
        assert [s.text for s in result] == [
            "苹果、", "香蕉、", "橘子，",
            "都是水果；",
            "牛奶、", "面包，",
            "都是早餐。",
        ]

    def test_clause_timing(self) -> None:
        result = Subtitle(_clause_rich_segment(), _ops).clauses().build()
        assert result[0].start == 0.0
        assert result[0].end == 0.6
        assert result[-1].start == 5.7
        assert result[-1].end == 10.0

    def test_sentences_then_clauses(self) -> None:
        result = (Subtitle(_asr_news_segments(), _ops)
                  .sentences()
                  .clauses()
                  .build())
        assert [s.text for s in result] == [
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

    def test_sentences_then_max_length(self) -> None:
        result = (Subtitle(_asr_news_segments(), _ops)
                  .sentences()
                  .max_length(10)
                  .build())
        assert [s.text for s in result] == [
            "近年来，人工智能技术",
            "蓬勃发展。",
            "专家认为，这一趋势将",
            "持续加速；然而，也有",
            "学者表达了担忧。",
            "我们需要审慎评估新",
            "技术的风险。",
        ]

    def test_sentences_then_clauses_then_max_length(self) -> None:
        result = (Subtitle(_asr_news_segments(), _ops)
                  .sentences()
                  .clauses()
                  .max_length(8)
                  .build())
        assert [s.text for s in result] == [
            "近年来，",
            "人工智能技术",
            "蓬勃发展。",
            "专家认为，",
            "这一趋势将持续",
            "加速；",
            "然而，",
            "也有学者表达了",
            "担忧。",
            "我们需要审慎评估",
            "新技术的风险。",
        ]

    def test_short_text_no_split(self) -> None:
        result = Subtitle(_short_segments(), _ops).max_length(50).build()
        assert [s.text for s in result] == ["你好世界。今天天气不错！"]

    def test_mixed_language_split(self) -> None:
        result = (Subtitle(_mixed_language_segments(), _ops)
                  .sentences()
                  .max_length(10)
                  .build())
        assert [s.text for s in result] == [
            "我正在学习",
            "Python 编程。",
            "This 是混合测试！",
        ]

    def test_extreme_length_word_fallback(self) -> None:
        result = (Subtitle(_extreme_length_segment(), _ops)
                  .sentences()
                  .max_length(15)
                  .build())
        assert [s.text for s in result] == [
            "这是一个超级长的英文单词",
            "supercalifragilisticexpialidocious",
            "测试",
        ]


# ---------------------------------------------------------------------------
# Merge (greedy bin-packing)
# ---------------------------------------------------------------------------

class TestChineseMerge:

    def test_merge_clauses_back(self) -> None:
        """clauses → merge: small clauses recombined under max_length."""
        clause_result = (Subtitle(_asr_news_segments(), _ops)
                         .sentences().clauses().build())
        merged_result = (Subtitle(_asr_news_segments(), _ops)
                         .sentences().clauses().merge(15).build())
        for seg in merged_result:
            assert _ops.length(seg.text) <= 15, \
                f"Segment too long: {seg.text!r} ({_ops.length(seg.text)})"
        assert len(merged_result) <= len(clause_result)

    def test_merge_preserves_text(self) -> None:
        result = (Subtitle(_asr_news_segments(), _ops)
                  .sentences().clauses().merge(20).build())
        merged_text = "".join(s.text for s in result)
        original_text = "".join(s.text for s in _asr_news_segments())
        assert merged_text == original_text

    def test_merge_exact_results(self) -> None:
        """Clause-rich segment → clauses → merge with known output."""
        # "苹果、香蕉、橘子，都是水果；牛奶、面包，都是早餐。"
        # clauses → ["苹果、香蕉、橘子，", "都是水果；", "牛奶、面包，", "都是早餐。"]
        # merge(12): "苹果、香蕉、橘子，" len=9, +"都是水果；"=13>12 → flush
        #   "都是水果；" len=5, +"牛奶、面包，"=10, +"都是早餐。"=15>12 → flush
        #   "都是水果；牛奶、面包，" len=10, → try +"都是早餐。"=15>12 → flush
        result = (Subtitle(_clause_rich_segment(), _ops)
                  .sentences().clauses().merge(12).build())
        assert [s.text for s in result] == [
            "苹果、香蕉、橘子，",
            "都是水果；牛奶、面包，",
            "都是早餐。",
        ]

    def test_merge_all_fit(self) -> None:
        """When max_length fits everything, merge combines all chunks."""
        result = (Subtitle(_short_segments(), _ops)
                  .sentences().merge(100).build())
        # No group boundaries → merges into 1
        assert len(result) == 1
        assert "你好世界。" in result[0].text
        assert "今天天气不错！" in result[0].text

    def test_merge_nothing_fits(self) -> None:
        result = (Subtitle(_short_segments(), _ops)
                  .sentences().merge(3).build())
        assert len(result) == 2

    def test_merge_words_timing(self) -> None:
        result = (Subtitle(_short_segments(), _ops)
                  .sentences().merge(100).build())
        # Merges into 1 segment spanning all words
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 5.0

    def test_merge_chain_full(self) -> None:
        """Full chain: sentences → clauses → max_length → merge."""
        result = (Subtitle(_asr_news_segments(), _ops)
                  .sentences().clauses().max_length(8).merge(15).build())
        for seg in result:
            assert _ops.length(seg.text) <= 15, f"Too long: {seg.text!r}"
        merged_text = "".join(s.text for s in result)
        original_text = "".join(s.text for s in _asr_news_segments())
        assert merged_text == original_text

    def test_merge_respects_sentence_boundaries(self) -> None:
        """sentences → clauses → merge: merge only within each sentence."""
        proc = Subtitle(_asr_news_segments(), _ops).sentences().clauses()
        merged = proc.merge(200).build()
        # Each sentence's clauses are merged, but sentences stay separate
        sentence_count = len(
            Subtitle(_asr_news_segments(), _ops).sentences().build()
        )
        assert len(merged) == sentence_count


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestChineseRecords:

    def test_records_structure(self) -> None:
        records = Subtitle(_short_segments(), _ops).records()
        assert len(records) == 2
        assert records[0].src_text == "你好世界。"
        assert records[1].src_text == "今天天气不错！"
        assert records[0].start == 0.0
        assert records[0].end == 2.0

    def test_records_with_max_length(self) -> None:
        records = Subtitle(_asr_news_segments(), _ops).records(max_length=8)
        assert [rec.src_text for rec in records] == [
            "近年来，人工智能技术蓬勃发展。",
            "专家认为，这一趋势将持续加速；然而，也有学者表达了担忧。",
            "我们需要审慎评估新技术的风险。",
        ]
        assert [[seg.text for seg in rec.segments] for rec in records] == [
            ["近年来，", "人工智能技术", "蓬勃发展。"],
            ["专家认为，", "这一趋势将持续", "加速；", "然而，", "也有学者表达了", "担忧。"],
            ["我们需要审慎评估", "新技术的风险。"],
        ]
        assert all(len(seg.words) >= 1 for rec in records for seg in rec.segments)


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------

class TestChineseSpeaker:

    def test_speaker_change_creates_boundary(self) -> None:
        result = (Subtitle(_multi_speaker_segments(), _ops, split_by_speaker=True)
                  .sentences()
                  .build())
        assert [s.text for s in result] == ["你觉得怎么样？", "我觉得非常好！", "真的吗？", "当然是真的。"]

    def test_same_speaker_no_extra_splits(self) -> None:
        segments = [S("你好世界。今天不错！", 0.0, 5.0, words=[
            W("你", 0.0, 0.3, speaker="A"), W("好", 0.3, 0.6, speaker="A"),
            W("世", 0.6, 0.9, speaker="A"), W("界", 0.9, 1.5, speaker="A"),
            W("。", 1.5, 2.0, speaker="A"),
            W("今", 2.0, 2.3, speaker="A"), W("天", 2.3, 2.6, speaker="A"),
            W("不", 2.6, 3.0, speaker="A"), W("错", 3.0, 4.0, speaker="A"),
            W("！", 4.0, 5.0, speaker="A"),
        ])]
        result_with = Subtitle(segments, _ops, split_by_speaker=True).sentences().build()
        result_without = Subtitle(segments, _ops).sentences().build()
        expected = ["你好世界。", "今天不错！"]
        assert [s.text for s in result_with] == expected
        assert [s.text for s in result_without] == expected


# ---------------------------------------------------------------------------
# Auto-fill words
# ---------------------------------------------------------------------------

class TestChineseAutoFill:

    def test_no_words_auto_filled(self) -> None:
        segments = [
            S("你好世界。", 0.0, 2.0),
            S("今天天气好。", 2.0, 5.0),
        ]
        result = Subtitle(segments, _ops).sentences().build()
        assert [s.text for s in result] == ["你好世界。", "今天天气好。"]
        assert len(result[0].words) >= 1
        assert len(result[1].words) >= 1


# ---------------------------------------------------------------------------
# Stream mode
# ---------------------------------------------------------------------------

class TestChineseStream:

    def test_stream_incremental(self) -> None:
        stream = Subtitle.stream(_ops)
        all_done: list[Segment] = []
        for seg in _asr_news_segments():
            all_done.extend(stream.feed(seg))
        all_done.extend(stream.flush())
        assert [s.text for s in all_done] == [
            "近年来，人工智能技术蓬勃发展。",
            "专家认为，这一趋势将持续加速；然而，也有学者表达了担忧。",
            "我们需要审慎评估新技术的风险。",
        ]

    def test_stream_flush_empty(self) -> None:
        stream = Subtitle.stream(_ops)
        expected = []
        assert stream.flush() == expected

    def test_stream_single_segment(self) -> None:
        stream = Subtitle.stream(_ops)
        done = stream.feed(S("你好世界。", 0.0, 2.0, words=[
            W("你", 0.0, 0.4), W("好", 0.4, 0.8),
            W("世", 0.8, 1.3), W("界", 1.3, 1.8),
            W("。", 1.8, 2.0),
        ]))
        assert done == []

        rest = stream.flush()
        assert [s.text for s in rest] == ["你好世界。"]
