"""Tests for TextPipeline + align_segments integration — pipeline → timed Segments."""

import pytest

from subtitle import Word
from subtitle.align import align_segments
from lang_ops import LangOps


# ---------------------------------------------------------------------------
# English pipeline
# ---------------------------------------------------------------------------


class TestPipelineSegments:
    def test_sentences_segments(self):
        ops = LangOps.for_language("en")
        words = [
            Word("Hello", 0, 0.5),
            Word("world.", 0.6, 1),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you?", 1.7, 2),
        ]
        chunks = ops.chunk("Hello world. How are you?").sentences().result()
        segs = align_segments(chunks, words)
        assert len(segs) == 2
        assert segs[0].text == "Hello world."
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert segs[1].text == "How are you?"
        assert segs[1].start == pytest.approx(1.1)
        assert segs[1].end == pytest.approx(2.0)

    def test_clauses_segments(self):
        ops = LangOps.for_language("en")
        words = [
            Word("Well,", 0, 0.5),
            Word("hello", 0.6, 1),
            Word("world.", 1.1, 1.5),
            Word("How", 1.6, 1.8),
            Word("are", 1.9, 2.1),
            Word("you?", 2.2, 2.5),
        ]
        chunks = ops.chunk("Well, hello world. How are you?").clauses().result()
        segs = align_segments(chunks, words)
        assert len(segs) == 3
        assert segs[0].text == "Well,"
        assert segs[1].text == "hello world."
        assert segs[2].text == "How are you?"


# ---------------------------------------------------------------------------
# Multilingual parametrized pipeline tests
# ---------------------------------------------------------------------------

PIPELINE_CASES = {
    "zh": (
        "zh",
        "你好。再见。",
        [
            Word("你", 0, 0.2),
            Word("好", 0.2, 0.4),
            Word("。", 0.4, 0.5),
            Word("再", 0.5, 0.7),
            Word("见", 0.7, 0.9),
            Word("。", 0.9, 1.0),
        ],
        2,
        ["你好。", "再见。"],
    ),
    "ko": (
        "ko",
        "안녕하세요. 반갑습니다.",
        [Word("안녕하세요.", 0, 1), Word("반갑습니다.", 1.5, 2.5)],
        2,
        None,
    ),  # None = skip text check, verify count + timing
}


@pytest.mark.parametrize(
    "lang,text,words,expected_count,expected_texts",
    PIPELINE_CASES.values(),
    ids=PIPELINE_CASES.keys(),
)
def test_pipeline_segments_multilingual(lang, text, words, expected_count, expected_texts):
    ops = LangOps.for_language(lang)
    chunks = ops.chunk(text).sentences().result()
    segs = align_segments(chunks, words)
    assert len(segs) == expected_count
    if expected_texts is not None:
        assert [s.text for s in segs] == expected_texts
    for i in range(1, len(segs)):
        assert segs[i].start >= segs[i - 1].start
