"""Tests for SegmentBuilder — segment restructuring."""

from __future__ import annotations

import pytest
from subtitle import Segment, Word, SentenceRecord, SegmentBuilder
from lang_ops import TextOps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(text: str, start: float, end: float, speaker: str | None = None) -> Word:
    return Word(word=text, start=start, end=end, speaker=speaker)


def _seg(text: str, start: float, end: float, words: list[Word] | None = None) -> Segment:
    return Segment(start=start, end=end, text=text, words=words or [])


_en = TextOps.for_language("en")


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _make_en_segments() -> list[Segment]:
    """Two segments whose combined text has two sentences."""
    return [
        _seg("Hello world.", 0.0, 2.0, words=[
            _word("Hello", 0.0, 0.8),
            _word("world.", 0.9, 2.0),
        ]),
        _seg("How are you?", 2.5, 5.0, words=[
            _word("How", 2.5, 3.0),
            _word("are", 3.1, 3.5),
            _word("you?", 3.6, 5.0),
        ]),
    ]


def _make_en_multi_sentence() -> list[Segment]:
    """Three sentences split across two segments."""
    return [
        _seg("First sentence. Second", 0.0, 4.0, words=[
            _word("First", 0.0, 1.0),
            _word("sentence.", 1.0, 2.0),
            _word("Second", 2.5, 4.0),
        ]),
        _seg("sentence. Third sentence.", 4.0, 8.0, words=[
            _word("sentence.", 4.0, 5.5),
            _word("Third", 5.5, 6.5),
            _word("sentence.", 6.5, 8.0),
        ]),
    ]


# ---------------------------------------------------------------------------
# SegmentBuilder — batch operations
# ---------------------------------------------------------------------------


class TestSegmentBuilderSentences:
    """Test .sentences().build()"""

    def test_basic_two_sentences(self) -> None:
        segments = _make_en_segments()
        result = SegmentBuilder(segments, _en).sentences().build()

        assert len(result) == 2
        assert result[0].text == "Hello world."
        assert result[1].text == " How are you?"
        # Timing from words
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 2.5
        assert result[1].end == 5.0

    def test_sentences_across_segment_boundaries(self) -> None:
        segments = _make_en_multi_sentence()
        result = SegmentBuilder(segments, _en).sentences().build()

        assert len(result) == 3
        assert result[0].text == "First sentence."
        assert result[1].text == " Second sentence."
        assert result[2].text == " Third sentence."
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[2].start == 5.5
        assert result[2].end == 8.0

    def test_words_are_preserved(self) -> None:
        segments = _make_en_segments()
        result = SegmentBuilder(segments, _en).sentences().build()

        assert len(result[0].words) == 2
        assert result[0].words[0].word == "Hello"
        assert result[0].words[1].word == "world."
        assert len(result[1].words) == 3

    def test_single_segment_single_sentence(self) -> None:
        segments = [_seg("Hello world.", 0.0, 2.0, words=[
            _word("Hello", 0.0, 1.0),
            _word("world.", 1.0, 2.0),
        ])]
        result = SegmentBuilder(segments, _en).sentences().build()

        assert len(result) == 1
        assert result[0].text == "Hello world."

    def test_empty_input(self) -> None:
        result = SegmentBuilder([], _en).sentences().build()
        assert result == []


class TestSegmentBuilderClauses:
    """Test .clauses().build()"""

    def test_clauses_split(self) -> None:
        segments = [_seg("First clause, second clause.", 0.0, 4.0, words=[
            _word("First", 0.0, 0.8),
            _word("clause,", 0.8, 1.5),
            _word("second", 1.5, 2.5),
            _word("clause.", 2.5, 4.0),
        ])]
        result = SegmentBuilder(segments, _en).clauses().build()

        assert len(result) == 2
        assert result[0].text == "First clause,"
        assert result[1].text == " second clause."

    def test_sentence_then_clauses(self) -> None:
        segments = [_seg("Hello world. First, second.", 0.0, 6.0, words=[
            _word("Hello", 0.0, 0.8),
            _word("world.", 0.8, 2.0),
            _word("First,", 2.5, 3.5),
            _word("second.", 3.5, 6.0),
        ])]
        result = SegmentBuilder(segments, _en).sentences().clauses().build()

        assert len(result) == 3
        assert result[0].text == "Hello world."
        assert result[1].text == " First,"
        assert result[2].text == " second."


class TestSegmentBuilderByLength:
    """Test .by_length().build()"""

    def test_by_length(self) -> None:
        segments = [_seg("Hello world. How are you today?", 0.0, 6.0, words=[
            _word("Hello", 0.0, 1.0),
            _word("world.", 1.0, 2.0),
            _word("How", 2.5, 3.0),
            _word("are", 3.0, 3.5),
            _word("you", 3.5, 4.5),
            _word("today?", 4.5, 6.0),
        ])]
        result = SegmentBuilder(segments, _en).by_length(15).build()

        assert all(len(seg.text) <= 15 or len(seg.text.split()) == 1
                   for seg in result)
        # Verify all text is covered
        full_text = "".join(seg.text for seg in result)
        assert full_text.replace(" ", "") == "Hello world. How are you today?".replace(" ", "")

    def test_chain_sentences_then_length(self) -> None:
        segments = _make_en_segments()
        result = (SegmentBuilder(segments, _en)
                  .sentences()
                  .by_length(8)
                  .build())

        # Each chunk should be ≤ 8 chars (or a single oversized token)
        for seg in result:
            assert _en.length(seg.text) <= 8 or len(_en.split(seg.text)) == 1


class TestSegmentBuilderRecords:
    """Test .records()"""

    def test_basic_records(self) -> None:
        segments = _make_en_segments()
        records = SegmentBuilder(segments, _en).records()

        assert len(records) == 2
        assert isinstance(records[0], SentenceRecord)
        assert records[0].src_text == "Hello world."
        assert records[1].src_text == " How are you?"
        # Without max_length, each record has one segment = the sentence itself
        assert len(records[0].segments) == 1
        assert records[0].segments[0].text == "Hello world."

    def test_records_with_max_length(self) -> None:
        segments = [_seg("Hello world. This is a long second sentence here.", 0.0, 8.0, words=[
            _word("Hello", 0.0, 0.8),
            _word("world.", 0.8, 2.0),
            _word("This", 2.5, 3.0),
            _word("is", 3.0, 3.5),
            _word("a", 3.5, 3.8),
            _word("long", 3.8, 4.5),
            _word("second", 4.5, 5.5),
            _word("sentence", 5.5, 7.0),
            _word("here.", 7.0, 8.0),
        ])]
        records = SegmentBuilder(segments, _en).records(max_length=15)

        assert len(records) == 2
        # First sentence fits in 15 chars
        assert records[0].src_text == "Hello world."
        assert len(records[0].segments) == 1
        # Second sentence is longer, should be sub-split
        assert records[1].src_text == " This is a long second sentence here."
        assert len(records[1].segments) >= 2
        # All sub-segments have timing
        for seg in records[1].segments:
            assert seg.words

    def test_records_timing(self) -> None:
        segments = _make_en_segments()
        records = SegmentBuilder(segments, _en).records()

        assert records[0].start == 0.0
        assert records[0].end == 2.0
        assert records[1].start == 2.5
        assert records[1].end == 5.0


class TestSegmentBuilderNoWords:
    """Test that segments without words get auto-filled."""

    def test_auto_fill_words(self) -> None:
        segments = [
            _seg("Hello world.", 0.0, 2.0),  # no words
            _seg("How are you?", 2.5, 5.0),  # no words
        ]
        result = SegmentBuilder(segments, _en).sentences().build()

        assert len(result) == 2
        assert result[0].text == "Hello world."
        assert result[1].text == " How are you?"
        # Words should have been interpolated
        assert len(result[0].words) >= 1
        assert len(result[1].words) >= 1
        # Timing should be reasonable
        assert result[0].start >= 0.0
        assert result[1].end <= 5.0


class TestSegmentBuilderSpeaker:
    """Test split_by_speaker."""

    def test_speaker_change_splits(self) -> None:
        segments = [_seg("Hello world. How are you?", 0.0, 5.0, words=[
            _word("Hello", 0.0, 0.8, speaker="A"),
            _word("world.", 0.8, 2.0, speaker="A"),
            _word("How", 2.5, 3.0, speaker="B"),
            _word("are", 3.0, 3.5, speaker="B"),
            _word("you?", 3.5, 5.0, speaker="B"),
        ])]
        result = (SegmentBuilder(segments, _en, split_by_speaker=True)
                  .sentences()
                  .build())

        # Speaker change should force a break even within same chunk
        assert len(result) >= 2

    def test_no_speaker_change(self) -> None:
        segments = [_seg("Hello world.", 0.0, 2.0, words=[
            _word("Hello", 0.0, 1.0, speaker="A"),
            _word("world.", 1.0, 2.0, speaker="A"),
        ])]
        result = (SegmentBuilder(segments, _en, split_by_speaker=True)
                  .sentences()
                  .build())

        assert len(result) == 1


class TestSegmentBuilderImmutability:
    """Verify that chaining creates new builders."""

    def test_chaining_does_not_mutate(self) -> None:
        segments = _make_en_segments()
        builder = SegmentBuilder(segments, _en)
        b1 = builder.sentences()
        b2 = builder.clauses()

        r1 = b1.build()
        r2 = b2.build()
        # Both should work independently
        assert len(r1) >= 1
        assert len(r2) >= 1
        # Original builder still works
        r0 = builder.build()
        assert len(r0) == 1  # single merged chunk


# ---------------------------------------------------------------------------
# Stream mode
# ---------------------------------------------------------------------------

class TestStreamBuilder:
    """Test SegmentBuilder.stream()"""

    def test_stream_basic(self) -> None:
        stream = SegmentBuilder.stream(_en)

        # Feed first segment — not enough to confirm sentence
        done = stream.feed(_seg("Hello world.", 0.0, 2.0, words=[
            _word("Hello", 0.0, 0.8),
            _word("world.", 0.8, 2.0),
        ]))
        # May or may not emit depending on whether second segment confirms it
        # (single sentence, nothing confirmed yet)

        done2 = stream.feed(_seg("How are you?", 2.5, 5.0, words=[
            _word("How", 2.5, 3.0),
            _word("are", 3.0, 3.5),
            _word("you?", 3.5, 5.0),
        ]))

        # First sentence should be confirmed
        assert len(done2) >= 1
        assert "Hello world." in done2[0].text

        # Flush remaining
        rest = stream.flush()
        assert len(rest) >= 1

    def test_stream_flush_empty(self) -> None:
        stream = SegmentBuilder.stream(_en)
        assert stream.flush() == []

    def test_stream_single_segment_flush(self) -> None:
        stream = SegmentBuilder.stream(_en)
        done = stream.feed(_seg("Hello world.", 0.0, 2.0, words=[
            _word("Hello", 0.0, 1.0),
            _word("world.", 1.0, 2.0),
        ]))
        assert done == []  # Can't confirm without more input

        rest = stream.flush()
        assert len(rest) == 1
        assert rest[0].text == "Hello world."


# ---------------------------------------------------------------------------
# CJK
# ---------------------------------------------------------------------------

class TestSegmentBuilderCJK:
    """Test with Chinese ops."""

    def test_zh_sentences(self) -> None:
        zh = TextOps.for_language("zh")
        segments = [
            _seg("你好世界。", 0.0, 2.0, words=[
                _word("你好", 0.0, 1.0),
                _word("世界", 1.0, 1.8),
                _word("。", 1.8, 2.0),
            ]),
            _seg("今天天气不错！", 2.0, 5.0, words=[
                _word("今天", 2.0, 2.8),
                _word("天气", 2.8, 3.5),
                _word("不错", 3.5, 4.5),
                _word("！", 4.5, 5.0),
            ]),
        ]
        result = SegmentBuilder(segments, zh).sentences().build()

        assert len(result) == 2
        assert result[0].text == "你好世界。"
        assert result[1].text == "今天天气不错！"
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 2.0
        assert result[1].end == 5.0

    def test_zh_no_space_join(self) -> None:
        """CJK segments should be joined without spaces."""
        zh = TextOps.for_language("zh")
        segments = [
            _seg("你好", 0.0, 1.0, words=[_word("你好", 0.0, 1.0)]),
            _seg("世界", 1.0, 2.0, words=[_word("世界", 1.0, 2.0)]),
        ]
        result = SegmentBuilder(segments, zh).build()

        assert len(result) == 1
        assert result[0].text == "你好世界"
