"""Tests for subtitle.words — word-level timing utilities."""

import pytest

from subtitle import Word, Segment, fill_words, find_words, distribute_words, align_segments
from subtitle.words import _strip_punct


# ---------------------------------------------------------------------------
# _strip_punct
# ---------------------------------------------------------------------------

class TestStripPunct:
    def test_no_punct(self):
        assert _strip_punct("hello") == "hello"

    def test_leading(self):
        assert _strip_punct("...hello") == "hello"

    def test_trailing(self):
        assert _strip_punct("hello!") == "hello"

    def test_both(self):
        assert _strip_punct('"world"') == "world"

    def test_all_punct(self):
        assert _strip_punct("...") == ""

    def test_empty(self):
        assert _strip_punct("") == ""


# ---------------------------------------------------------------------------
# fill_words
# ---------------------------------------------------------------------------

class TestFillWords:
    def test_already_has_words(self):
        w = [Word("Hi", 0.0, 1.0)]
        seg = Segment(start=0.0, end=1.0, text="Hi", words=w)
        result = fill_words(seg)
        assert result.words is w  # unchanged

    def test_whitespace_split(self):
        seg = Segment(start=0.0, end=4.0, text="Hello world")
        result = fill_words(seg)
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[1].word == "world"
        # Proportional: "Hello"(5) + "world"(5) → equal split
        assert result.words[0].start == pytest.approx(0.0)
        assert result.words[0].end == pytest.approx(2.0)
        assert result.words[1].start == pytest.approx(2.0)
        assert result.words[1].end == pytest.approx(4.0)

    def test_unequal_tokens(self):
        seg = Segment(start=0.0, end=10.0, text="I am fine")
        result = fill_words(seg)
        assert len(result.words) == 3
        # "I"(1) + "am"(2) + "fine"(4) = 7 chars
        assert result.words[0].word == "I"
        assert result.words[0].end == pytest.approx(10.0 * 1 / 7)
        assert result.words[2].word == "fine"
        assert result.words[2].end == pytest.approx(10.0)

    def test_custom_split_fn(self):
        seg = Segment(start=0.0, end=2.0, text="你好世界")
        result = fill_words(seg, split_fn=lambda t: list(t))
        assert len(result.words) == 4
        assert result.words[0].word == "你"
        assert result.words[3].word == "界"

    def test_empty_text(self):
        seg = Segment(start=0.0, end=1.0, text="")
        result = fill_words(seg)
        assert result.words == []

    def test_original_segment_unchanged(self):
        seg = Segment(start=0.0, end=1.0, text="Hi there")
        result = fill_words(seg)
        assert seg.words == []  # original not mutated
        assert len(result.words) == 2


# ---------------------------------------------------------------------------
# find_words
# ---------------------------------------------------------------------------

class TestFindWords:
    def setup_method(self):
        self.words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]

    def test_exact_match(self):
        assert find_words(self.words, "Hello world") == (0, 2)

    def test_with_start_offset(self):
        assert find_words(self.words, "How are you", start=2) == (2, 5)

    def test_punct_in_text_not_word(self):
        """Text has punctuation, ASR words don't (most common case)."""
        assert find_words(self.words, "Hello, world.") == (0, 2)

    def test_punct_in_word_not_text(self):
        """ASR word has punctuation, text doesn't."""
        words = [Word("Hello,", 0.0, 0.5), Word("world!", 0.6, 1.0)]
        assert find_words(words, "Hello world") == (0, 2)

    def test_single_word(self):
        assert find_words(self.words, "Hello") == (0, 1)
        assert find_words(self.words, "How", start=2) == (2, 3)

    def test_no_match(self):
        assert find_words(self.words, "xyz") == (0, 0)

    def test_empty_text(self):
        assert find_words(self.words, "") == (0, 0)
        assert find_words(self.words, "   ") == (0, 0)

    def test_empty_words(self):
        assert find_words([], "Hello") == (0, 0)

    def test_start_beyond_end(self):
        assert find_words(self.words, "Hello", start=10) == (10, 10)

    def test_case_insensitive_match(self):
        """Words with different casing than text."""
        words = [Word("hello", 0.0, 0.5), Word("WORLD", 0.6, 1.0)]
        assert find_words(words, "Hello World") == (0, 2)

    def test_whisper_leading_space(self):
        """Whisper-style tokens with leading whitespace."""
        words = [Word(" Hello", 0.0, 0.5), Word(" world", 0.6, 1.0)]
        assert find_words(words, "Hello world") == (0, 2)

    def test_case_and_punct_combined(self):
        """Both casing and punctuation mismatch."""
        words = [Word("hello", 0.0, 0.5), Word("world", 0.6, 1.0)]
        assert find_words(words, "Hello, World!") == (0, 2)

    def test_chinese_char_level_words(self):
        """Real-world: Chinese words are single characters."""
        words = [
            Word("你", 0.0, 0.2), Word("好", 0.2, 0.4),
            Word("世", 0.4, 0.6), Word("界", 0.6, 0.8),
        ]
        assert find_words(words, "你好") == (0, 2)
        assert find_words(words, "世界", start=2) == (2, 4)

    def test_chinese_char_level_with_punct(self):
        """Chinese char words matched against text with punctuation."""
        words = [
            Word("你", 0.0, 0.2), Word("好", 0.2, 0.4),
            Word("。", 0.4, 0.5),
            Word("再", 0.5, 0.7), Word("见", 0.7, 0.9),
        ]
        assert find_words(words, "你好。") == (0, 3)
        assert find_words(words, "再见", start=3) == (3, 5)


# ---------------------------------------------------------------------------
# distribute_words
# ---------------------------------------------------------------------------

class TestDistributeWords:
    def test_two_sentences(self):
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]
        texts = ["Hello world.", "How are you?"]
        groups = distribute_words(words, texts)
        assert len(groups) == 2
        assert [w.word for w in groups[0]] == ["Hello", "world"]
        assert [w.word for w in groups[1]] == ["How", "are", "you"]

    def test_single_piece(self):
        words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
        groups = distribute_words(words, ["Hi there"])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_empty_texts(self):
        words = [Word("Hi", 0.0, 0.5)]
        groups = distribute_words(words, [])
        assert groups == []

    def test_timing_from_groups(self):
        """Demonstrate the typical usage: get timing from word groups."""
        words = [
            Word("A", 1.0, 1.5),
            Word("B", 2.0, 2.5),
            Word("C", 3.0, 3.5),
        ]
        groups = distribute_words(words, ["A B", "C"])
        assert groups[0][0].start == pytest.approx(1.0)
        assert groups[0][-1].end == pytest.approx(2.5)
        assert groups[1][0].start == pytest.approx(3.0)
        assert groups[1][-1].end == pytest.approx(3.5)

    def test_end_to_end_with_fill(self):
        """Full workflow: fill_words → split → distribute."""
        seg = Segment(start=0.0, end=10.0, text="Hello world. How are you?")
        seg = fill_words(seg)
        sentences = ["Hello world.", "How are you?"]
        groups = distribute_words(seg.words, sentences)
        assert len(groups) == 2
        assert groups[0][0].start == pytest.approx(0.0)
        assert groups[1][-1].end == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# align_segments
# ---------------------------------------------------------------------------

class TestAlignSegments:
    def test_basic(self):
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]
        chunks = ["Hello world.", "How are you?"]
        segs = align_segments(chunks, words)
        assert len(segs) == 2
        assert segs[0].text == "Hello world."
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert [w.word for w in segs[0].words] == ["Hello", "world"]
        assert segs[1].text == "How are you?"
        assert segs[1].start == pytest.approx(1.1)
        assert segs[1].end == pytest.approx(2.0)
        assert [w.word for w in segs[1].words] == ["How", "are", "you"]

    def test_single_chunk(self):
        words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
        segs = align_segments(["Hi there"], words)
        assert len(segs) == 1
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert len(segs[0].words) == 2

    def test_empty_chunks(self):
        words = [Word("Hi", 0.0, 0.5)]
        segs = align_segments([], words)
        assert segs == []

    def test_no_words(self):
        segs = align_segments(["Hello world"], [])
        assert len(segs) == 1
        assert segs[0].text == "Hello world"
        assert segs[0].start == 0.0
        assert segs[0].end == 0.0
        assert segs[0].words == []

    def test_end_to_end_with_fill(self):
        """Full workflow: fill_words → pipeline → align_segments."""
        seg = Segment(start=0.0, end=10.0, text="Hello world. How are you?")
        seg = fill_words(seg)
        chunks = ["Hello world.", "How are you?"]
        result = align_segments(chunks, seg.words)
        assert len(result) == 2
        assert result[0].start == pytest.approx(0.0)
        assert result[1].end == pytest.approx(10.0)
        assert result[0].text == "Hello world."
        assert result[1].text == "How are you?"


# ---------------------------------------------------------------------------
# Pipeline .segments() integration
# ---------------------------------------------------------------------------

class TestPipelineSegments:
    def test_sentences_segments(self):
        """Pipeline .sentences().segments(words) end-to-end."""
        from lang_ops import LangOps
        ops = LangOps.for_language("en")
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world.", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you?", 1.7, 2.0),
        ]
        text = "Hello world. How are you?"
        segs = ops.chunk(text).sentences().segments(words)
        assert len(segs) == 2
        assert segs[0].text == "Hello world."
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert segs[1].text == "How are you?"
        assert segs[1].start == pytest.approx(1.1)
        assert segs[1].end == pytest.approx(2.0)

    def test_clauses_segments(self):
        """Pipeline .clauses().segments(words) end-to-end."""
        from lang_ops import LangOps
        ops = LangOps.for_language("en")
        words = [
            Word("Well,", 0.0, 0.5),
            Word("hello", 0.6, 1.0),
            Word("world.", 1.1, 1.5),
            Word("How", 1.6, 1.8),
            Word("are", 1.9, 2.1),
            Word("you?", 2.2, 2.5),
        ]
        text = "Well, hello world. How are you?"
        segs = ops.chunk(text).clauses().segments(words)
        assert len(segs) == 3
        assert segs[0].text == "Well,"
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(0.5)
        assert segs[1].text == "hello world."
        assert segs[1].start == pytest.approx(0.6)
        assert segs[1].end == pytest.approx(1.5)
        assert segs[2].text == "How are you?"
        assert segs[2].start == pytest.approx(1.6)
        assert segs[2].end == pytest.approx(2.5)
