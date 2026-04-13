"""Tests for subtitle.align — word-level timing utilities."""

import pytest

from subtitle import Word, Segment, fill_words, find_words, distribute_words, align_segments, attach_punct_words
from subtitle.align import _strip_punct


# ---------------------------------------------------------------------------
# _strip_punct
# ---------------------------------------------------------------------------

class TestStripPunct:
    def test_strip_punct(self):
        def test_no_punct():
            assert _strip_punct("hello") == "hello"

        def test_leading():
            assert _strip_punct("...hello") == "hello"

        def test_trailing():
            assert _strip_punct("hello!") == "hello"

        def test_both():
            assert _strip_punct('"world"') == "world"

        def test_all_punct():
            assert _strip_punct("...") == ""

        def test_middle_dot():
            assert _strip_punct("deeplearning.ai") == "deeplearning.ai"

        def test_middle_apostrophe():
            assert _strip_punct("I'm") == "I'm"

        def test_url():
            assert _strip_punct("https://www.deeplearning.ai") == "https://www.deeplearning.ai"

        def test_empty():
            assert _strip_punct("") == ""
        
        test_no_punct()
        test_leading()
        test_trailing()
        test_both()
        test_all_punct()
        test_middle_dot()
        test_middle_apostrophe()
        test_url()
        test_empty()


# ---------------------------------------------------------------------------
# attach_punct_words
# ---------------------------------------------------------------------------

class TestAttachPunctWords:
    def test_attach_punct_words(self):
        def test_trailing_punct_attaches_to_prev():
            words = [
                Word("Hello", 0.0, 0.5),
                Word(",", 0.5, 0.55),
                Word("world", 0.6, 1.0),
            ]
            result = attach_punct_words(words)
            assert [w.word for w in result] == ["Hello,", "world"]
            assert result[0].end == 0.55
            assert result[1].start == 0.6

        def test_opening_punct_attaches_to_next():
            words = [
                Word("(", 0.0, 0.1),
                Word("hello", 0.1, 0.5),
                Word(")", 0.5, 0.6),
            ]
            result = attach_punct_words(words)
            assert [w.word for w in result] == ["(hello)"]
            assert result[0].start == 0.0
            assert result[0].end == 0.6

        def test_no_punct_returns_same_list():
            words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
            result = attach_punct_words(words)
            assert result is words

        def test_empty_returns_same():
            result = attach_punct_words([])
            assert result == []

        def test_cjk_sentence_end():
            words = [
                Word("你好", 0.0, 0.5),
                Word("。", 0.5, 0.55),
                Word("再见", 0.6, 1.0),
            ]
            result = attach_punct_words(words)
            assert [w.word for w in result] == ["你好。", "再见"]
            assert result[0].end == 0.55

        def test_multiple_trailing_punct():
            words = [
                Word("Really", 0.0, 0.5),
                Word("?", 0.5, 0.55),
                Word("!", 0.55, 0.6),
            ]
            result = attach_punct_words(words)
            assert [w.word for w in result] == ["Really?!"]
            assert result[0].end == 0.6

        def test_all_punct_returns_as_is():
            words = [Word(",", 0.0, 0.1), Word(".", 0.1, 0.2)]
            result = attach_punct_words(words)
            # All punct — closing attaches to... nothing before first,
            # so first stays, second attaches to first
            assert len(result) == 1
            assert result[0].word == ",."

        def test_whisper_style_leading_space():
            words = [
                Word(" Hello", 0.0, 0.5),
                Word(",", 0.5, 0.55),
                Word(" world", 0.6, 1.0),
            ]
            result = attach_punct_words(words)
            assert [w.word for w in result] == [" Hello,", " world"]

        test_trailing_punct_attaches_to_prev()
        test_opening_punct_attaches_to_next()
        test_no_punct_returns_same_list()
        test_empty_returns_same()
        test_cjk_sentence_end()
        test_multiple_trailing_punct()
        test_all_punct_returns_as_is()
        test_whisper_style_leading_space()

class TestFillWords:
    def test_fill_words(self):
        def test_already_has_words():
            w = [Word("Hi", 0.0, 1.0)]
            seg = Segment(start=0.0, end=1.0, text="Hi", words=w)
            result = fill_words(seg)
            assert result.words is w  # unchanged

        def test_whitespace_split():
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

        def test_unequal_tokens():
            seg = Segment(start=0.0, end=10.0, text="I am fine")
            result = fill_words(seg)
            assert len(result.words) == 3
            # "I"(1) + "am"(2) + "fine"(4) = 7 chars
            assert result.words[0].word == "I"
            assert result.words[0].end == pytest.approx(10.0 * 1 / 7)
            assert result.words[2].word == "fine"
            assert result.words[2].end == pytest.approx(10.0)

        def test_custom_split_fn():
            seg = Segment(start=0.0, end=2.0, text="你好世界")
            result = fill_words(seg, split_fn=lambda t: list(t))
            assert len(result.words) == 4
            assert result.words[0].word == "你"
            assert result.words[3].word == "界"

        def test_attached_punctuation_tokens():
            seg = Segment(start=0.0, end=6.0, text="Hello, world!")
            result = fill_words(seg)
            assert [w.word for w in result.words] == ["Hello,", "world!"]
            assert result.words[0].start == pytest.approx(0.0)
            assert result.words[0].end == pytest.approx(3.0)
            assert result.words[1].start == pytest.approx(3.0)
            assert result.words[1].end == pytest.approx(6.0)

        def test_custom_split_fn_attaches_punctuation():
            seg = Segment(start=0.0, end=6.0, text="你好，世界！")
            result = fill_words(seg, split_fn=lambda t: list(t))
            assert [w.word for w in result.words] == ["你", "好，", "世", "界！"]
            assert result.words[0].start == pytest.approx(0.0)
            assert result.words[-1].end == pytest.approx(6.0)

        def test_empty_text():
            seg = Segment(start=0.0, end=1.0, text="")
            result = fill_words(seg)
            assert result.words == []

        def test_original_segment_unchanged():
            seg = Segment(start=0.0, end=1.0, text="Hi there")
            result = fill_words(seg)
            assert seg.words == []  # original not mutated
            assert len(result.words) == 2
        
        test_already_has_words()
        test_whitespace_split()
        test_unequal_tokens()
        test_custom_split_fn()
        test_attached_punctuation_tokens()
        test_custom_split_fn_attaches_punctuation()
        test_empty_text()
        test_original_segment_unchanged()


# ---------------------------------------------------------------------------
# find_words
# ---------------------------------------------------------------------------

class TestFindWords:
    def test_find_words(self):
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]

        def test_exact_and_offset_matches():
            assert find_words(words, "Hello world") == (0, 2)
            assert find_words(words, "How are you", start=2) == (2, 5)
            assert find_words(words, "Hello") == (0, 1)
            assert find_words(words, "How", start=2) == (2, 3)

        def test_punctuation_matching():
            assert find_words(words, "Hello, world.") == (0, 2)
            punct_words = [Word("Hello,", 0.0, 0.5), Word("world!", 0.6, 1.0)]
            assert find_words(punct_words, "Hello world") == (0, 2)

        def test_case_and_whitespace_tolerance():
            case_words = [Word("hello", 0.0, 0.5), Word("WORLD", 0.6, 1.0)]
            assert find_words(case_words, "Hello World") == (0, 2)
            whisper_words = [Word(" Hello", 0.0, 0.5), Word(" world", 0.6, 1.0)]
            assert find_words(whisper_words, "Hello world") == (0, 2)
            assert find_words(case_words, "Hello, World!") == (0, 2)

        def test_chinese_char_level_matching():
            zh_words = [
                Word("你", 0.0, 0.2), Word("好", 0.2, 0.4),
                Word("世", 0.4, 0.6), Word("界", 0.6, 0.8),
            ]
            assert find_words(zh_words, "你好") == (0, 2)
            assert find_words(zh_words, "世界", start=2) == (2, 4)

            zh_punct_words = [
                Word("你", 0.0, 0.2), Word("好", 0.2, 0.4),
                Word("。", 0.4, 0.5),
                Word("再", 0.5, 0.7), Word("见", 0.7, 0.9),
            ]
            assert find_words(zh_punct_words, "你好。") == (0, 3)
            assert find_words(zh_punct_words, "再见", start=3) == (3, 5)

        def test_edge_cases():
            assert find_words(words, "xyz") == (0, 0)
            assert find_words(words, "") == (0, 0)
            assert find_words(words, "   ") == (0, 0)
            assert find_words([], "Hello") == (0, 0)
            assert find_words(words, "Hello", start=10) == (10, 10)

        test_exact_and_offset_matches()
        test_punctuation_matching()
        test_case_and_whitespace_tolerance()
        test_chinese_char_level_matching()
        test_edge_cases()


# ---------------------------------------------------------------------------
# distribute_words
# ---------------------------------------------------------------------------

class TestDistributeWords:
    def test_distribute_words(self):
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]

        def test_basic_distribution():
            texts = ["Hello world.", "How are you?"]
            groups = distribute_words(words, texts)
            assert len(groups) == 2
            assert [w.word for w in groups[0]] == ["Hello", "world"]
            assert [w.word for w in groups[1]] == ["How", "are", "you"]

        def test_single_piece_and_empty_texts():
            hi_words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
            groups = distribute_words(hi_words, ["Hi there"])
            assert len(groups) == 1
            assert len(groups[0]) == 2

            assert distribute_words([Word("Hi", 0.0, 0.5)], []) == []

        def test_timing_from_groups():
            timed_words = [
                Word("A", 1.0, 1.5),
                Word("B", 2.0, 2.5),
                Word("C", 3.0, 3.5),
            ]
            groups = distribute_words(timed_words, ["A B", "C"])
            assert groups[0][0].start == pytest.approx(1.0)
            assert groups[0][-1].end == pytest.approx(2.5)
            assert groups[1][0].start == pytest.approx(3.0)
            assert groups[1][-1].end == pytest.approx(3.5)

        def test_end_to_end_with_fill():
            seg = Segment(start=0.0, end=10.0, text="Hello world. How are you?")
            filled = fill_words(seg)
            groups = distribute_words(filled.words, ["Hello world.", "How are you?"])
            assert len(groups) == 2
            assert groups[0][0].start == pytest.approx(0.0)
            assert groups[1][-1].end == pytest.approx(10.0)

        test_basic_distribution()
        test_single_piece_and_empty_texts()
        test_timing_from_groups()
        test_end_to_end_with_fill()


# ---------------------------------------------------------------------------
# align_segments
# ---------------------------------------------------------------------------

class TestAlignSegments:
    def test_align_segments(self):
        words = [
            Word("Hello", 0.0, 0.5),
            Word("world", 0.6, 1.0),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2.0),
        ]

        def test_basic_alignment():
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

        def test_single_chunk_and_empty_cases():
            hi_words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
            segs = align_segments(["Hi there"], hi_words)
            assert len(segs) == 1
            assert segs[0].start == pytest.approx(0.0)
            assert segs[0].end == pytest.approx(1.0)
            assert len(segs[0].words) == 2

            assert align_segments([], [Word("Hi", 0.0, 0.5)]) == []

            no_word_segs = align_segments(["Hello world"], [])
            assert len(no_word_segs) == 1
            assert no_word_segs[0].text == "Hello world"
            assert no_word_segs[0].start == 0.0
            assert no_word_segs[0].end == 0.0
            assert no_word_segs[0].words == []

        def test_end_to_end_with_fill():
            seg = Segment(start=0.0, end=10.0, text="Hello world. How are you?")
            filled = fill_words(seg)
            result = align_segments(["Hello world.", "How are you?"], filled.words)
            assert len(result) == 2
            assert result[0].start == pytest.approx(0.0)
            assert result[1].end == pytest.approx(10.0)
            assert result[0].text == "Hello world."
            assert result[1].text == "How are you?"

        test_basic_alignment()
        test_single_chunk_and_empty_cases()
        test_end_to_end_with_fill()


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
