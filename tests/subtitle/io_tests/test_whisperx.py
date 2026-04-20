"""Tests for WhisperX word-level sanitization."""

from __future__ import annotations

import pytest

from model import Word
from subtitle.io.whisperx import (
    sanitize_whisperx,
    parse_whisperx,
    _dedup_untimed,
    _interpolate_timestamps,
    _attach_punctuation,
    _collapse_repeats,
    _replace_long_words,
)


# ── helpers ───────────────────────────────────────────────────────────


def _w(word: str, start=None, end=None, score=0.5) -> dict:
    """Shorthand for a word dict.  Omit start/end for untimed."""
    d: dict = {"word": word}
    if start is not None:
        d["start"] = start
        d["end"] = end
        d["score"] = score
    return d


# ── _dedup_untimed ────────────────────────────────────────────────────


class TestDedupUntimed:
    def test_empty(self):
        assert _dedup_untimed([]) == []

    def test_no_duplicates(self):
        ws = [_w("hello", 0, 1), _w("world", 1, 2)]
        assert len(_dedup_untimed(ws)) == 2

    def test_timed_duplicates_kept(self):
        ws = [_w("the", 0, 0.5), _w("the", 0.5, 1)]
        assert len(_dedup_untimed(ws)) == 2

    def test_untimed_duplicates_collapsed(self):
        """♪ ♪ ♪ ♪ → ♪"""
        ws = [_w("♪"), _w("♪"), _w("♪"), _w("♪")]
        result = _dedup_untimed(ws)
        assert len(result) == 1
        assert result[0]["word"] == "♪"

    def test_mixed_untimed_different_words_kept(self):
        ws = [_w("1."), _w("5"), _w("6")]
        result = _dedup_untimed(ws)
        assert len(result) == 3

    def test_untimed_then_timed_same_word(self):
        """Untimed 'the' followed by timed 'the' — both kept."""
        ws = [_w("the"), _w("the", 1, 2)]
        assert len(_dedup_untimed(ws)) == 2


# ── _interpolate_timestamps ──────────────────────────────────────────


class TestInterpolateTimestamps:
    def test_empty(self):
        assert _interpolate_timestamps([]) == []

    def test_all_timed_unchanged(self):
        ws = [_w("hello", 0, 0.5), _w("world", 0.5, 1)]
        result = _interpolate_timestamps(ws)
        assert result[0]["start"] == 0
        assert result[1]["start"] == 0.5

    def test_single_untimed_gets_timestamp(self):
        ws = [_w("hello", 0, 0.5), _w("1876."), _w("world", 1.0, 1.5)]
        result = _interpolate_timestamps(ws)
        mid = result[1]
        assert mid["start"] == pytest.approx(0.5, abs=0.1)
        assert mid["end"] <= 1.0
        assert mid["end"] > mid["start"]

    def test_multiple_untimed_between_timed(self):
        ws = [_w("a", 0, 0.5), _w("1"), _w("2"), _w("3"), _w("b", 2, 2.5)]
        result = _interpolate_timestamps(ws)
        # All should have start < end
        for r in result:
            assert r["start"] < r["end"] or r["start"] == r["end"] == 0
        # Monotonic
        for i in range(1, len(result)):
            assert result[i]["start"] >= result[i - 1]["start"]

    def test_untimed_at_end(self):
        """Untimed words at the end (no next timed word)."""
        ws = [_w("hello", 0, 0.5), _w("world")]
        result = _interpolate_timestamps(ws)
        assert result[1]["start"] == pytest.approx(0.5, abs=0.01)
        assert result[1]["end"] > result[1]["start"]

    def test_score_set_to_zero_for_interpolated(self):
        ws = [_w("a", 0, 0.5), _w("1876."), _w("b", 1, 1.5)]
        result = _interpolate_timestamps(ws)
        assert result[1]["score"] == 0.0


# ── _attach_punctuation ──────────────────────────────────────────────


class TestAttachPunctuation:
    def test_empty(self):
        assert _attach_punctuation([]) == []

    def test_standalone_period_merged(self):
        ws = [
            _w("word", 0, 0.5),
            _w(".", 0.5, 0.51),
        ]
        result = _attach_punctuation(ws)
        assert len(result) == 1
        assert result[0]["word"] == "word."
        assert result[0]["end"] == 0.51

    def test_standalone_comma_merged(self):
        ws = [
            _w("hello", 0, 0.5),
            _w(",", 0.5, 0.51),
            _w("world", 0.6, 1.0),
        ]
        result = _attach_punctuation(ws)
        assert len(result) == 2
        assert result[0]["word"] == "hello,"

    def test_multiple_punct_merged(self):
        ws = [
            _w("word", 0, 0.5),
            _w(".", 0.5, 0.51),
            _w(".", 0.51, 0.52),
        ]
        result = _attach_punctuation(ws)
        assert len(result) == 1
        assert result[0]["word"] == "word.."

    def test_leading_punct_not_merged_to_nothing(self):
        """First word is punctuation — no previous word to merge into."""
        ws = [_w(".", 0, 0.1), _w("hello", 0.1, 0.5)]
        result = _attach_punctuation(ws)
        assert len(result) == 2  # punct stays as-is

    def test_normal_words_unchanged(self):
        ws = [_w("hello", 0, 0.5), _w("world", 0.5, 1)]
        result = _attach_punctuation(ws)
        assert len(result) == 2


# ── _collapse_repeats ────────────────────────────────────────────────


class TestCollapseRepeats:
    def test_empty(self):
        assert _collapse_repeats([]) == []

    def test_no_repeats(self):
        ws = [_w("a"), _w("b"), _w("c")]
        assert len(_collapse_repeats(ws, 2, 4)) == 3

    def test_2gram_repeat_4x_collapsed(self):
        """[A, B] × 4 → [A, B]"""
        ws = [_w("iPhone"), _w("7?s")] * 4
        result = _collapse_repeats(ws, pattern_len=2, min_repeats=4)
        assert len(result) == 2
        assert result[0]["word"] == "iPhone"
        assert result[1]["word"] == "7?s"

    def test_2gram_repeat_3x_kept(self):
        """[A, B] × 3 is below threshold — kept as-is."""
        ws = [_w("iPhone"), _w("7?s")] * 3
        result = _collapse_repeats(ws, pattern_len=2, min_repeats=4)
        assert len(result) == 6

    def test_3gram_repeat(self):
        ws = [_w("a"), _w("b"), _w("c")] * 5
        result = _collapse_repeats(ws, pattern_len=3, min_repeats=4)
        assert len(result) == 3

    def test_repeat_in_middle(self):
        """Non-repeating → repeating → non-repeating."""
        prefix = [_w("start")]
        repeat = [_w("x"), _w("y")] * 5
        suffix = [_w("end")]
        ws = prefix + repeat + suffix
        result = _collapse_repeats(ws, pattern_len=2, min_repeats=4)
        words = [r["word"] for r in result]
        assert words == ["start", "x", "y", "end"]


# ── _replace_long_words ──────────────────────────────────────────────


class TestReplaceLongWords:
    def test_normal_words_unchanged(self):
        ws = [_w("hello", 0, 0.5)]
        result = _replace_long_words(ws)
        assert result[0]["word"] == "hello"

    def test_all_upper_long_replaced(self):
        ws = [_w("A" * 35, 0, 1)]
        result = _replace_long_words(ws)
        assert result[0]["word"] == "..."

    def test_mixed_case_under_50_kept(self):
        """Mixed-case long word under 50 chars — kept."""
        ws = [_w("Spanish-American-Cuban-Philippine", 0, 1)]
        result = _replace_long_words(ws)
        assert result[0]["word"] == "Spanish-American-Cuban-Philippine"

    def test_very_long_always_replaced(self):
        """Over 50 chars — always replaced regardless of case."""
        ws = [_w("a" * 51, 0, 1)]
        result = _replace_long_words(ws)
        assert result[0]["word"] == "..."


# ── sanitize_whisperx (integration) ──────────────────────────────────


class TestSanitizeWhisperx:
    def test_empty(self):
        assert sanitize_whisperx([]) == []

    def test_returns_word_objects(self):
        ws = [_w("Hello", 0, 0.5), _w("world.", 0.5, 1.0)]
        result = sanitize_whisperx(ws)
        assert all(isinstance(w, Word) for w in result)

    def test_basic_timed_words(self):
        ws = [_w("Hello", 0, 0.5), _w("world", 0.5, 1.0)]
        result = sanitize_whisperx(ws)
        assert len(result) == 2
        assert result[0].word == "Hello"
        assert result[0].start == 0
        assert result[1].word == "world"

    def test_untimed_number_gets_timestamp(self):
        ws = [
            _w("year", 0, 0.5),
            _w("1876."),
            _w("The", 1.0, 1.3),
        ]
        result = sanitize_whisperx(ws)
        assert len(result) == 3
        num = result[1]
        assert num.start >= 0.5
        assert num.end <= 1.0

    def test_dedup_then_interpolate(self):
        """Multiple untimed ♪ → single ♪ with interpolated timestamp."""
        ws = [
            _w("Hello", 0, 0.5),
            _w("♪"),
            _w("♪"),
            _w("♪"),
            _w("world", 1.0, 1.5),
        ]
        result = sanitize_whisperx(ws)
        symbols = [w for w in result if w.word == "♪"]
        assert len(symbols) == 1
        assert symbols[0].start >= 0.5
        assert symbols[0].end <= 1.0

    def test_punct_attached(self):
        ws = [
            _w("word", 0, 0.5),
            _w(".", 0.5, 0.51),
            _w("next", 0.6, 1.0),
        ]
        result = sanitize_whisperx(ws)
        assert len(result) == 2
        assert result[0].word == "word."

    def test_repeats_collapsed(self):
        repeat = []
        for _ in range(5):
            repeat.append(_w("iPhone", 0, 0.1))
            repeat.append(_w("7?s", 0.1, 0.2))
        result = sanitize_whisperx(repeat)
        iphone_count = sum(1 for w in result if w.word == "iPhone")
        assert iphone_count == 1

    def test_full_pipeline_from_whisperx_helper_example(self):
        """Adapted from the __main__ example in the old whisperx_helper.py."""
        ws = [
            {"word": "7?s", "start": 1182.621, "end": 1182.661, "score": 0.5},
            {"word": "iPhone", "start": 1182.721, "end": 1183.042, "score": 0.542},
            {"word": "7?s", "start": 1183.142, "end": 1183.162, "score": 0.0},
            {"word": "iPhone", "start": 1183.182, "end": 1183.302, "score": 0.0},
            {"word": "7?s", "start": 1183.322, "end": 1183.342, "score": 0.0},
            {"word": "iPhone", "start": 1183.362, "end": 1183.482, "score": 0.0},
            {"word": "7?s", "start": 1183.502, "end": 1183.542, "score": 0.5},
            {"word": "iPhone", "start": 1183.662, "end": 1186.604, "score": 0.549},
            {"word": "7?s", "start": 1186.644, "end": 1186.664, "score": 0.0},
            {"word": "iPhone", "start": 1186.684, "end": 1186.804, "score": 0.331},
            {"word": "7?s", "start": 1186.824, "end": 1186.844, "score": 0.0},
            {"word": "iPhone", "start": 1186.864, "end": 1187.004, "score": 0.078},
            {"word": "7?s", "start": 1187.024, "end": 1187.044, "score": 0.0},
            {"word": "iPhone", "start": 1187.064, "end": 1187.204, "score": 0.166},
            {"word": "7?s", "start": 1187.224, "end": 1187.244, "score": 0.0},
            {"word": "iPhone", "start": 1187.264, "end": 1187.585, "score": 0.353},
            {"word": "7?s", "start": 1187.605, "end": 1187.625, "score": 0.0},
            {"word": "iPhone", "start": 1187.645, "end": 1187.825, "score": 0.36},
            {"word": "7?s", "start": 1187.845, "end": 1187.865, "score": 0.0},
            {"word": "iPhone", "start": 1187.885, "end": 1188.065, "score": 0.338},
            {"word": "7?s", "start": 1188.105, "end": 1188.125, "score": 0.0},
        ]
        result = sanitize_whisperx(ws)
        # Should collapse the repeating "7?s iPhone" pattern
        assert len(result) <= 4, f"Expected collapse, got {len(result)} words"

    def test_speaker_preserved(self):
        ws = [{"word": "Hello", "start": 0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_01"}]
        result = sanitize_whisperx(ws)
        assert result[0].speaker == "SPEAKER_01"


# ── parse_whisperx ───────────────────────────────────────────────────


class TestParseWhisperx:
    def test_valid_json(self):
        data = {
            "word_segments": [
                {"word": "Hello", "start": 0, "end": 0.5, "score": 0.9},
                {"word": "world", "start": 0.5, "end": 1.0, "score": 0.8},
            ]
        }
        result = parse_whisperx(data)
        assert len(result) == 2
        assert all(isinstance(w, Word) for w in result)

    def test_missing_word_segments_key(self):
        with pytest.raises(KeyError, match="word_segments"):
            parse_whisperx({"segments": []})

    def test_empty_word_segments(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_whisperx({"word_segments": []})
