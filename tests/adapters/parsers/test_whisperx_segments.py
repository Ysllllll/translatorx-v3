"""Tests for segment-aware WhisperX word extraction."""

from __future__ import annotations

import pytest

from adapters.parsers import parse_whisperx
from adapters.parsers.whisperx.segments import _synthesize_words, _tokenize, extract_word_dicts


class TestTokenize:
    def test_whitespace(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_cjk_no_space(self):
        assert _tokenize("你好世界") == ["你好世界"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_only_whitespace(self):
        assert _tokenize("   \t\n  ") == []


class TestSynthesizeWords:
    def test_distributes_evenly(self):
        seg = {"text": "a b c d", "start": 0.0, "end": 4.0}
        out = _synthesize_words(seg)
        assert [w["word"] for w in out] == ["a", "b", "c", "d"]
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 1.0
        assert out[3]["end"] == 4.0  # last bucket pinned to segment.end

    def test_inherits_speaker(self):
        seg = {"text": "hi", "start": 0.0, "end": 1.0, "speaker": "S1"}
        out = _synthesize_words(seg)
        assert out[0]["speaker"] == "S1"

    def test_zero_duration_collapses(self):
        seg = {"text": "a b c", "start": 1.0, "end": 1.0}
        out = _synthesize_words(seg)
        assert len(out) == 1
        assert out[0]["word"] == "a b c"
        assert out[0]["start"] == out[0]["end"] == 1.0

    def test_missing_timing_returns_empty(self):
        assert _synthesize_words({"text": "a", "start": 0.0}) == []
        assert _synthesize_words({"text": "a", "end": 1.0}) == []

    def test_inverted_timing_returns_empty(self):
        assert _synthesize_words({"text": "a", "start": 2.0, "end": 1.0}) == []

    def test_empty_text(self):
        assert _synthesize_words({"text": "  ", "start": 0.0, "end": 1.0}) == []


class TestExtractWordDicts:
    def test_uses_inner_words(self):
        data = {"segments": [{"text": "hi there", "start": 0.0, "end": 1.0, "words": [{"word": "hi", "start": 0.0, "end": 0.5}, {"word": "there", "start": 0.5, "end": 1.0}]}], "word_segments": []}
        out = extract_word_dicts(data)
        assert [w["word"] for w in out] == ["hi", "there"]

    def test_synthesizes_when_words_missing(self):
        data = {"segments": [{"text": "alpha beta gamma", "start": 0.0, "end": 3.0}]}
        out = extract_word_dicts(data)
        assert [w["word"] for w in out] == ["alpha", "beta", "gamma"]
        # evenly distributed timing
        assert out[0]["start"] == 0.0
        assert out[2]["end"] == 3.0

    def test_mixed_segments_preserves_order(self):
        data = {"segments": [{"text": "hi", "start": 0.0, "end": 1.0, "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}, {"text": "lost segment", "start": 1.0, "end": 3.0}, {"text": "back", "start": 3.0, "end": 4.0, "words": [{"word": "back", "start": 3.0, "end": 4.0}]}]}
        out = extract_word_dicts(data)
        assert [w["word"] for w in out] == ["hi", "lost", "segment", "back"]
        assert out[1]["start"] == 1.0
        assert out[2]["end"] == 3.0

    def test_empty_inner_words_falls_back_to_synthesis(self):
        data = {"segments": [{"text": "fallback text", "start": 0.0, "end": 2.0, "words": []}]}
        out = extract_word_dicts(data)
        assert [w["word"] for w in out] == ["fallback", "text"]

    def test_segment_speaker_inherited(self):
        data = {"segments": [{"text": "hi there", "start": 0.0, "end": 1.0, "speaker": "SPK_01", "words": [{"word": "hi", "start": 0.0, "end": 0.5}, {"word": "there", "start": 0.5, "end": 1.0, "speaker": "SPK_02"}]}]}
        out = extract_word_dicts(data)
        assert out[0]["speaker"] == "SPK_01"  # inherited
        assert out[1]["speaker"] == "SPK_02"  # preserved

    def test_falls_back_to_word_segments(self):
        data = {"word_segments": [{"word": "legacy", "start": 0.0, "end": 1.0}]}
        out = extract_word_dicts(data)
        assert out[0]["word"] == "legacy"

    def test_segments_take_precedence_over_word_segments(self):
        data = {"segments": [{"text": "from segments", "start": 0.0, "end": 1.0, "words": [{"word": "from", "start": 0.0, "end": 0.5}, {"word": "segments", "start": 0.5, "end": 1.0}]}], "word_segments": [{"word": "from_top", "start": 0.0, "end": 1.0}]}
        out = extract_word_dicts(data)
        assert [w["word"] for w in out] == ["from", "segments"]


class TestParseWhisperxSegmentsPath:
    def test_recovers_word_less_segment(self):
        data = {"segments": [{"text": "hi", "start": 0.0, "end": 1.0, "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}, {"text": "missed words", "start": 1.0, "end": 3.0}]}
        words = parse_whisperx(data)
        # Pipeline preserves all 3 words (1 from inner + 2 synthesized)
        assert [w.word for w in words] == ["hi", "missed", "words"]

    def test_synthesized_words_pass_through_pipeline(self):
        # W5 attach-punct should still work on synthesized words.
        data = {"segments": [{"text": "Hello , world", "start": 0.0, "end": 3.0}]}
        words = parse_whisperx(data)
        # ',' attaches to 'Hello' via W5
        assert any("Hello" in w.word and "," in w.word for w in words)


@pytest.mark.parametrize("n_tokens,duration", [(1, 1.0), (2, 1.0), (5, 2.5), (10, 0.5)])
def test_synthesize_intervals_are_contiguous(n_tokens, duration):
    text = " ".join(f"w{i}" for i in range(n_tokens))
    seg = {"text": text, "start": 10.0, "end": 10.0 + duration}
    out = _synthesize_words(seg)
    assert len(out) == n_tokens
    for i in range(len(out) - 1):
        assert out[i]["end"] == pytest.approx(out[i + 1]["start"])
    assert out[-1]["end"] == pytest.approx(10.0 + duration)
