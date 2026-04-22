"""Tests for the 2-piece recovery helper (mirror of legacy check_and_correct_split_sentence)."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.reconstruct import chunks_match_source, recover_pair


class TestChunksMatchSourceLanguage:
    def test_english_exact_space_join(self) -> None:
        assert chunks_match_source(["hello world", "how are you"], "hello world how are you", language="en")

    def test_chinese_concat_join(self) -> None:
        # CJK uses concat-join, verified through ops.join in the language path.
        assert chunks_match_source(["你好世界", "今天天气不错"], "你好世界今天天气不错", language="zh")

    def test_alnum_fallback_tolerates_spacing(self) -> None:
        # LLM drops or duplicates an inner space; the alnum fallback still accepts.
        assert chunks_match_source(["hello world ", "how are you"], "hello world how are you")

    def test_negative_case(self) -> None:
        assert not chunks_match_source(["hello", "totally different"], "hello world")


class TestRecoverPairEnglish:
    def test_no_recovery_needed_returns_normalized(self) -> None:
        source = "Hello world, this is a test."
        parts = ["Hello world,", "this is a test."]
        assert chunks_match_source(parts, source, language="en")

    def test_second_half_missing_derives_from_source(self) -> None:
        source = "Hello world, this is a test."
        # LLM returned only the first half (second empty).
        recovered = recover_pair(["Hello world,", ""], source, language="en")
        assert recovered is not None
        assert chunks_match_source(recovered, source, language="en")
        assert recovered[0].strip() == "Hello world,"
        assert "test" in recovered[1]

    def test_first_half_missing_derives_from_source(self) -> None:
        source = "Hello world, this is a test."
        recovered = recover_pair(["", "this is a test."], source, language="en")
        assert recovered is not None
        assert chunks_match_source(recovered, source, language="en")

    def test_reversed_order_accepted_without_can_reverse(self) -> None:
        source = "Hello world, this is a test."
        parts = ["this is a test.", "Hello world,"]
        recovered = recover_pair(parts, source, language="en", can_reverse=False)
        assert recovered is not None
        # With can_reverse=False, the returned order must join back correctly.
        assert chunks_match_source(recovered, source, language="en")

    def test_unrecoverable_returns_none(self) -> None:
        source = "Hello world, this is a test."
        parts = ["totally unrelated text", "also unrelated"]
        assert recover_pair(parts, source, language="en") is None

    def test_wrong_length_returns_none(self) -> None:
        assert recover_pair(["a", "b", "c"], "abc", language="en") is None
        assert recover_pair(["only one"], "only one more", language="en") is None


class TestRecoverPairChinese:
    def test_chinese_second_half_missing(self) -> None:
        source = "你好世界，今天天气真好。"
        recovered = recover_pair(["你好世界，", ""], source, language="zh")
        assert recovered is not None
        assert chunks_match_source(recovered, source, language="zh")

    def test_chinese_reversed_order_restored(self) -> None:
        """When LLM returns parts in reversed order, recovery restores the valid order.

        The legacy tail-punctuation swap is intentionally omitted
        because it breaks strict reconstruction, which
        :class:`Chunker._finalize` requires.
        """
        source = "你好世界，今天天气真好。"
        parts = ["今天天气真好。", "你好世界，"]
        recovered = recover_pair(parts, source, language="zh", can_reverse=True)
        assert recovered is not None
        # After recovery, parts reconstruct the source.
        assert chunks_match_source(recovered, source, language="zh")
        assert recovered[0].strip() == "你好世界，"
        assert recovered[1].strip() == "今天天气真好。"
