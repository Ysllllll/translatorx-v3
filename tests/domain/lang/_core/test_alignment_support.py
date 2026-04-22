"""Tests for alignment-support methods on LangOps."""

from __future__ import annotations

import pytest

from domain.lang import LangOps


EN = LangOps.for_language("en")
ZH = LangOps.for_language("zh")


class TestFindHalfJoinBalance:
    def test_two_texts_always_returns_one(self):
        assert EN.find_half_join_balance(["hello", "world"]) == [1]

    def test_prefers_clause_boundary(self):
        # Four segments; boundary after segment 1 ends with ",", segment 2 ends plain.
        texts = ["Hello there,", "friend of mine", "long ramble", "end here."]
        candidates = EN.find_half_join_balance(texts)
        # Boundary 1 (after "Hello there,") should rank first because the
        # preceding text ends on a clause separator.
        assert candidates[0] == 1

    def test_falls_back_to_length_balance(self):
        # None end with clause punct — pick the most balanced length split.
        texts = ["aaaa", "bb", "ccccc", "dd"]
        candidates = EN.find_half_join_balance(texts)
        # Total len=13; boundary 2 (left=6, right=7) is most balanced.
        assert candidates[0] == 2


class TestCheckAndCorrectSplitSentence:
    def test_simple_accept(self):
        good, fixed = EN.check_and_correct_split_sentence(["hello there", "friend of mine"], "hello there friend of mine")
        assert good
        assert fixed == ["hello there", "friend of mine"]

    def test_mismatch_returns_false(self):
        good, fixed = EN.check_and_correct_split_sentence(["banana", "apple"], "hello world goodbye")
        assert not good

    def test_cjk_reverse_swap_when_terminator_on_first(self):
        # CJK: first ends with sentence terminator, second with clause sep →
        # reversed concat matches → swap so first gets clause sep, second gets terminator.
        # Original sentence flows naturally as "，...。"
        good, fixed = ZH.check_and_correct_split_sentence(["世界。", "你好，"], "你好，世界。")
        assert good
        # Expect swap: sentence-terminator attaches to the piece that matched
        # at the END of the sentence ("世界"), clause-sep to the piece at the START.
        assert fixed[0].endswith("，")
        assert fixed[1].endswith("。")


class TestLengthRatio:
    def test_same_length(self):
        assert abs(EN.length_ratio("hello", "hello") - 1.0) < 1e-3

    def test_longer_numerator_ratio_above_one(self):
        r = EN.length_ratio("hello world", "hi")
        assert r > 1.0


class TestEndsWithClausePunct:
    def test_en_comma(self):
        assert EN.ends_with_clause_punct("Hello there,")

    def test_en_period(self):
        assert EN.ends_with_clause_punct("Done.")

    def test_plain_word(self):
        assert not EN.ends_with_clause_punct("hello")

    def test_cjk_full_width(self):
        assert ZH.ends_with_clause_punct("你好，")
        assert ZH.ends_with_clause_punct("结束。")
