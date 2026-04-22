"""Tests for :func:`domain.subtitle.rebalance_segment_words`."""

from __future__ import annotations

from domain.lang import LangOps
from domain.model import Segment, Word
from domain.subtitle import rebalance_segment_words


EN = LangOps.for_language("en")


def _w(word: str, start: float, end: float) -> Word:
    return Word(word=word, start=start, end=end)


def test_rebalance_prefers_target_ratio():
    # Original: "hello" | "world today" (lengths 5 vs 11 → ratio 5/11)
    # Retarget to ratio 2 (left should be ~2x right).
    seg_a = Segment(start=0.0, end=1.0, text="hello", words=[_w("hello", 0.0, 1.0)])
    seg_b = Segment(start=1.0, end=3.0, text="world today", words=[_w("world", 1.0, 2.0), _w("today", 2.0, 3.0)])
    new_a, new_b = rebalance_segment_words(seg_a, seg_b, target_ratio=2.0, max_chunk_len=100, ops=EN)
    # With 3 words total and target_ratio=2, boundary at i=2 gives left="hello world" (11)
    # and right="today" (5), ratio=2.2 — closest to 2.
    assert new_a.text == "hello world"
    assert new_b.text == "today"
    assert new_a.words[-1].word == "world"
    assert new_b.words[0].word == "today"


def test_rebalance_returns_unchanged_when_max_len_too_tight():
    seg_a = Segment(start=0.0, end=1.0, text="aaaa", words=[_w("aaaa", 0.0, 1.0)])
    seg_b = Segment(start=1.0, end=2.0, text="bbbb", words=[_w("bbbb", 1.0, 2.0)])
    # max_chunk_len=3 forbids any arrangement (each piece already ≥ 4).
    new_a, new_b = rebalance_segment_words(seg_a, seg_b, target_ratio=1.0, max_chunk_len=3, ops=EN)
    assert new_a is seg_a
    assert new_b is seg_b


def test_rebalance_short_circuits_on_insufficient_words():
    seg_a = Segment(start=0.0, end=1.0, text="only", words=[_w("only", 0.0, 1.0)])
    seg_b = Segment(start=1.0, end=2.0, text="", words=[])
    new_a, new_b = rebalance_segment_words(seg_a, seg_b, target_ratio=1.0, max_chunk_len=100, ops=EN)
    assert new_a is seg_a
    assert new_b is seg_b
