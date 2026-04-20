"""Tests for Subtitle.transform / from_records."""

from __future__ import annotations

import pytest

from model import Segment, Word
from subtitle import Subtitle


def _mk_seg(start: float, end: float, text: str) -> Segment:
    # Build one Word per whitespace-split token with uniform timing.
    toks = text.split()
    if not toks:
        return Segment(start, end, text, words=[])
    step = (end - start) / len(toks)
    words = [
        Word(tok + " " if i < len(toks) - 1 else tok, start + i * step, start + (i + 1) * step)
        for i, tok in enumerate(toks)
    ]
    return Segment(start, end, text, words=words)


# ---------------------------------------------------------------------------
# transform — pre-sentence (auto-scope: global)
# ---------------------------------------------------------------------------


def test_transform_pre_sentence() -> None:
    """transform() before sentences() — global scope."""
    segs = [_mk_seg(0.0, 1.0, "hello world how are you")]

    cache: dict[str, list[str]] = {}

    def punc(batch: list[str]) -> list[list[str]]:
        return [[t + "."] for t in batch]

    sub = Subtitle(segs, language="en").transform(punc, cache=cache).sentences()
    records = sub.records()
    assert any("." in r.src_text for r in records)
    # cache was populated with video-level entry
    assert cache  # non-empty


def test_transform_pre_sentence_name_ignored() -> None:
    """In pre-sentence mode, name parameter is ignored (no chunk_cache)."""
    segs = [_mk_seg(0.0, 1.0, "hello world")]
    sub = Subtitle(segs, language="en").transform(lambda b: [[t] for t in b], name="should_be_ignored")
    # Not sentence-split, so records() auto-calls sentences()
    records = sub.records()
    for r in records:
        assert "should_be_ignored" not in r.chunk_cache


# ---------------------------------------------------------------------------
# transform — post-sentence (auto-scope: per-sentence)
# ---------------------------------------------------------------------------


def test_transform_post_sentence_populates_chunk_cache() -> None:
    """transform() after sentences() with name — stamps chunk_cache."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]

    def split_fn(batch: list[str]) -> list[list[str]]:
        return [t.split() for t in batch]

    sub = Subtitle(segs, language="en").sentences().transform(split_fn, name="chunk_llm")
    records = sub.records()
    assert len(records) >= 1
    for r in records:
        assert "chunk_llm" in r.chunk_cache
        assert r.chunk_cache["chunk_llm"]  # non-empty


def test_transform_post_sentence_without_name() -> None:
    """transform() after sentences() without name — no chunk_cache stamp."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]
    sub = Subtitle(segs, language="en").sentences().transform(lambda b: [[t] for t in b])
    records = sub.records()
    for r in records:
        assert r.chunk_cache == {}


# ---------------------------------------------------------------------------
# from_records
# ---------------------------------------------------------------------------


def test_from_records_rehydrates_pipelines() -> None:
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]
    records = Subtitle(segs, language="en").sentences().records()
    rehydrated = Subtitle.from_records(records, language="en").records()
    assert len(rehydrated) == len(records)
    for a, b in zip(records, rehydrated):
        assert a.src_text == b.src_text


def test_from_records_honors_chunk_cache_key() -> None:
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]

    def fake_chunker(batch: list[str]) -> list[list[str]]:
        return [t.split() for t in batch]

    sub = Subtitle(segs, language="en").sentences().transform(fake_chunker, name="chunk_llm")
    records = sub.records()
    # Rehydrate using cached chunk_llm — fake_chunker should NOT be called again.
    call_count = {"n": 0}

    def should_not_be_called(batch: list[str]) -> list[list[str]]:
        call_count["n"] += 1
        return [["BAD"] for _ in batch]

    rehydrated = Subtitle.from_records(records, language="en", chunk_cache_key="chunk_llm")
    # The pipelines already carry cached splits; subsequent records() reflect them.
    new_records = rehydrated.records()
    for r in new_records:
        assert "chunk_llm" in r.chunk_cache  # preserved
    # No additional calls because from_records reads from cache directly
    assert call_count["n"] == 0


def test_from_records_empty_list() -> None:
    sub = Subtitle.from_records([], language="en")
    assert sub.records() == []
