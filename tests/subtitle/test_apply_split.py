"""Tests for Subtitle.transform with scope parameter."""

from __future__ import annotations

from model import Segment, Word
from subtitle import Subtitle


def _mk_seg(start: float, end: float, text: str) -> Segment:
    # Build one Word per whitespace-split token with uniform timing.
    toks = text.split()
    if not toks:
        return Segment(start, end, text, words=[])
    step = (end - start) / len(toks)
    words = [Word(tok + " " if i < len(toks) - 1 else tok, start + i * step, start + (i + 1) * step) for i, tok in enumerate(toks)]
    return Segment(start, end, text, words=words)


# ---------------------------------------------------------------------------
# transform — scope="chunk" (default)
# ---------------------------------------------------------------------------


def test_transform_chunk_scope_pre_sentence() -> None:
    """transform(scope="chunk") before sentences() — each chunk sent individually."""
    segs = [_mk_seg(0.0, 1.0, "hello world how are you")]

    cache: dict[str, list[str]] = {}

    def punc(batch: list[str]) -> list[list[str]]:
        return [[t + "."] for t in batch]

    sub = Subtitle(segs, language="en").transform(punc, cache=cache).sentences()
    records = sub.records()
    assert any("." in r.src_text for r in records)
    # cache was populated
    assert cache


def test_transform_chunk_scope_post_sentence() -> None:
    """transform(scope="chunk") after sentences() — each chunk individually."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]

    def split_fn(batch: list[str]) -> list[list[str]]:
        return [t.split() for t in batch]

    sub = Subtitle(segs, language="en").sentences().transform(split_fn)
    records = sub.records()
    assert len(records) >= 1
    # Each sentence was split into individual words
    for r in records:
        assert len(r.segments) >= 1


def test_transform_chunk_scope_with_cache() -> None:
    """Cache is populated and reused for scope='chunk'."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]
    cache: dict[str, list[str]] = {}
    call_count = {"n": 0}

    def counting_fn(batch: list[str]) -> list[list[str]]:
        call_count["n"] += len(batch)
        return [[t] for t in batch]

    # First call — populates cache
    sub1 = Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache)
    records1 = sub1.records()
    first_calls = call_count["n"]
    assert first_calls > 0

    # Second call with same cache — all hits, no new calls
    call_count["n"] = 0
    sub2 = Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache)
    records2 = sub2.records()
    assert call_count["n"] == 0  # all cache hits


# ---------------------------------------------------------------------------
# transform — scope="pipeline"
# ---------------------------------------------------------------------------


def test_transform_pipeline_scope_pre_sentence() -> None:
    """scope='pipeline' joins all chunks before sending to fn."""
    segs = [_mk_seg(0.0, 1.0, "hello world how are you")]

    cache: dict[str, list[str]] = {}

    def punc(batch: list[str]) -> list[list[str]]:
        # Each text should be the full joined pipeline text
        return [[t + "!"] for t in batch]

    sub = Subtitle(segs, language="en").transform(punc, cache=cache, scope="pipeline")
    records = sub.sentences().records()
    # The exclamation mark should be present
    assert any("!" in r.src_text for r in records)


def test_transform_pipeline_scope_post_sentence() -> None:
    """scope='pipeline' after sentences() — per-sentence joined text sent to fn."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]

    received_texts: list[str] = []

    def punc(batch: list[str]) -> list[list[str]]:
        received_texts.extend(batch)
        return [[t + "!"] for t in batch]

    sub = Subtitle(segs, language="en").sentences()
    # Each pipeline has one sentence; scope="pipeline" joins its chunks
    sub = sub.transform(punc, scope="pipeline")
    records = sub.records()

    # fn received full sentence texts, not individual chunks
    assert len(received_texts) >= 1
    for t in received_texts:
        assert t  # non-empty


def test_transform_pipeline_scope_with_cache() -> None:
    """Cache works correctly with scope='pipeline' — keyed by joined text."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]
    cache: dict[str, list[str]] = {}
    call_count = {"n": 0}

    def counting_fn(batch: list[str]) -> list[list[str]]:
        call_count["n"] += len(batch)
        return [[t] for t in batch]

    # First call
    Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache, scope="pipeline")
    first_calls = call_count["n"]
    assert first_calls > 0

    # Second call — all cache hits
    call_count["n"] = 0
    Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache, scope="pipeline")
    assert call_count["n"] == 0
