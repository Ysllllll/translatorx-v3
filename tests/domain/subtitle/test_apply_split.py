"""Tests for Subtitle.transform with scope parameter."""

from __future__ import annotations

from domain.model import Segment, Word
from domain.subtitle import Subtitle


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
    actual_src_texts = [r.src_text for r in records]
    expected_src_texts = ["hello world.", "foo bar."]
    assert actual_src_texts == expected_src_texts
    # split_fn("hello world.") → ["hello", "world."]; transform places each
    # token in its own sub-segment within the parent record.
    actual_segments_per_record = [len(r.segments) for r in records]
    expected_segments_per_record = [2, 2]
    assert actual_segments_per_record == expected_segments_per_record


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
# transform — scope="joined"
# ---------------------------------------------------------------------------


def test_transform_joined_scope_pre_sentence() -> None:
    """scope='joined' joins all chunks before sending to fn."""
    segs = [_mk_seg(0.0, 1.0, "hello world how are you")]

    cache: dict[str, list[str]] = {}

    def punc(batch: list[str]) -> list[list[str]]:
        # Each text should be the full joined pipeline text
        return [[t + "!"] for t in batch]

    sub = Subtitle(segs, language="en").transform(punc, cache=cache, scope="joined")
    records = sub.sentences().records()
    # The exclamation mark should be present
    assert any("!" in r.src_text for r in records)


def test_transform_joined_scope_post_sentence() -> None:
    """scope='joined' after sentences() — per-sentence joined text sent to fn."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]

    received_texts: list[str] = []

    def punc(batch: list[str]) -> list[list[str]]:
        received_texts.extend(batch)
        return [[t + "!"] for t in batch]

    sub = Subtitle(segs, language="en").sentences()
    # Each pipeline has one sentence; scope="joined" joins its chunks
    sub = sub.transform(punc, scope="joined")
    records = sub.records()

    # fn received the joined text of each sentence pipeline (one entry per
    # sentence), not individual chunks.
    assert received_texts == ["hello world.", "foo bar."]


def test_transform_joined_scope_with_cache() -> None:
    """Cache works correctly with scope='joined' — keyed by joined text."""
    segs = [_mk_seg(0.0, 2.0, "hello world. foo bar.")]
    cache: dict[str, list[str]] = {}
    call_count = {"n": 0}

    def counting_fn(batch: list[str]) -> list[list[str]]:
        call_count["n"] += len(batch)
        return [[t] for t in batch]

    # First call
    Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache, scope="joined")
    first_calls = call_count["n"]
    assert first_calls > 0

    # Second call — all cache hits
    call_count["n"] = 0
    Subtitle(segs, language="en").sentences().transform(counting_fn, cache=cache, scope="joined")
    assert call_count["n"] == 0
