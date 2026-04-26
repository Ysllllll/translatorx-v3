"""Tests for application/stages/structure.py — Punc / Chunk / Merge."""

from __future__ import annotations

import pytest

from application.stages.structure import ChunkParams, ChunkStage, MergeParams, MergeStage, PuncParams, PuncStage
from domain.model import SentenceRecord


def _rec(text: str, start: float = 0.0, end: float = 1.0, **kw) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=start, end=end, **kw)


# ----- punc -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_punc_applies_per_record() -> None:
    fn = lambda texts: [[t.capitalize() + "."] for t in texts]
    stage = PuncStage(PuncParams(language="en"), fn)
    out = await stage.apply([_rec("hello world"), _rec("goodbye")], None)
    assert out[0].src_text == "Hello world."
    assert out[1].src_text == "Goodbye."


@pytest.mark.asyncio
async def test_punc_empty_input() -> None:
    stage = PuncStage(PuncParams(language="en"), lambda x: [])
    assert await stage.apply([], None) == []


@pytest.mark.asyncio
async def test_punc_length_mismatch_raises() -> None:
    stage = PuncStage(PuncParams(language="en"), lambda texts: [["a"], ["b"]])
    with pytest.raises(RuntimeError, match="returned 2 groups for 1"):
        await stage.apply([_rec("x")], None)


# ----- chunk ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_explodes_records() -> None:
    fn = lambda texts: [t.split() for t in texts]
    stage = ChunkStage(ChunkParams(language="en"), fn)
    out = await stage.apply([_rec("a b c"), _rec("x y")], None)
    assert [r.src_text for r in out] == ["a", "b", "c", "x", "y"]
    # All sub-records share the same time window as the source
    assert all(r.start == 0.0 and r.end == 1.0 for r in out[:3])


@pytest.mark.asyncio
async def test_chunk_unchanged_passes_through() -> None:
    fn = lambda texts: [[t] for t in texts]
    stage = ChunkStage(ChunkParams(language="en"), fn)
    inp = [_rec("hello")]
    out = await stage.apply(inp, None)
    assert out == inp


# ----- merge ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_concatenates_short_neighbors() -> None:
    stage = MergeStage(MergeParams(max_len=20))
    out = await stage.apply([_rec("hi", end=1.0), _rec("there", start=1.0, end=2.0)], None)
    assert len(out) == 1
    assert out[0].src_text == "hi there"
    assert out[0].start == 0.0
    assert out[0].end == 2.0


@pytest.mark.asyncio
async def test_merge_respects_max_len() -> None:
    stage = MergeStage(MergeParams(max_len=4))
    out = await stage.apply([_rec("hi"), _rec("there")], None)
    assert len(out) == 2  # cannot merge
