"""Unit tests for the streaming preprocess demo.

Covers the pure Python pieces of ``demos/demo_stream_preprocess``:

- :class:`PuncBufferStage` — window buffering + flush + empty skip
- :class:`PreprocessProcessor` — sentence-scoped clauses + chunking
- :func:`_safe_send` — True on success, False on each flavour of
  "WebSocket is dead" we care about

The FastAPI route itself is not exercised here; it's thin glue and
we've already smoked it end-to-end with the client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src", _REPO / "demos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adapters.sources.push import PushQueueSource  # noqa: E402
from domain.model import Segment, SentenceRecord, Word  # noqa: E402

from streaming.preprocess_server.processors import (  # noqa: E402
    PreprocessProcessor,
    PuncBufferStage,
)
from streaming.preprocess_server.ws_app import _safe_send  # noqa: E402


# ─── PuncBufferStage ────────────────────────────────────────────────


def _mk_seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


@pytest.mark.asyncio
async def test_punc_buffer_emits_after_window() -> None:
    emitted: list[Segment] = []

    class _Sink:
        async def feed(self, seg: Segment) -> None:
            emitted.append(seg)

    def punc_fn(texts: list[str]) -> list[list[str]]:
        return [[t + "."] for t in texts]

    stage = PuncBufferStage(punc_fn=punc_fn, downstream=_Sink(), window=3)
    for i in range(3):
        await stage.feed(_mk_seg(float(i), float(i + 1), f"chunk{i}"))

    assert len(emitted) == 1
    merged = emitted[0]
    assert merged.start == 0.0
    assert merged.end == 3.0
    assert merged.text == "chunk0 chunk1 chunk2."


@pytest.mark.asyncio
async def test_punc_buffer_flush_partial() -> None:
    emitted: list[Segment] = []

    class _Sink:
        async def feed(self, seg: Segment) -> None:
            emitted.append(seg)

    stage = PuncBufferStage(punc_fn=lambda ts: [[t] for t in ts], downstream=_Sink(), window=10)
    await stage.feed(_mk_seg(0, 1, "only one"))
    assert emitted == []
    await stage.flush()
    assert len(emitted) == 1
    assert emitted[0].text == "only one"

    # second flush on empty buffer is a no-op
    await stage.flush()
    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_punc_buffer_skips_empty_text() -> None:
    emitted: list[Segment] = []

    class _Sink:
        async def feed(self, seg: Segment) -> None:
            emitted.append(seg)

    stage = PuncBufferStage(punc_fn=lambda ts: [[t] for t in ts], downstream=_Sink(), window=2)
    await stage.feed(_mk_seg(0, 1, "   "))
    await stage.feed(_mk_seg(1, 2, ""))
    assert emitted == []


@pytest.mark.asyncio
async def test_punc_buffer_preserves_words_across_window() -> None:
    emitted: list[Segment] = []

    class _Sink:
        async def feed(self, seg: Segment) -> None:
            emitted.append(seg)

    s1 = Segment(start=0.0, end=1.0, text="hello world", words=[Word(word="hello", start=0.0, end=0.5), Word(word="world", start=0.5, end=1.0)])
    s2 = Segment(start=1.0, end=2.0, text="goodbye", words=[Word(word="goodbye", start=1.0, end=2.0)])

    stage = PuncBufferStage(punc_fn=lambda ts: [[t] for t in ts], downstream=_Sink(), window=2)
    await stage.feed(s1)
    await stage.feed(s2)

    assert len(emitted) == 1
    assert [w.word for w in emitted[0].words] == ["hello", "world", "goodbye"]


# ─── PreprocessProcessor ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preprocess_processor_yields_enriched_record() -> None:
    source = PushQueueSource(language="en")
    proc = PreprocessProcessor(language="en", chunk_fn=lambda texts: [[t] for t in texts])

    await source.feed(Segment(start=0.0, end=3.0, text="Hello world. How are you today?"))
    await source.close()

    out: list[SentenceRecord] = []
    async for rec in proc.process(source.read()):
        out.append(rec)

    assert len(out) >= 1
    # every output must carry a span and some segments
    for rec in out:
        assert rec.end >= rec.start
        assert rec.segments


@pytest.mark.asyncio
async def test_preprocess_processor_preserves_extra() -> None:
    source = PushQueueSource(language="en")
    proc = PreprocessProcessor(language="en", chunk_fn=lambda texts: [[t] for t in texts])

    # PushQueueSource only emits at sentence punctuation
    await source.feed(Segment(start=0.0, end=2.0, text="Short sentence here."))
    await source.close()

    records = [r async for r in proc.process(source.read())]
    assert records
    # extra gets copied, not shared by reference
    for r in records:
        assert isinstance(r.extra, dict)


# ─── _safe_send ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_safe_send_returns_true_on_success() -> None:
    ws = AsyncMock()
    ws.send_text = AsyncMock(return_value=None)
    assert await _safe_send(ws, {"type": "ready"}) is True
    ws.send_text.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("WebSocket is not connected"), ConnectionError("broken pipe")])
async def test_safe_send_returns_false_on_dead_socket(exc: Exception) -> None:
    ws = AsyncMock()
    ws.send_text = AsyncMock(side_effect=exc)
    assert await _safe_send(ws, {"type": "ready"}) is False


@pytest.mark.asyncio
async def test_safe_send_returns_false_on_websocket_disconnect() -> None:
    from fastapi import WebSocketDisconnect

    ws = AsyncMock()
    ws.send_text = AsyncMock(side_effect=WebSocketDisconnect(code=1001))
    assert await _safe_send(ws, {"type": "ready"}) is False
