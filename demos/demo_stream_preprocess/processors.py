"""Streaming stages used by the demo's WS handler.

Two stages sit between "raw segments from the wire" and "enriched
:class:`SentenceRecord` to send back":

- :class:`PuncBufferStage` — bridges unpunctuated ASR cues into
  :class:`PushQueueSource`, which otherwise would never emit because
  sentence-ending punctuation is its only boundary signal.
- :class:`PreprocessProcessor` — per-record clauses + length-bounded
  chunking. Mirrors the shape of a production
  :class:`ports.processor.Processor` but without Store/VideoKey, which
  keeps the demo small and testable without a full App.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import AsyncIterator, Callable

from adapters.sources.push import PushQueueSource
from domain.model import Segment, SentenceRecord
from domain.subtitle import Subtitle


class PuncBufferStage:
    """Buffer incoming segments and flush punc-restored text back out.

    ``PushQueueSource`` relies on sentence-ending punctuation to cut
    sentence boundaries. Raw ASR cues have none, so we interpose this
    stage: accumulate up to ``window`` segments (or until ``flush`` is
    requested), run punc restore on the joined text, and feed the
    result back as a single merged :class:`Segment` whose span covers
    the buffered range.

    This loses per-segment timing granularity inside the window, but
    ``SentenceRecord`` only needs the sentence span + word alignment,
    so it's fine for the preprocess demo.
    """

    def __init__(
        self,
        *,
        punc_fn: Callable[[list[str]], list[list[str]]],
        downstream: PushQueueSource,
        window: int = 4,
    ) -> None:
        self._punc_fn = punc_fn
        self._downstream = downstream
        self._window = max(1, window)
        self._buf: list[Segment] = []

    async def feed(self, segment: Segment) -> None:
        self._buf.append(segment)
        if len(self._buf) >= self._window:
            await self._emit()

    async def flush(self) -> None:
        if self._buf:
            await self._emit()

    async def _emit(self) -> None:
        buf, self._buf = self._buf, []
        joined = " ".join(s.text.strip() for s in buf if s.text.strip())
        if not joined:
            return
        restored_groups = self._punc_fn([joined])
        restored = " ".join(restored_groups[0]) if restored_groups else joined
        merged = Segment(
            start=buf[0].start,
            end=buf[-1].end,
            text=restored,
            speaker=buf[0].speaker,
            words=[w for s in buf for w in s.words],
            extra=dict(buf[0].extra or {}),
        )
        await self._downstream.feed(merged)


class PreprocessProcessor:
    """Per-record clauses + length-bounded chunking."""

    name = "preprocess_stream_demo"

    def __init__(
        self,
        *,
        language: str,
        chunk_fn: Callable[[list[str]], list[list[str]]],
        max_len: int = 60,
        merge_under: int = 90,
    ) -> None:
        self._language = language
        self._chunk_fn = chunk_fn
        self._max_len = max_len
        self._merge_under = merge_under

    def fingerprint(self) -> str:
        return "demo"

    def output_is_stale(self, rec: SentenceRecord) -> bool:
        return False

    async def process(self, upstream: AsyncIterator[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
        async for rec in upstream:
            # ``chunk_fn`` may invoke blocking LLM / spaCy I/O, so we
            # off-load the whole build to a worker thread — otherwise the
            # event loop can't answer WS pings during a long LLM call.
            records = await asyncio.to_thread(self._build_records, rec)
            if not records:
                continue
            extra = dict(rec.extra or {})
            for new in records:
                yield replace(new, extra=dict(extra))

    def _build_records(self, rec: SentenceRecord) -> list[SentenceRecord]:
        # ``.sentences()`` scopes downstream ops to a sentence-level
        # pipeline so ``.records()`` yields one enriched record (chunks
        # stay *inside* the sentence instead of being promoted to their
        # own records).
        #
        # The final ``.merge(max_len)`` is the standard split→merge pair:
        # ``transform(chunk_fn, scope="chunk")`` can leave adjacent
        # short chunks (e.g. one-clause tail fragments) that fit well
        # within ``max_len`` when recombined. Without the merge we'd
        # ship more, shorter segments than necessary.
        sub = Subtitle(list(rec.segments), language=self._language)
        sub = sub.sentences().clauses(merge_under=self._merge_under).transform(self._chunk_fn, scope="chunk").merge(self._max_len)
        return sub.records()

    async def aclose(self) -> None:
        return None
