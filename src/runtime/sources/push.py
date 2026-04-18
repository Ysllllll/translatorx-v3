"""PushQueueSource — live segment feeder for browser-plugin / streaming.

Callers push raw :class:`Segment` items via :meth:`feed` and close the
stream via :meth:`close`. The :meth:`read` async generator yields
completed :class:`SentenceRecord` items as the internal
:class:`SubtitleStream` confirms full sentences, and one final batch
after ``close()`` flushes any trailing buffer (D-067 Stream mode).

The source is **one-shot**: ``read()`` must be consumed by exactly one
downstream processor chain.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from model import SentenceRecord, Segment
from subtitle import Subtitle


_SENTINEL = object()


class PushQueueSource:
    """Async queue-backed source; feed segments, iterate sentence records.

    Parameters
    ----------
    language:
        Source language code.
    split_by_speaker:
        If True, sentences are not merged across speaker changes.
    id_start:
        Starting id for ``extra["id"]`` allocation. Defaults to ``0``.
    maxsize:
        Queue capacity. ``0`` (default) means unbounded; a positive value
        applies back-pressure to ``feed()``.
    """

    def __init__(
        self,
        language: str,
        *,
        split_by_speaker: bool = False,
        id_start: int = 0,
        maxsize: int = 0,
    ) -> None:
        self._language = language
        self._split_by_speaker = split_by_speaker
        self._next_id = id_start
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def feed(self, segment: Segment) -> None:
        """Push one raw segment into the queue."""
        if self._closed:
            raise RuntimeError("PushQueueSource is closed; cannot feed")
        await self._queue.put(segment)

    async def close(self) -> None:
        """Signal end-of-stream. ``read()`` will flush and terminate."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_SENTINEL)

    async def read(self) -> AsyncIterator[SentenceRecord]:
        stream = Subtitle.stream(
            language=self._language,
            split_by_speaker=self._split_by_speaker,
        )
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                for rec in stream.flush_records():
                    yield self._tag(rec)
                return
            assert isinstance(item, Segment)
            for rec in stream.feed_records(item):
                yield self._tag(rec)

    def _tag(self, rec: SentenceRecord) -> SentenceRecord:
        from dataclasses import replace

        extra = dict(rec.extra or {})
        extra["id"] = self._next_id
        self._next_id += 1
        return replace(rec, extra=extra)


__all__ = ["PushQueueSource"]
