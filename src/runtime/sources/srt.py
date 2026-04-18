"""SrtSource — parse an SRT file into a stream of sentence records."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from model import SentenceRecord
from subtitle import Subtitle
from subtitle.io import read_srt

from runtime.sources._common import assign_ids


class SrtSource:
    """Parse an SRT file into a stream of :class:`SentenceRecord`.

    Each emitted record represents a full sentence (``Subtitle.sentences()``
    output) with its sub-segments attached. Ids are auto-assigned starting
    at ``id_start`` so downstream processors can persist via
    ``Store.patch_video``.

    Parameters
    ----------
    path:
        Path to the ``.srt`` file.
    language:
        Source language code (forwarded to :class:`Subtitle`).
    split_by_speaker:
        If True, sentences are not merged across speaker changes.
    id_start:
        Starting id for ``extra["id"]`` allocation. Defaults to ``0``.
    """

    def __init__(
        self,
        path: str | Path,
        language: str,
        *,
        split_by_speaker: bool = False,
        id_start: int = 0,
    ) -> None:
        self._path = Path(path)
        self._language = language
        self._split_by_speaker = split_by_speaker
        self._id_start = id_start

    async def read(self) -> AsyncIterator[SentenceRecord]:
        segments = read_srt(self._path)
        sub = Subtitle(
            segments,
            language=self._language,
            split_by_speaker=self._split_by_speaker,
        )
        records = sub.sentences().records()
        for rec in assign_ids(records, start=self._id_start):
            yield rec


__all__ = ["SrtSource"]
