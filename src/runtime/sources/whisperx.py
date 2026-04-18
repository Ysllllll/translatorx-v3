"""WhisperXSource — parse a WhisperX JSON into a stream of sentence records."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from model import SentenceRecord
from subtitle import Subtitle
from subtitle.io import read_whisperx

from runtime.sources._common import assign_ids


class WhisperXSource:
    """Parse a WhisperX JSON file into a stream of :class:`SentenceRecord`.

    WhisperX outputs word-level timings; :class:`Subtitle.from_words`
    assembles them into segments, then ``sentences()`` yields one record
    per complete sentence. Ids are auto-assigned starting at ``id_start``.

    Parameters
    ----------
    path:
        Path to the WhisperX ``.json`` file.
    language:
        Source language code.
    id_start:
        Starting id for ``extra["id"]`` allocation. Defaults to ``0``.
    """

    def __init__(
        self,
        path: str | Path,
        language: str,
        *,
        id_start: int = 0,
    ) -> None:
        self._path = Path(path)
        self._language = language
        self._id_start = id_start

    async def read(self) -> AsyncIterator[SentenceRecord]:
        words = read_whisperx(self._path)
        sub = Subtitle.from_words(words, language=self._language)
        records = sub.sentences().records()
        for rec in assign_ids(records, start=self._id_start):
            yield rec


__all__ = ["WhisperXSource"]

