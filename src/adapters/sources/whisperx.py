"""WhisperXSource — parse a WhisperX JSON into a stream of sentence records.

Mirrors :class:`SrtSource` (D-073 / D-074): optional Store-backed raw_segment
sidecar (``<video>.words.jsonl``), optional preprocessing hooks
(``restore_punc`` / ``chunk``), and video-level cache persistence.

``punc_position`` controls where punctuation restoration runs (see
:class:`SrtSource` for details).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Callable, Literal

from domain.model import SentenceRecord
from domain.subtitle import Subtitle
from adapters.parsers import read_whisperx

from ports.source import VideoKey
from adapters.sources.common import assign_ids
from adapters.storage.store import Store

ApplyFn = Callable[[list[str]], list[list[str]]]


class WhisperXSource:
    """Parse a WhisperX JSON file into a stream of :class:`SentenceRecord`.

    Parameters match :class:`SrtSource`. See that class for detailed docs.
    ``segment_type`` is always ``"whisperx"``.
    """

    def __init__(
        self,
        path: str | Path,
        language: str,
        *,
        store: Store | None = None,
        video_key: VideoKey | None = None,
        restore_punc: ApplyFn | None = None,
        punc_position: Literal["global", "sentence", "both"] = "global",
        chunk_llm: ApplyFn | None = None,
        merge_under: int | None = None,
        max_len: int | None = None,
        id_start: int = 0,
    ) -> None:
        if (store is None) ^ (video_key is None):
            raise ValueError("store and video_key must be supplied together")
        self._path = Path(path)
        self._language = language
        self._store = store
        self._video_key = video_key
        self._restore_punc = restore_punc
        self._punc_position = punc_position
        self._chunk_llm = chunk_llm
        self._merge_under = merge_under
        self._max_len = max_len
        self._id_start = id_start

    async def read(self) -> AsyncIterator[SentenceRecord]:
        words = await asyncio.to_thread(read_whisperx, self._path)

        # Cold write of raw_segment sidecar (word-level).
        if self._store is not None and self._video_key is not None:
            vid = self._video_key.video
            ref = await self._store.write_raw_segment(vid, words, "whisperx")
            await self._store.patch_video(vid, segment_type="whisperx", raw_segment_ref=ref)

        sub = Subtitle.from_words(words, language=self._language)

        # Load video-level caches for warm hits.
        punc_cache: dict[str, list[str]] | None = None
        chunk_cache: dict[str, list[str]] | None = None

        if self._store is not None and self._video_key is not None:
            prior = await self._store.load_video(self._video_key.video)
            if self._restore_punc is not None:
                punc_cache = dict(prior.get("punc_cache") or {})
            if self._chunk_llm is not None:
                chunk_cache = dict(prior.get("chunk_cache") or {})

        try:
            # ① Global punc — before sentences()
            if self._restore_punc is not None and self._punc_position in ("global", "both"):
                sub = sub.transform(self._restore_punc, cache=punc_cache, scope="joined")

            # ② Sentence splitting
            sub = sub.sentences()

            # ③ Per-sentence punc — after sentences()
            if self._restore_punc is not None and self._punc_position in (
                "sentence",
                "both",
            ):
                sub = sub.transform(self._restore_punc, cache=punc_cache, scope="joined")

            # ④ Clause splitting
            if self._merge_under is not None:
                sub = sub.clauses(merge_under=self._merge_under)

            # ⑤ Chunking
            if self._chunk_llm is not None:
                sub = sub.transform(self._chunk_llm, cache=chunk_cache)

            # ⑥ Length-based split fallback
            if self._max_len is not None:
                sub = sub.split(self._max_len)
        finally:
            # Persist caches even on failure so partial LLM results are not lost.
            if self._store is not None and self._video_key is not None:
                vid = self._video_key.video
                if punc_cache:
                    await self._store.patch_video(vid, punc_cache=punc_cache)
                if chunk_cache:
                    await self._store.patch_video(vid, chunk_cache=chunk_cache)

        for rec in assign_ids(sub.records(), start=self._id_start):
            yield rec


__all__ = ["WhisperXSource"]
