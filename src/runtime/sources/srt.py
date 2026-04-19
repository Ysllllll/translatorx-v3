"""SrtSource — parse an SRT file into a stream of sentence records.

Full design (D-073 / D-074):

- On cold start, the raw :class:`~model.Segment` list is persisted to the
  ``zzz_subtitle_jsonl/<video>.segments.jsonl`` sidecar and ``raw_segment_ref``
  is patched into the main video JSON via :meth:`Store.patch_video`.
- Optional preprocessing hooks follow the locked pipeline:
  ``apply_global("restore_punc") → clauses → apply_per_sentence("chunk_llm") → split``.
  Each hook is only executed when the caller supplies a callable.
- Video-level caches (``punc_cache``) and per-sentence caches
  (``chunk_cache["chunk_llm"]``) ride along each :class:`SentenceRecord` so
  downstream processors can persist them without re-executing the LLM.
- When neither ``store`` nor preprocessing hooks are supplied the source
  degrades to its pre-refactor behaviour: read file, split into sentences,
  yield records.  This keeps it usable in unit tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Callable

from model import SentenceRecord
from subtitle import Subtitle
from subtitle.io import read_srt

from runtime.protocol import VideoKey
from runtime.sources._common import assign_ids
from runtime.store import Store

ApplyFn = Callable[[list[str]], list[list[str]]]


class SrtSource:
    """Parse an SRT file into a stream of :class:`SentenceRecord`.

    Parameters
    ----------
    path:
        Path to the ``.srt`` file.
    language:
        Source language code (forwarded to :class:`Subtitle`).
    store, video_key:
        If both are supplied, the source persists the raw_segment sidecar
        and any populated ``punc_cache`` via :class:`Store`.
    restore_punc:
        Optional ``apply_global`` callable (text batch → list[list[str]]).
        When provided, a video-level ``punc_cache`` is reloaded from the
        store (if any) and passed through so repeats hit the cache.
    chunk_llm:
        Optional ``apply_per_sentence`` callable. The per-record output is
        stamped onto :attr:`SentenceRecord.chunk_cache` under key
        ``"chunk_llm"``.
    merge_under:
        Forwarded to :meth:`Subtitle.clauses`; skipped when ``None``.
    max_len:
        Forwarded to :meth:`Subtitle.split`; skipped when ``None``.
    split_by_speaker:
        Keep sentences speaker-isolated (forwarded to :class:`Subtitle`).
    id_start:
        Starting id for auto-assigned record ids.
    """

    def __init__(
        self,
        path: str | Path,
        language: str,
        *,
        store: Store | None = None,
        video_key: VideoKey | None = None,
        restore_punc: ApplyFn | None = None,
        chunk_llm: ApplyFn | None = None,
        merge_under: int | None = None,
        max_len: int | None = None,
        split_by_speaker: bool = False,
        id_start: int = 0,
    ) -> None:
        if (store is None) ^ (video_key is None):
            raise ValueError("store and video_key must be supplied together")
        self._path = Path(path)
        self._language = language
        self._store = store
        self._video_key = video_key
        self._restore_punc = restore_punc
        self._chunk_llm = chunk_llm
        self._merge_under = merge_under
        self._max_len = max_len
        self._split_by_speaker = split_by_speaker
        self._id_start = id_start

    async def read(self) -> AsyncIterator[SentenceRecord]:
        segments = await asyncio.to_thread(read_srt, self._path)

        # Cold write of raw_segment sidecar (D-069).
        if self._store is not None and self._video_key is not None:
            vid = self._video_key.video
            ref = await self._store.write_raw_segment(vid, segments, "srt")
            await self._store.patch_video(
                vid, segment_type="srt", raw_segment_ref=ref
            )

        sub = Subtitle(
            segments,
            language=self._language,
            split_by_speaker=self._split_by_speaker,
        )

        # Load any prior video-level punc_cache for warm hits.
        punc_cache: dict[str, list[str]] | None = None
        if self._restore_punc is not None:
            punc_cache = {}
            if self._store is not None and self._video_key is not None:
                prior = await self._store.load_video(self._video_key.video)
                punc_cache.update(prior.get("punc_cache") or {})
            sub = sub.apply_global("restore_punc", self._restore_punc, cache=punc_cache)

        sub = sub.sentences()
        if self._merge_under is not None:
            sub = sub.clauses(merge_under=self._merge_under)
        if self._chunk_llm is not None:
            sub = sub.apply_per_sentence("chunk_llm", self._chunk_llm)
        if self._max_len is not None:
            sub = sub.split(self._max_len)

        # Persist punc_cache if it was populated.
        if (
            punc_cache
            and self._store is not None
            and self._video_key is not None
        ):
            await self._store.patch_video(
                self._video_key.video, punc_cache=punc_cache
            )

        for rec in assign_ids(sub.records(), start=self._id_start):
            yield rec


__all__ = ["SrtSource"]

