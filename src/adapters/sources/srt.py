"""SrtSource — parse an SRT file into a stream of sentence records.

Full design (D-073 / D-074):

- On cold start, the raw :class:`~model.Segment` list is persisted to the
  ``zzz_subtitle_jsonl/<video>.segments.jsonl`` sidecar and ``raw_segment_ref``
  is patched into the main video JSON via :meth:`Store.patch_video`.
- Optional preprocessing hooks follow the locked pipeline:
  ``transform(restore_punc, scope="joined") → sentences →
  transform(restore_punc, scope="joined") → clauses →
  transform(chunk, scope="chunk") → split``.
  Each hook is only executed when the caller supplies a callable.
- ``punc_position`` controls where punctuation restoration runs:
  ``"global"`` (before sentences), ``"sentence"`` (after sentences),
  ``"both"`` (both positions).
- Video-level caches (``punc_cache``, ``chunk_cache``) are loaded from
  and persisted to the Store via :meth:`Store.patch_video`.  This avoids
  redundant LLM calls on resume.
- When neither ``store`` nor preprocessing hooks are supplied the source
  degrades to its pre-refactor behaviour: read file, split into sentences,
  yield records.  This keeps it usable in unit tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Callable, Literal

from domain.model import SentenceRecord
from domain.subtitle import Subtitle
from adapters.parsers import read_srt

from ports.source import VideoKey
from adapters.sources.common import assign_ids
from adapters.storage.store import Store

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
        and any populated caches via :class:`Store`.
    restore_punc:
        Optional callable (text batch → list[list[str]]).
        Used at the position(s) indicated by *punc_position*.
    punc_position:
        ``"global"`` — run before ``sentences()`` (helps sentence splitting).
        ``"sentence"`` — run after ``sentences()`` (fixes per-sentence punc).
        ``"both"`` — run at both positions.
    chunk_llm:
        Optional transform callable for LLM-based chunking.
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
        punc_position: Literal["global", "sentence", "both"] = "global",
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
        self._punc_position = punc_position
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
            await self._store.patch_video(vid, segment_type="srt", raw_segment_ref=ref)

        sub = Subtitle(
            segments,
            language=self._language,
            split_by_speaker=self._split_by_speaker,
        )

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

            # ⑤ Chunking (spaCy or LLM)
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


__all__ = ["SrtSource"]
