"""Build-tier Stage adapters — wrap legacy ``Source`` classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel, Field

from adapters.sources.push import PushQueueSource
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource
from domain.model import SentenceRecord
from ports.source import VideoKey
from ports.transcriber import TranscribeOptions, Transcriber

__all__ = [
    "FromAudioParams",
    "FromAudioStage",
    "FromPushParams",
    "FromPushStage",
    "FromSrtParams",
    "FromSrtStage",
    "FromWhisperxParams",
    "FromWhisperxStage",
]


# ---------------------------------------------------------------------------
# from_srt
# ---------------------------------------------------------------------------


class FromSrtParams(BaseModel):
    path: Path
    language: str
    split_by_speaker: bool = False
    id_start: int = 0


class FromSrtStage:
    """Wrap :class:`SrtSource` as a :class:`SourceStage`.

    Pulls ``store`` and ``video_key`` from the runtime ``ctx`` so the
    adapter doesn't duplicate session state.
    """

    name = "from_srt"

    __slots__ = ("_params", "_iter", "_source")

    def __init__(self, params: FromSrtParams) -> None:
        self._params = params
        self._iter: AsyncIterator[SentenceRecord] | None = None
        self._source: SrtSource | None = None

    async def open(self, ctx: Any) -> None:
        store = getattr(ctx, "store", None)
        vk: VideoKey | None = getattr(getattr(ctx, "session", None), "video_key", None)
        kw: dict[str, Any] = {}
        if store is not None and vk is not None:
            kw["store"] = store
            kw["video_key"] = vk
        self._source = SrtSource(
            self._params.path,
            language=self._params.language,
            split_by_speaker=self._params.split_by_speaker,
            id_start=self._params.id_start,
            **kw,
        )
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None, "FromSrtStage.open() must be called first"
        return self._iter

    async def close(self) -> None:
        self._iter = None
        self._source = None


# ---------------------------------------------------------------------------
# from_whisperx
# ---------------------------------------------------------------------------


class FromWhisperxParams(BaseModel):
    path: Path
    language: str
    id_start: int = 0


class FromWhisperxStage:
    """Wrap :class:`WhisperXSource` as a :class:`SourceStage`."""

    name = "from_whisperx"

    __slots__ = ("_params", "_iter", "_source")

    def __init__(self, params: FromWhisperxParams) -> None:
        self._params = params
        self._iter: AsyncIterator[SentenceRecord] | None = None
        self._source: WhisperXSource | None = None

    async def open(self, ctx: Any) -> None:
        store = getattr(ctx, "store", None)
        vk: VideoKey | None = getattr(getattr(ctx, "session", None), "video_key", None)
        kw: dict[str, Any] = {}
        if store is not None and vk is not None:
            kw["store"] = store
            kw["video_key"] = vk
        self._source = WhisperXSource(
            self._params.path,
            language=self._params.language,
            id_start=self._params.id_start,
            **kw,
        )
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None
        return self._iter

    async def close(self) -> None:
        self._iter = None
        self._source = None


# ---------------------------------------------------------------------------
# from_push
# ---------------------------------------------------------------------------


class FromPushParams(BaseModel):
    language: str
    split_by_speaker: bool = False
    id_start: int = 0
    maxsize: int = Field(default=0, ge=0)


class FromPushStage:
    """Wrap :class:`PushQueueSource` as a :class:`SourceStage`.

    The wrapped source is exposed via :attr:`source` so the API layer
    (e.g. an SSE feed endpoint) can call ``feed()`` / ``close()`` on it.
    """

    name = "from_push"

    __slots__ = ("_params", "_source", "_iter")

    def __init__(self, params: FromPushParams) -> None:
        self._params = params
        self._source: PushQueueSource | None = None
        self._iter: AsyncIterator[SentenceRecord] | None = None

    @property
    def source(self) -> PushQueueSource:
        assert self._source is not None, "FromPushStage.open() must be called first"
        return self._source

    async def open(self, ctx: Any) -> None:
        self._source = PushQueueSource(
            language=self._params.language,
            split_by_speaker=self._params.split_by_speaker,
            id_start=self._params.id_start,
            maxsize=self._params.maxsize,
        )
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None
        return self._iter

    async def close(self) -> None:
        if self._source is not None:
            await self._source.close()
        self._iter = None
        self._source = None


# ---------------------------------------------------------------------------
# from_audio  — transcribe + WhisperX-shaped pipeline source
# ---------------------------------------------------------------------------


class FromAudioParams(BaseModel):
    """Params for :class:`FromAudioStage`.

    The transcriber is resolved by the :func:`make_default_registry`
    factory using ``library`` (mirrors :meth:`App.transcriber`). The
    stage transcribes ``audio_path`` to a WhisperX-shaped JSON file
    under the workspace, then delegates streaming to
    :class:`WhisperXSource`.
    """

    audio_path: Path
    library: str | None = None
    language: str | None = None
    word_timestamps: bool = True
    id_start: int = 0


class FromAudioStage:
    """Run a :class:`Transcriber`, persist WhisperX JSON, then stream records.

    The stage takes closures (transcriber, ``json_path_resolver``,
    optional punc/chunk apply fns) so it can stay free of any direct
    :class:`App` dependency. The registry is responsible for binding
    them.

    The detected language is exposed via :attr:`language` once
    :meth:`open` returns, so downstream stages (``punc``, ``chunk``)
    that were configured with ``language="auto"`` can reconcile.
    """

    name = "from_audio"

    __slots__ = (
        "_params",
        "_transcriber",
        "_json_path_resolver",
        "_punc_factory",
        "_chunk_factory",
        "_punc_position",
        "_merge_under",
        "_max_len",
        "_iter",
        "_source",
        "_language",
    )

    def __init__(
        self,
        params: FromAudioParams,
        *,
        transcriber: Transcriber,
        json_path_resolver: Callable[[VideoKey], Path],
        punc_factory: Callable[[str], Any] | None = None,
        chunk_factory: Callable[[str], Any] | None = None,
        punc_position: str = "head",
        merge_under: int = 0,
        max_len: int = 0,
    ) -> None:
        self._params = params
        self._transcriber = transcriber
        self._json_path_resolver = json_path_resolver
        self._punc_factory = punc_factory
        self._chunk_factory = chunk_factory
        self._punc_position = punc_position
        self._merge_under = merge_under
        self._max_len = max_len
        self._iter: AsyncIterator[SentenceRecord] | None = None
        self._source: WhisperXSource | None = None
        self._language: str | None = None

    @property
    def language(self) -> str | None:
        """Language detected by the transcriber (set after :meth:`open`)."""
        return self._language

    async def open(self, ctx: Any) -> None:
        store = getattr(ctx, "store", None)
        vk: VideoKey | None = getattr(getattr(ctx, "session", None), "video_key", None)

        opts = TranscribeOptions(
            language=self._params.language,
            word_timestamps=self._params.word_timestamps,
        )
        result = await self._transcriber.transcribe(self._params.audio_path, opts)

        detected = result.language or self._params.language or ""
        if not detected:
            raise ValueError(
                "FromAudioStage: transcriber did not return a language and none supplied",
            )
        self._language = detected

        if vk is None:
            raise RuntimeError(
                "FromAudioStage requires a VideoKey on ctx.session to resolve the JSON path",
            )
        json_path = self._json_path_resolver(vk)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": detected,
            "duration": result.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in result.segments
            ],
            "word_segments": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "speaker": w.speaker,
                }
                for seg in result.segments
                for w in seg.words
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        kw: dict[str, Any] = {}
        if store is not None:
            kw["store"] = store
            kw["video_key"] = vk
        if self._punc_factory is not None:
            fn = self._punc_factory(detected)
            if fn is not None:
                kw["restore_punc"] = fn
                kw["punc_position"] = self._punc_position
        if self._chunk_factory is not None:
            fn = self._chunk_factory(detected)
            if fn is not None:
                kw["chunk_llm"] = fn
                kw["merge_under"] = self._merge_under
                kw["max_len"] = self._max_len

        self._source = WhisperXSource(
            json_path,
            language=detected,
            id_start=self._params.id_start,
            **kw,
        )
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None, "FromAudioStage.open() must be called first"
        return self._iter

    async def close(self) -> None:
        self._iter = None
        self._source = None
