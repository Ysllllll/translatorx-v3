"""Build-tier Stage adapters — wrap legacy ``Source`` classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from adapters.sources.push import PushQueueSource
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource
from domain.model import SentenceRecord
from ports.source import VideoKey

__all__ = [
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
