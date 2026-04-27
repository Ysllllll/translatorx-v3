"""FastAPI + WebSocket application layer.

Only this module knows about FastAPI / starlette; the backends and
processors modules are plain async Python. That keeps the pipeline
stages importable + unit-testable without spinning up an ASGI app.

Lifecycle of one WS connection (``/ws/preprocess``)::

    accept                       ws.accept()
      │
      ▼
    warm backends                _get_punc / _get_chunk  (cached)
      │
      ▼
    send ready                   _safe_send(...)
      │
      ▼
    ┌──► recv loop               segment / flush / close
    │       │
    │       └─► feed PuncBufferStage ──► PushQueueSource
    │                                        │
    │                                        ▼
    │                                    PreprocessProcessor
    │                                        │
    │                                        ▼
    └──◄────── pump ◄──── send record to client

    close:
      drain PuncBufferStage.flush()
      close PushQueueSource (EOF)
      wait pump to drain remaining records  (≤ 600 s)
      send {"type": "done"}  + ws.close
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from domain.model import Segment, SentenceRecord

from adapters.sources.push import PushQueueSource

from .backends import build_chunk_fn, build_punc_fn
from .processors import PreprocessProcessor, PuncBufferStage


# ── process-wide backend cache ───────────────────────────────────────


# Shared across connections and with ``--warmup`` so each real punc /
# chunk model loads exactly once per (language, max_len).
_punc_cache: dict[str, Callable[[list[str]], list[list[str]]]] = {}
_chunk_cache: dict[tuple[str, int], Callable[[list[str]], list[list[str]]]] = {}
_cache_lock = asyncio.Lock()


async def _get_punc(language: str, real: bool) -> Callable[[list[str]], list[list[str]]]:
    async with _cache_lock:
        if language not in _punc_cache:
            _punc_cache[language] = await asyncio.to_thread(build_punc_fn, language, real)
        return _punc_cache[language]


async def _get_chunk(language: str, real: bool, engine_url: str | None, max_len: int) -> Callable[[list[str]], list[list[str]]]:
    key = (language, max_len)
    async with _cache_lock:
        if key not in _chunk_cache:
            _chunk_cache[key] = await asyncio.to_thread(build_chunk_fn, language, real, engine_url, max_len)
        return _chunk_cache[key]


# ── safe WebSocket IO ────────────────────────────────────────────────


# Three categories of "the WebSocket blew up" we care about:
#   - WebSocketDisconnect : starlette raises this on a clean close
#   - RuntimeError        : raised by starlette's receive_text/send_text
#                           when the socket is already closed at the
#                           protocol level ("WebSocket is not connected")
#   - ConnectionError     : lower-level TCP hang-up
_WS_DEAD = (WebSocketDisconnect, RuntimeError, ConnectionError)


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> bool:
    """Send ``payload`` as JSON. Return ``False`` if the socket is dead."""
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except _WS_DEAD:
        return False


async def _safe_close(ws: WebSocket, code: int = 1000) -> None:
    try:
        await ws.close(code=code)
    except _WS_DEAD:
        pass


# ── JSON encoding helpers ────────────────────────────────────────────


def _segment_to_dict(seg: Segment) -> dict[str, Any]:
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "speaker": seg.speaker,
        "words": [{"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker} for w in (seg.words or [])],
    }


def _record_to_dict(rec: SentenceRecord) -> dict[str, Any]:
    extra = rec.extra or {}
    return {
        "type": "record",
        "id": extra.get("id"),
        "start": rec.start,
        "end": rec.end,
        "src_text": rec.src_text,
        "segments": [_segment_to_dict(s) for s in rec.segments],
    }


# ── FastAPI factory ──────────────────────────────────────────────────


def build_app(*, real: bool, engine_url: str | None) -> FastAPI:
    app = FastAPI(title="translatorx-v3 · stream-preprocess demo")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "translatorx-v3 stream-preprocess demo",
            "ws": "/ws/preprocess",
            "real": real,
        }

    @app.websocket("/ws/preprocess")
    async def preprocess_ws(
        ws: WebSocket,
        language: str = Query("en"),
        restore_punc: bool = Query(True),
        max_len: int = Query(60),
        window: int = Query(4),
    ) -> None:
        await ws.accept()

        # Backend init may take seconds on first call. It's offloaded to
        # a worker thread inside _get_*, so the event loop stays
        # responsive while one connection warms up.
        try:
            punc_fn = await _get_punc(language, real) if restore_punc else None
            chunk_fn = await _get_chunk(language, real, engine_url, max_len)
        except Exception as exc:  # noqa: BLE001
            await _safe_send(ws, {"type": "error", "message": f"backend init failed: {exc!r}"})
            await _safe_close(ws, code=1011)
            return

        source = PushQueueSource(language=language)
        punc_buf = PuncBufferStage(punc_fn=punc_fn, downstream=source, window=window) if punc_fn is not None else None
        proc = PreprocessProcessor(language=language, chunk_fn=chunk_fn, max_len=max_len)

        if not await _safe_send(
            ws,
            {
                "type": "ready",
                "language": language,
                "restore_punc": restore_punc,
                "max_len": max_len,
                "window": window,
            },
        ):
            return  # peer vanished before we even sent ready

        # Shared flag so the pump stops pushing once we know the socket
        # is gone (otherwise send_text keeps raising after disconnect).
        disconnected = asyncio.Event()

        async def pump() -> None:
            try:
                async for out in proc.process(source.read()):
                    if disconnected.is_set():
                        break
                    if not await _safe_send(ws, _record_to_dict(out)):
                        disconnected.set()
                        break
            except Exception as exc:  # noqa: BLE001
                if not disconnected.is_set():
                    await _safe_send(ws, {"type": "error", "message": f"pump crashed: {exc!r}"})

        pump_task = asyncio.create_task(pump())

        async def _feed_segment(msg: dict[str, Any]) -> None:
            seg = Segment(
                start=float(msg.get("start", 0.0)),
                end=float(msg.get("end", 0.0)),
                text=str(msg.get("text", "")),
                speaker=msg.get("speaker"),
            )
            if punc_buf is not None:
                await punc_buf.feed(seg)
            else:
                await source.feed(seg)

        # ── main recv loop ──
        try:
            while True:
                try:
                    raw = await ws.receive_text()
                except _WS_DEAD:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    if not await _safe_send(ws, {"type": "error", "message": "invalid JSON"}):
                        break
                    continue

                mtype = msg.get("type")
                if mtype == "segment":
                    await _feed_segment(msg)
                elif mtype == "flush":
                    if punc_buf is not None:
                        await punc_buf.flush()
                elif mtype == "close":
                    break
                elif not await _safe_send(ws, {"type": "error", "message": f"unknown type: {mtype!r}"}):
                    break
        finally:
            # ── drain: let the pump finish any records still in flight ──
            if punc_buf is not None:
                try:
                    await punc_buf.flush()
                except Exception:  # noqa: BLE001
                    pass
            await source.close()
            try:
                # Real LLM chunker can be slow. Client requested all
                # records, so give the pump a generous window.
                await asyncio.wait_for(asyncio.shield(pump_task), timeout=600.0)
            except asyncio.TimeoutError:
                pump_task.cancel()

            # ── teardown ──
            disconnected.set()
            await proc.aclose()
            await _safe_send(ws, {"type": "done"})
            await _safe_close(ws)

    return app
