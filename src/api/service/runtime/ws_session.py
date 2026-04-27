"""WebSocket session — per-connection orchestration.

Wraps a single :class:`fastapi.WebSocket` and an :class:`App`, owns
the :class:`LiveStreamHandle` lifecycle, and pumps three concurrent
loops:

1. **Inbound** — receive frames, parse via
   :func:`api.service.runtime.ws_protocol.parse_client_frame`, dispatch
   to the appropriate handler.
2. **Records** — forward :meth:`LiveStreamHandle.records` items as
   :class:`WsFinal` frames.
3. **Events** — subscribe to ``channel.*`` / ``stage.*`` / ``bus.*``
   :class:`DomainEvent` and forward as :class:`WsProgress` /
   :class:`WsError` frames.

The session is fully cancel-safe: closing the WebSocket cancels all
three loops and triggers :meth:`LiveStreamHandle.close` under
``asyncio.shield`` so in-flight records are flushed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from api.service.runtime.ws_protocol import (
    WsAbort,
    WsAudioChunk,
    WsClosed,
    WsConfigUpdate,
    WsError,
    WsFinal,
    WsPing,
    WsPong,
    WsProgress,
    WsSegment,
    WsStart,
    WsStarted,
    dump_frame,
    parse_client_frame,
)
from domain.model import Segment

if TYPE_CHECKING:
    from api.app import App
    from api.app.stream import LiveStreamHandle


_logger = logging.getLogger(__name__)


class WsSession:
    """Owns one WebSocket connection's worth of streaming state.

    Lifecycle:

    * Created with the accepted :class:`WebSocket` and the bound
      :class:`App`.
    * :meth:`run` blocks until the connection closes (client abort,
      server error, transport drop). It sends :class:`WsClosed` as a
      best-effort final frame before returning.

    The session is **single-shot** — one ``start`` per connection.
    Subsequent ``start`` frames are answered with :class:`WsError`
    (``category="invalid_state"``) so clients can't accidentally
    multiplex.
    """

    __slots__ = (
        "_ws",
        "_app",
        "_handle",
        "_stream_id",
        "_records_task",
        "_events_task",
        "_subscription",
        "_send_lock",
        "_closed",
        "_tenant_id",
    )

    def __init__(self, ws: WebSocket, *, app: "App", tenant_id: str | None = None) -> None:
        self._ws = ws
        self._app = app
        self._handle: LiveStreamHandle | None = None
        self._stream_id: str | None = None
        self._records_task: asyncio.Task | None = None
        self._events_task: asyncio.Task | None = None
        self._subscription = None
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._tenant_id = tenant_id

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    async def run(self) -> None:
        reason = "completed"
        aborted = False
        try:
            while True:
                try:
                    raw = await self._ws.receive_text()
                except WebSocketDisconnect:
                    reason = "client_abort" if not aborted else reason
                    # Phase 4 🔴 #8 — best-effort closing frame on transport
                    # drop. _send is no-op when the socket is already gone,
                    # so this is safe even when the disconnect wasn't graceful.
                    with _suppress():
                        await self._send(WsClosed(reason=reason))
                    break

                try:
                    frame = parse_client_frame(raw)
                except ValidationError as exc:
                    await self._send(
                        WsError(category="invalid_frame", message=str(exc.errors()[:1])),
                    )
                    continue

                stop = await self._dispatch(frame)
                if stop:
                    aborted = True
                    reason = "client_abort"
                    if self._handle is not None and not self._handle.is_closed:
                        with _suppress():
                            await asyncio.shield(self._handle.close())
                        if self._records_task is not None:
                            try:
                                await asyncio.wait_for(asyncio.shield(self._records_task), timeout=5.0)
                            except asyncio.TimeoutError:
                                self._records_task.cancel()
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                _logger.debug("records task drained with error", exc_info=True)
                    await self._send(WsClosed(reason=reason))
                    break
        except Exception as exc:  # pragma: no cover - defensive
            _logger.exception("WsSession crashed: %s", exc)
            reason = "error"
            with _suppress():
                await self._send(WsError(category="internal", message=str(exc)))
            # C28 — always emit a terminal WsClosed frame so clients see
            # a deterministic end-of-stream marker rather than relying
            # on transport drop. _send is no-op once the socket is gone.
            with _suppress():
                await self._send(WsClosed(reason=reason))
        finally:
            try:
                await asyncio.shield(self._teardown(reason=reason))
            except asyncio.CancelledError:
                # Intentional swallow: teardown ran under shield() so no
                # in-flight work was lost. Re-raising would propagate into
                # Starlette's TestClient portal and crash unrelated tests
                # (see module docstring). Tightened from a previous
                # ``except BaseException`` so KeyboardInterrupt /
                # SystemExit / GeneratorExit still propagate normally.
                pass
            except Exception:
                _logger.debug("WsSession teardown failed", exc_info=True)

    # ------------------------------------------------------------------
    # Frame dispatch.
    # ------------------------------------------------------------------

    async def _dispatch(self, frame) -> bool:
        if isinstance(frame, WsStart):
            await self._handle_start(frame)
            return False
        if isinstance(frame, WsSegment):
            await self._handle_segment(frame)
            return False
        if isinstance(frame, WsAudioChunk):
            await self._send(
                WsError(
                    category="unsupported_frame",
                    message="audio_chunk requires a transcribe stage; not configured",
                ),
            )
            return False
        if isinstance(frame, WsConfigUpdate):
            await self._send(
                WsError(
                    category="unsupported_frame",
                    message="config_update is reserved for Phase 5",
                ),
            )
            return False
        if isinstance(frame, WsAbort):
            return True
        if isinstance(frame, WsPing):
            await self._send(WsPong())
            return False
        await self._send(WsError(category="invalid_frame", message="unknown frame"))
        return False

    # ------------------------------------------------------------------
    # Handlers.
    # ------------------------------------------------------------------

    async def _handle_start(self, frame: WsStart) -> None:
        if self._handle is not None:
            await self._send(
                WsError(category="invalid_state", message="stream already started"),
            )
            return

        try:
            builder = self._app.stream(
                course=frame.course,
                video=frame.video,
                language=frame.src,
            ).translate(src=frame.src, tgt=frame.tgt)
            if self._tenant_id is not None:
                # Phase 5 (方案 L) — admission control through FairScheduler.
                # WS clients get an immediate quota_exceeded close instead of
                # blocking the upgrade indefinitely.
                builder = builder.tenant(self._tenant_id, wait=False)
            self._handle = await builder.start_async()
        except Exception as exc:
            from application.scheduler import QuotaExceeded

            if isinstance(exc, QuotaExceeded):
                await self._send(
                    WsError(
                        category="quota_exceeded",
                        message=str(exc),
                    ),
                )
                # Best-effort closing frame so clients see a clean
                # rejection. Connection is then closed by the surrounding
                # ws_streams handler when run() returns.
                with _suppress():
                    await self._send(WsClosed(reason="quota_exceeded"))
                return
            await self._send(WsError(category="start_failed", message=str(exc)))
            return

        self._stream_id = uuid.uuid4().hex
        await self._send(WsStarted(stream_id=self._stream_id))

        self._records_task = asyncio.create_task(
            self._pump_records(),
            name=f"ws-records-{self._stream_id}",
        )
        # Events subscription is best-effort. If the App has no
        # EventBus we skip it entirely; the records pump is enough to
        # drive WsFinal output. Skipping when no real bus avoids stuck
        # cancellation paths during TestClient teardown.
        bus = getattr(self._app, "event_bus", None)
        if bus is not None and hasattr(bus, "subscribe"):
            self._events_task = asyncio.create_task(
                self._pump_events(course=frame.course, video=frame.video),
                name=f"ws-events-{self._stream_id}",
            )

    async def _handle_segment(self, frame: WsSegment) -> None:
        if self._handle is None:
            await self._send(
                WsError(category="invalid_state", message="must send 'start' first"),
            )
            return
        try:
            await self._handle.feed(
                Segment(
                    start=frame.start,
                    end=frame.end,
                    text=frame.text,
                    speaker=frame.speaker,
                ),
            )
        except Exception as exc:
            await self._send(WsError(category="feed_failed", message=str(exc)))

    # ------------------------------------------------------------------
    # Background pumps.
    # ------------------------------------------------------------------

    async def _pump_records(self) -> None:
        assert self._handle is not None
        try:
            async for rec in self._handle.records():
                tgt_text = ""
                # ``rec.translations`` is dict[lang, dict[variant_key, text]];
                # walk the first language's first variant.
                if rec.translations:
                    first_lang = next(iter(rec.translations.keys()))
                    candidate = rec.get_translation(first_lang)
                    if isinstance(candidate, str):
                        tgt_text = candidate
                await self._send(
                    WsFinal(
                        record_id=getattr(rec, "id", "") or f"rec-{int(rec.start * 1000)}",
                        src=rec.src_text,
                        tgt=tgt_text,
                        start=rec.start,
                        end=rec.end,
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with _suppress():
                await self._send(WsError(category="pipeline", message=str(exc)))

    async def _pump_events(self, *, course: str, video: str) -> None:
        bus = getattr(self._app, "event_bus", None)
        if bus is None or not hasattr(bus, "subscribe"):
            return
        try:
            sub = bus.subscribe(course=course, video=video)
        except Exception:
            return
        self._subscription = sub
        try:
            async for ev in sub:
                t = ev.type
                if t.startswith("channel."):
                    payload = ev.payload or {}
                    fill = payload.get("filled")
                    cap = payload.get("capacity")
                    ratio = (fill / cap) if (isinstance(fill, (int, float)) and isinstance(cap, (int, float)) and cap) else None
                    await self._send(
                        WsProgress(
                            stage=str(payload.get("stage_id", "")),
                            channel_fill=ratio,
                        ),
                    )
                elif t.startswith("stage.error") or t == "bus.publish_failed":
                    payload = ev.payload or {}
                    await self._send(
                        WsError(
                            category=t,
                            message=str(payload.get("error", "")),
                        ),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.debug("WsSession events pump exited", exc_info=True)

    # ------------------------------------------------------------------
    # Send + teardown.
    # ------------------------------------------------------------------

    async def _send(self, frame) -> None:
        if self._closed:
            return
        if self._ws.client_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            try:
                await self._ws.send_text(dump_frame(frame))
            except Exception:
                _logger.debug("WsSession send failed", exc_info=True)

    async def _teardown(self, *, reason: str) -> None:
        if self._closed:
            return
        self._closed = True

        # Stop the events pump first so we don't race the bus.
        if self._subscription is not None:
            with _suppress():
                self._subscription.close()
        if self._events_task is not None:
            self._events_task.cancel()
            with _suppress():
                await self._events_task

        # Close the handle (idempotent if abort already drove it).
        if self._handle is not None:
            try:
                await asyncio.shield(self._handle.close())
            except Exception:
                _logger.debug("LiveStreamHandle.close failed", exc_info=True)

        if self._records_task is not None and not self._records_task.done():
            try:
                await asyncio.wait_for(self._records_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._records_task.cancel()
                with _suppress():
                    await self._records_task
            except Exception:
                _logger.debug("records pump exited with error", exc_info=True)


class _suppress:
    """Mini async-aware ``contextlib.suppress(Exception)`` — keeps
    teardown noise-free without pulling in ``contextlib``.
    """

    def __enter__(self) -> "_suppress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, Exception)


__all__ = ["WsSession"]
