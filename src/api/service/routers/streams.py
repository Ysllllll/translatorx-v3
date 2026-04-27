"""Streams router — live translation streams (StreamBuilder-backed)."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from api.service.auth import Principal, RequirePrincipal
from api.service.schemas import CreateStreamRequest, StreamInfo, StreamSegmentIn
from api.service.runtime.stream_registry import InMemoryStreamRegistry, LiveStream
from domain.model import Segment


router = APIRouter(prefix="/api/streams", tags=["streams"])


# Re-export for callers that still reference the old name.
_LiveStream = LiveStream


def _registry(request: Request):
    reg = getattr(request.app.state, "streams", None)
    if reg is None:
        reg = InMemoryStreamRegistry()
        request.app.state.streams = reg
    # Back-compat: if someone stashed a plain dict (tests), wrap lookups.
    if isinstance(reg, dict):

        class _DictShim:
            def __init__(self, d):
                self._d = d

            def get(self, sid):
                return self._d.get(sid)

            def put(self, s):
                self._d[s.stream_id] = s

            def remove(self, sid):
                self._d.pop(sid, None)

            def values(self):
                return list(self._d.values())

            def list_ids(self):
                return list(self._d.keys())

            async def close(self):
                self._d.clear()

        return _DictShim(reg)
    return reg


async def _pump(stream: _LiveStream) -> None:
    try:
        async for rec in stream.handle.records():
            try:
                stream.queue.put_nowait(
                    {
                        "event": "record",
                        "data": json.dumps(
                            {
                                "src_text": rec.src_text,
                                "translations": dict(rec.translations),
                                "start": rec.start,
                                "end": rec.end,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            except asyncio.QueueFull:
                pass
    finally:
        stream.status = "closed"
        stream.queue.put_nowait(None)


@router.post("", response_model=StreamInfo, status_code=status.HTTP_201_CREATED)
async def open_stream(
    body: CreateStreamRequest,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> StreamInfo:
    app = request.app.state.app
    builder = app.stream(course=body.course, video=body.video, language=body.src)
    builder = builder.translate(src=body.src, tgt=body.tgt, engine=body.engine)
    if principal.tenant is not None:
        # Phase 5 (方案 L) — admission control. SSE clients get HTTP 429 when
        # their tenant cap is exhausted; ``wait=False`` keeps the request
        # short instead of holding a TCP connection on the queue.
        builder = builder.tenant(principal.tenant, wait=False)

    try:
        handle = await builder.start_async()
    except Exception as exc:
        from application.scheduler import QuotaExceeded

        if isinstance(exc, QuotaExceeded):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"quota_exceeded: {exc}",
            )
        raise

    reg = _registry(request)
    sid = uuid.uuid4().hex
    stream = _LiveStream(
        stream_id=sid,
        course=body.course,
        video=body.video,
        src=body.src,
        tgt=body.tgt,
        handle=handle,
        principal_user_id=principal.user_id,
    )
    reg.put(stream)
    stream.pump_task = asyncio.get_running_loop().create_task(_pump(stream))
    return StreamInfo(
        stream_id=sid,
        course=stream.course,
        video=stream.video,
        src=stream.src,
        tgt=stream.tgt,
        status=stream.status,
    )


def _assert_stream_owner(stream: _LiveStream, principal: Principal) -> None:
    """R1 — only the stream's submitting principal may push, observe or
    close it. Anonymous (dev) streams are accessible to all anonymous
    callers (legacy behaviour).
    """
    if stream.principal_user_id is None:
        return
    if stream.principal_user_id == principal.user_id:
        return
    raise HTTPException(status_code=404, detail="stream not found")


@router.post("/{stream_id}/segments", status_code=status.HTTP_202_ACCEPTED)
async def push_segment(
    stream_id: str,
    body: StreamSegmentIn,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> dict[str, str]:
    stream = _registry(request).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    _assert_stream_owner(stream, principal)
    if stream.status != "open":
        raise HTTPException(status_code=409, detail="stream is not open")
    await stream.handle.feed(
        Segment(start=body.start, end=body.end, text=body.text, speaker=body.speaker),
    )
    return {"status": "accepted"}


@router.get("/{stream_id}/events")
async def stream_events(
    stream_id: str,
    request: Request,
    principal: Principal = RequirePrincipal,
):
    stream = _registry(request).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    _assert_stream_owner(stream, principal)

    async def gen() -> AsyncIterator[dict]:
        while True:
            try:
                item = await asyncio.wait_for(stream.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": json.dumps({"stream_id": stream.stream_id})}
                continue
            if item is None:
                break
            yield item

    return EventSourceResponse(gen())


@router.post("/{stream_id}/close", response_model=StreamInfo)
async def close_stream(
    stream_id: str,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> StreamInfo:
    reg = _registry(request)
    stream = reg.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    _assert_stream_owner(stream, principal)
    stream.status = "closing"
    await stream.handle.close()
    if stream.pump_task is not None:
        try:
            await asyncio.wait_for(stream.pump_task, timeout=30.0)
        except asyncio.TimeoutError:
            stream.pump_task.cancel()
    stream.status = "closed"
    return StreamInfo(
        stream_id=stream.stream_id,
        course=stream.course,
        video=stream.video,
        src=stream.src,
        tgt=stream.tgt,
        status=stream.status,
    )


__all__ = ["router"]
