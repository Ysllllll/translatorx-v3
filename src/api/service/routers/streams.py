"""Streams router — live translation streams (StreamBuilder-backed)."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from api.app.stream import LiveStreamHandle
from api.service.auth import Principal, RequirePrincipal
from api.service.schemas import CreateStreamRequest, StreamInfo, StreamSegmentIn
from domain.model import Segment


router = APIRouter(prefix="/api/streams", tags=["streams"])


@dataclass
class _LiveStream:
    stream_id: str
    course: str
    video: str
    src: str
    tgt: str
    handle: LiveStreamHandle
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pump_task: asyncio.Task | None = None
    status: str = "open"


def _registry(request: Request) -> dict[str, _LiveStream]:
    reg: dict[str, _LiveStream] | None = getattr(request.app.state, "streams", None)
    if reg is None:
        reg = {}
        request.app.state.streams = reg
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
    _p: Principal = RequirePrincipal,
) -> StreamInfo:
    app = request.app.state.app
    builder = app.stream(course=body.course, video=body.video, language=body.src)
    builder = builder.translate(src=body.src, tgt=body.tgt, engine=body.engine)
    handle = builder.start()

    reg = _registry(request)
    sid = uuid.uuid4().hex
    stream = _LiveStream(
        stream_id=sid,
        course=body.course,
        video=body.video,
        src=body.src,
        tgt=body.tgt,
        handle=handle,
    )
    reg[sid] = stream
    stream.pump_task = asyncio.get_running_loop().create_task(_pump(stream))
    return StreamInfo(
        stream_id=sid,
        course=stream.course,
        video=stream.video,
        src=stream.src,
        tgt=stream.tgt,
        status=stream.status,
    )


@router.post("/{stream_id}/segments", status_code=status.HTTP_202_ACCEPTED)
async def push_segment(
    stream_id: str,
    body: StreamSegmentIn,
    request: Request,
    _p: Principal = RequirePrincipal,
) -> dict[str, str]:
    stream = _registry(request).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
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
    _p: Principal = RequirePrincipal,
):
    stream = _registry(request).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

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
    _p: Principal = RequirePrincipal,
) -> StreamInfo:
    reg = _registry(request)
    stream = reg.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
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
