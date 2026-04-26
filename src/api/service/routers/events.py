"""Events router — global SSE fan-out from the App-level :class:`EventBus`.

``GET /api/events/stream`` opens a long-lived SSE connection that yields
every :class:`DomainEvent` published on ``request.app.state.app.event_bus``.
Optional query parameters (``type_prefix``, ``course``, ``video``) filter
the stream server-side via :meth:`EventBus.subscribe`.

Heartbeat ``ping`` events are emitted every 30 seconds to keep proxies
from closing idle connections (mirrors :mod:`api.service.sse`).
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from api.service.auth import Principal, RequirePrincipal


router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("/stream")
async def events_stream(
    request: Request,
    type_prefix: str | None = Query(default=None),
    course: str | None = Query(default=None),
    video: str | None = Query(default=None),
    _p: Principal = RequirePrincipal,
):
    """Subscribe to the global :class:`EventBus` as Server-Sent Events.

    Each emitted SSE message has ``event=<DomainEvent.type>`` and
    ``data=<json of DomainEvent.to_dict()>``.
    """
    app = request.app.state.app
    bus = app.event_bus

    sub = bus.subscribe(type_prefix=type_prefix, course=course, video=video)

    async def gen() -> AsyncIterator[dict]:
        try:
            while not sub._closed:
                item = await sub.get(timeout=30.0)
                if item is None:
                    if sub._closed:
                        break
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {
                    "event": item.type,
                    "data": json.dumps(item.to_dict(), ensure_ascii=False),
                }
        finally:
            sub.close()

    return EventSourceResponse(gen())


__all__ = ["router"]
