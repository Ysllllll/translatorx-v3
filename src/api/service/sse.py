"""SSE helpers — async generator that pulls events from a Task's queue."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from api.service.runtime.tasks import Task


async def task_event_stream(task: Task) -> AsyncIterator[dict]:
    """Yield ``{event, data}`` dicts until the task reaches a terminal state.

    ``None`` on the queue is the sentinel indicating "no more events".
    """
    q = task.subscribe()
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat — keeps intermediaries from closing the
                # connection.
                yield {"event": "ping", "data": json.dumps({"task_id": task.task_id})}
                continue
            if item is None:
                break
            payload = dict(item)
            if isinstance(payload.get("data"), (dict, list)):
                payload["data"] = json.dumps(payload["data"])
            yield payload
    finally:
        task.unsubscribe(q)


__all__ = ["task_event_stream"]
