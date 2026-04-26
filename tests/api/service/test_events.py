"""Tests for ``GET /api/events/stream`` SSE endpoint and cross-thread bus."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from application.events import DomainEvent, EventBus
from api.service import create_app
from tests.api.service._helpers import make_app


def test_events_stream_route_registered(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    routes = {r.path for r in api.routes}
    assert "/api/events/stream" in routes


def test_publish_nowait_cross_thread() -> None:
    """``publish_nowait`` from a thread without a running loop must wake
    a subscriber awaiting on the owner loop (see ``Subscription._try_put``).

    This is the unit-level proof that the SSE end-to-end path is sound:
    the route's gen runs on the ASGI worker loop and blocks on
    ``sub.get(...)``; publishers from other threads (e.g. orchestrator
    work threads) must be able to wake it up.
    """
    bus = EventBus()
    received: list[DomainEvent] = []
    ready = threading.Event()
    done = threading.Event()

    def worker() -> None:
        async def main() -> None:
            sub = bus.subscribe(type_prefix="x.")
            ready.set()
            ev = await sub.get(timeout=2.0)
            if ev is not None:
                received.append(ev)
            sub.close()

        asyncio.run(main())
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    assert ready.wait(timeout=2.0)
    # Give the worker a moment to enter sub.get() before publishing.
    time.sleep(0.05)
    bus.publish_nowait(DomainEvent(type="x.hello", course="c", video="v"))
    assert done.wait(timeout=3.0)
    t.join()

    assert len(received) == 1
    assert received[0].type == "x.hello"


def test_publish_nowait_same_thread_returns_full_status() -> None:
    """When called from the loop owning the subscription, ``publish_nowait``
    should still report drops accurately (``Subscription.dropped`` and
    return value of ``_try_put``)."""

    async def main() -> None:
        bus = EventBus()
        sub = bus.subscribe(type_prefix="", queue_size=1)
        bus.publish_nowait(DomainEvent(type="a", course="c", video="v"))
        bus.publish_nowait(DomainEvent(type="b", course="c", video="v"))  # drops
        assert sub.dropped == 1
        sub.close()

    asyncio.run(main())
