"""Tests for :class:`application.events.bus.EventBus`."""

from __future__ import annotations

import asyncio

import pytest

from application.events import DomainEvent, EventBus


@pytest.mark.asyncio
class TestEventBus:
    async def test_publish_no_subscribers_no_error(self):
        bus = EventBus()
        await bus.publish(DomainEvent(type="x.y", course="c"))

    async def test_single_subscriber_receives_event(self):
        bus = EventBus()
        sub = bus.subscribe()
        ev = DomainEvent(type="video.foo", course="c1", video="v1")
        await bus.publish(ev)
        got = await sub.get(timeout=1.0)
        assert got is not None
        assert got.event_id == ev.event_id
        sub.close()

    async def test_multiple_subscribers_all_receive(self):
        bus = EventBus()
        s1 = bus.subscribe()
        s2 = bus.subscribe()
        s3 = bus.subscribe()
        ev = DomainEvent(type="video.foo", course="c1")
        await bus.publish(ev)
        for s in (s1, s2, s3):
            got = await s.get(timeout=1.0)
            assert got is not None and got.event_id == ev.event_id
            s.close()

    async def test_filter_by_type_prefix(self):
        bus = EventBus()
        video_sub = bus.subscribe(type_prefix="video.")
        course_sub = bus.subscribe(type_prefix="course.")
        await bus.publish(DomainEvent(type="video.foo", course="c1"))
        await bus.publish(DomainEvent(type="course.bar", course="c1"))

        v = await video_sub.get(timeout=1.0)
        c = await course_sub.get(timeout=1.0)
        assert v is not None and v.type == "video.foo"
        assert c is not None and c.type == "course.bar"

        # Each subscriber should NOT receive the other's event
        assert await video_sub.get(timeout=0.05) is None
        assert await course_sub.get(timeout=0.05) is None
        video_sub.close()
        course_sub.close()

    async def test_filter_by_course(self):
        bus = EventBus()
        sub = bus.subscribe(course="c1")
        await bus.publish(DomainEvent(type="x", course="c1"))
        await bus.publish(DomainEvent(type="x", course="c2"))
        got = await sub.get(timeout=1.0)
        assert got is not None and got.course == "c1"
        assert await sub.get(timeout=0.05) is None
        sub.close()

    async def test_filter_by_video(self):
        bus = EventBus()
        sub = bus.subscribe(video="v1")
        await bus.publish(DomainEvent(type="x", course="c1", video="v1"))
        await bus.publish(DomainEvent(type="x", course="c1", video="v2"))
        got = await sub.get(timeout=1.0)
        assert got is not None and got.video == "v1"
        assert await sub.get(timeout=0.05) is None
        sub.close()

    async def test_close_unsubscribes(self):
        bus = EventBus()
        sub = bus.subscribe()
        assert bus.subscriber_count == 1
        sub.close()
        assert bus.subscriber_count == 0

    async def test_async_iterator(self):
        bus = EventBus()
        events = [DomainEvent(type=f"x.{i}", course="c") for i in range(3)]
        sub = bus.subscribe()
        for ev in events:
            await bus.publish(ev)
        sub.close()  # sentinel after all events buffered
        received = [ev async for ev in sub]
        assert [e.type for e in received] == ["x.0", "x.1", "x.2"]

    async def test_async_context_manager(self):
        bus = EventBus()
        async with bus.subscribe() as sub:
            assert bus.subscriber_count == 1
            await bus.publish(DomainEvent(type="x", course="c"))
            got = await sub.get(timeout=1.0)
            assert got is not None
        assert bus.subscriber_count == 0

    async def test_slow_subscriber_drops_no_block(self):
        bus = EventBus()
        sub = bus.subscribe(queue_size=2)
        for i in range(5):
            await bus.publish(DomainEvent(type=f"x.{i}", course="c"))
        # Queue fits 2; remaining 3 dropped
        assert sub.dropped == 3
        # The queue still yields the first 2
        a = await sub.get(timeout=0.5)
        b = await sub.get(timeout=0.5)
        assert a is not None and b is not None
        sub.close()

    async def test_publish_nowait(self):
        bus = EventBus()
        sub = bus.subscribe()
        bus.publish_nowait(DomainEvent(type="x", course="c"))
        got = await sub.get(timeout=1.0)
        assert got is not None
        sub.close()

    async def test_close_bus_closes_all_subs(self):
        bus = EventBus()
        s1 = bus.subscribe()
        s2 = bus.subscribe()
        await bus.close()
        # close puts None sentinel — async iter exits
        assert [ev async for ev in s1] == []
        assert [ev async for ev in s2] == []

    async def test_get_timeout_returns_none(self):
        bus = EventBus()
        sub = bus.subscribe()
        got = await sub.get(timeout=0.05)
        assert got is None
        sub.close()


@pytest.mark.asyncio
class TestVideoSessionEmitsEvents:
    """Integration: VideoSession.flush publishes DomainEvent on EventBus."""

    async def test_flush_emits_records_patched_and_fingerprints(self, tmp_path):
        from adapters.storage.store import JsonFileStore
        from adapters.storage.workspace import Workspace
        from application.events import EventBus
        from application.orchestrator.session import VideoSession
        from domain.model import SentenceRecord
        from ports.source import VideoKey

        bus = EventBus()
        sub = bus.subscribe(type_prefix="video.")

        ws = Workspace(root=tmp_path, course="c1")
        store = JsonFileStore(ws)
        vk = VideoKey(course="c1", video="v1")
        sess = await VideoSession.load(store, vk, event_bus=bus)

        rec = SentenceRecord(src_text="hello", start=0.0, end=1.0, segments=[], extra={"id": 0})
        sess.set_translation(rec, "zh", "default", "你好", prompt_id="default", prompt="p")
        sess.set_fingerprint("translate", "fp123")

        await sess.flush(store)

        # Should receive two events
        ev1 = await sub.get(timeout=1.0)
        ev2 = await sub.get(timeout=1.0)
        assert ev1 is not None and ev2 is not None
        types = {ev1.type, ev2.type}
        assert types == {"video.records_patched", "video.fingerprints_set"}

        records_ev = ev1 if ev1.type == "video.records_patched" else ev2
        assert records_ev.course == "c1"
        assert records_ev.video == "v1"
        assert records_ev.payload["record_ids"] == [0]

        sub.close()

    async def test_flush_without_bus_is_safe(self, tmp_path):
        from adapters.storage.store import JsonFileStore
        from adapters.storage.workspace import Workspace
        from application.orchestrator.session import VideoSession
        from domain.model import SentenceRecord
        from ports.source import VideoKey

        ws = Workspace(root=tmp_path, course="c1")
        store = JsonFileStore(ws)
        vk = VideoKey(course="c1", video="v1")
        sess = await VideoSession.load(store, vk)
        rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, segments=[], extra={"id": 0})
        sess.set_translation(rec, "zh", "default", "你好", prompt_id="default", prompt="p")
        await sess.flush(store)  # no bus → no-op
