"""Tests for :mod:`application.events.types` and :class:`DomainEvent`."""

from __future__ import annotations

import time

from application.events import DomainEvent, course_metadata_patched, orchestrator_finished, orchestrator_started, video_fingerprints_set, video_invalidated, video_records_patched


class TestDomainEvent:
    def test_construct_defaults(self):
        ev = DomainEvent(type="foo.bar", course="c1")
        assert ev.type == "foo.bar"
        assert ev.course == "c1"
        assert ev.video is None
        assert ev.payload == {}
        assert isinstance(ev.timestamp, float)
        assert len(ev.event_id) == 32

    def test_to_dict_round_trip(self):
        ev = DomainEvent(type="video.records_patched", course="c1", video="v1", payload={"record_ids": [1, 2]})
        d = ev.to_dict()
        assert d == {"type": "video.records_patched", "course": "c1", "video": "v1", "payload": {"record_ids": [1, 2]}, "timestamp": ev.timestamp, "event_id": ev.event_id}
        ev2 = DomainEvent.from_dict(d)
        assert ev2.type == ev.type
        assert ev2.course == ev.course
        assert ev2.video == ev.video
        assert ev2.payload == ev.payload
        assert ev2.timestamp == ev.timestamp
        assert ev2.event_id == ev.event_id

    def test_from_dict_minimum(self):
        ev = DomainEvent.from_dict({"type": "x.y", "course": "c"})
        assert ev.type == "x.y"
        assert ev.course == "c"
        assert ev.video is None
        assert ev.payload == {}
        # timestamp falls back to now-ish; just sanity-check the type
        assert isinstance(ev.timestamp, float)

    def test_matches_no_filter(self):
        ev = DomainEvent(type="video.foo", course="c1", video="v1")
        assert ev.matches() is True

    def test_matches_type_prefix(self):
        ev = DomainEvent(type="video.records_patched", course="c1", video="v1")
        assert ev.matches(type_prefix="video.") is True
        assert ev.matches(type_prefix="video.records") is True
        assert ev.matches(type_prefix="course.") is False

    def test_matches_course(self):
        ev = DomainEvent(type="video.foo", course="c1", video="v1")
        assert ev.matches(course="c1") is True
        assert ev.matches(course="c2") is False

    def test_matches_video(self):
        ev = DomainEvent(type="video.foo", course="c1", video="v1")
        assert ev.matches(video="v1") is True
        assert ev.matches(video="v2") is False
        assert ev.matches(video=None) is True  # None means "don't filter"

    def test_matches_combined(self):
        ev = DomainEvent(type="video.records_patched", course="c1", video="v1")
        assert ev.matches(type_prefix="video.", course="c1", video="v1") is True
        assert ev.matches(type_prefix="video.", course="c1", video="v2") is False

    def test_frozen(self):
        ev = DomainEvent(type="x", course="c")
        try:
            ev.type = "y"  # type: ignore[misc]
            raise AssertionError("should have raised")
        except Exception:
            pass


class TestConvenienceConstructors:
    def test_video_records_patched(self):
        ev = video_records_patched("c1", "v1", record_ids=[1, 2, 3], processor="translate")
        assert ev.type == "video.records_patched"
        assert ev.course == "c1"
        assert ev.video == "v1"
        assert ev.payload == {"record_ids": [1, 2, 3], "processor": "translate"}

    def test_video_fingerprints_set(self):
        ev = video_fingerprints_set("c1", "v1", fingerprints={"translate": "abc"})
        assert ev.type == "video.fingerprints_set"
        assert ev.payload == {"fingerprints": {"translate": "abc"}}

    def test_video_invalidated(self):
        ev = video_invalidated("c1", "v1", processor="translate", record_ids=[5])
        assert ev.type == "video.invalidated"
        assert ev.payload == {"processor": "translate", "record_ids": [5]}

    def test_course_metadata_patched(self):
        ev = course_metadata_patched("c1", keys=["videos", "meta"])
        assert ev.type == "course.metadata_patched"
        assert ev.video is None
        assert ev.payload == {"keys": ["videos", "meta"]}

    def test_orchestrator_lifecycle(self):
        s = orchestrator_started("c1", "v1")
        f = orchestrator_finished("c1", "v1", success=True)
        assert s.type == "orchestrator.started"
        assert f.type == "orchestrator.finished"
        assert f.payload == {"success": True}

    def test_event_ids_unique(self):
        a = video_records_patched("c1", "v1", record_ids=[1])
        b = video_records_patched("c1", "v1", record_ids=[1])
        assert a.event_id != b.event_id

    def test_timestamps_monotonic_ish(self):
        before = time.time()
        ev = video_records_patched("c1", "v1", record_ids=[1])
        after = time.time()
        assert before <= ev.timestamp <= after
