"""VideoSession unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from application.orchestrator.session import VideoSession
from domain.model import SentenceRecord
from ports.source import VideoKey


class FakeStore:
    """Minimal in-memory Store stub satisfying the methods VideoSession uses."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        if initial is not None:
            self._data["v1"] = initial
        self.patch_calls: list[dict[str, Any]] = []
        self.fp_calls: list[dict[str, str]] = []

    async def load_video(self, video: str) -> dict[str, Any]:
        return self._data.get(video, {})

    async def patch_video(self, video: str, **kwargs: Any) -> None:
        self.patch_calls.append({"video": video, **kwargs})

    async def set_fingerprints(self, video: str, fingerprints: dict[str, str]) -> None:
        self.fp_calls.append({"video": video, **fingerprints})


def _key() -> VideoKey:
    return VideoKey(course="c1", video="v1")


def _rec(rec_id: int, src: str = "hello") -> SentenceRecord:
    return SentenceRecord(src_text=src, start=0.0, end=1.0, extra={"id": rec_id})


# ---------------------------------------------------------------------------
# load + hydrate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_empty_store() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())
    assert sess.is_dirty is False
    assert sess.pending_record_count == 0
    assert sess.stored_summary is None
    assert sess.stored_fingerprints == {}


@pytest.mark.asyncio
async def test_hydrate_merges_translations_and_alignment() -> None:
    initial = {"records": [{"id": 7, "translations": {"zh": {"default": "你好"}}, "alignment": {"zh": ["你好"]}, "selected": {"zh": "default"}}]}
    store = FakeStore(initial)
    sess = await VideoSession.load(store, _key())

    rec = _rec(7)
    out = sess.hydrate(rec)
    assert out.translations == {"zh": {"default": "你好"}}
    assert out.alignment == {"zh": ["你好"]}
    assert out.selected == {"zh": "default"}


@pytest.mark.asyncio
async def test_hydrate_promotes_legacy_string_translation() -> None:
    initial = {"records": [{"id": 3, "translations": {"zh": "你好"}}]}
    store = FakeStore(initial)
    sess = await VideoSession.load(store, _key())
    out = sess.hydrate(_rec(3))
    assert out.translations == {"zh": {"legacy": "你好"}}


@pytest.mark.asyncio
async def test_hydrate_in_memory_wins_over_stored() -> None:
    initial = {"records": [{"id": 1, "translations": {"zh": {"a": "stored"}}}]}
    store = FakeStore(initial)
    sess = await VideoSession.load(store, _key())

    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"id": 1}, translations={"zh": {"a": "memory"}})
    out = sess.hydrate(rec)
    # In-memory cell overrides stored cell with the same key.
    assert out.translations["zh"]["a"] == "memory"


@pytest.mark.asyncio
async def test_hydrate_no_id_returns_unchanged() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())
    rec = SentenceRecord(src_text="x", start=0.0, end=1.0)
    assert sess.hydrate(rec) is rec


# ---------------------------------------------------------------------------
# Dirty tracking + flush
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_translation_marks_dirty_and_flushes() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())

    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"id": 5}, translations={"zh": {"default": "你好"}})
    sess.set_translation(rec, "zh", "default", "你好")
    assert sess.is_dirty is True
    assert sess.pending_record_count == 1

    await sess.flush(store)
    assert sess.is_dirty is False
    assert sess.pending_record_count == 0
    assert len(store.patch_calls) == 1
    call = store.patch_calls[0]
    assert call["video"] == "v1"
    assert 5 in call["records"]
    patch = call["records"][5]
    assert patch[("translations", "zh", "default")] == "你好"


@pytest.mark.asyncio
async def test_flush_noop_when_clean() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())
    await sess.flush(store)
    assert store.patch_calls == []
    assert store.fp_calls == []


@pytest.mark.asyncio
async def test_set_alignment_and_segments_payload() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())

    sess.set_alignment(2, "zh", ["你", "好"])
    sess.set_segments_payload(2, [{"text": "hello"}])
    await sess.flush(store)

    patch = store.patch_calls[0]["records"][2]
    assert patch[("alignment", "zh")] == ["你", "好"]
    assert patch["segments"] == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_variant_key_with_dot_uses_tuple_path() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())

    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"id": 9}, translations={"zh": {"openai/gpt-3.5": "你好"}})
    sess.set_translation(rec, "zh", "openai/gpt-3.5", "你好")
    await sess.flush(store)

    patch = store.patch_calls[0]["records"][9]
    # tuple key — not dotted string
    assert ("translations", "zh", "openai/gpt-3.5") in patch


@pytest.mark.asyncio
async def test_set_summary_and_fingerprint() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())

    sess.set_summary({"title": "T"})
    sess.set_fingerprint("align", "fp123")
    await sess.flush(store)

    assert store.patch_calls[0]["summary"] == {"title": "T"}
    assert store.fp_calls == [{"video": "v1", "align": "fp123"}]


@pytest.mark.asyncio
async def test_repeat_set_translation_merges_patch() -> None:
    store = FakeStore()
    sess = await VideoSession.load(store, _key())

    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"id": 1}, translations={"zh": {"a": "x", "b": "y"}})
    sess.set_translation(rec, "zh", "a", "x")
    sess.set_translation(rec, "zh", "b", "y")
    assert sess.pending_record_count == 1  # same record, merged

    await sess.flush(store)
    patch = store.patch_calls[0]["records"][1]
    assert patch[("translations", "zh", "a")] == "x"
    assert patch[("translations", "zh", "b")] == "y"
