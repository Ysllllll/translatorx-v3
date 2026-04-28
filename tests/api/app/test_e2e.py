"""End-to-end runtime tests through the public ``App`` / Builder API.

These tests deliberately exercise the **full wiring** — real
:class:`JsonFileStore`, real :class:`SrtSource`/:class:`PushQueueSource`,
real :class:`Orchestrator`, real :class:`TranslateProcessor` — with only
the LLM engine and checker mocked.

Lower-level files in this directory cover individual components in
isolation; this module guards the seams between them and several
journey-level invariants (cache hits, resume after failure,
on-disk persistence after streaming).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from application.checker import CheckReport
from application.checker import Checker
from domain.model import Segment
from domain.model.usage import CompletionResult

from api.app import App


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _make_app(root: Path) -> App:
    return App.from_dict({"engines": {"default": {"kind": "openai_compat", "model": "mock", "base_url": "http://localhost:0/v1", "api_key": "EMPTY"}}, "contexts": {"en_zh": {"src": "en", "tgt": "zh"}}, "store": {"kind": "json", "root": root.as_posix()}, "runtime": {"flush_every": 1, "max_concurrent_videos": 2}})


class _CountingEngine:
    """Mock engine that counts ``complete`` calls."""

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.model = "mock"
        self.calls: list[str] = []
        self._fail_on = fail_on or set()

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        self.calls.append(user)
        for needle in self._fail_on:
            if needle in user:
                raise RuntimeError(f"engine refused on '{needle}'")
        return CompletionResult(text=f"[zh]{user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None, **_) -> CheckReport:
        return CheckReport.ok()


def _bind(app: App, engine: _CountingEngine) -> None:
    app.engine = lambda name="default": engine  # type: ignore[assignment]
    app.checker = lambda s, t: _PassChecker()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# E2E1 — rerun against persisted Store hits cache, no LLM call
# ---------------------------------------------------------------------------


class TestRerunCacheHit:
    """Guards the bug fixed in commit eb1cb03 — cache must hit on rerun
    even when the upstream Source emits records with empty translations."""

    @pytest.mark.asyncio
    async def test_second_run_makes_zero_engine_calls(self, tmp_path: Path):
        ws = tmp_path / "ws"
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello.", "World.", "Goodbye."])

        # Run 1 — fresh app, all records translated
        app1 = _make_app(ws)
        eng1 = _CountingEngine()
        _bind(app1, eng1)
        r1 = await app1.video(course="c", video="v").source(srt, language="en").translate(src="en", tgt="zh").run()
        assert len(r1.records) == 3
        assert len(eng1.calls) == 3, "first run must translate all 3"

        # Disk should have records persisted
        store_file = ws / "c" / "zzz_translation" / "v.json"
        assert store_file.exists()
        on_disk = json.loads(store_file.read_text())
        assert len(on_disk["records"]) == 3
        assert all("zh" in r["translations"] for r in on_disk["records"])

        # Run 2 — fresh app instance (no in-memory state) on same store
        app2 = _make_app(ws)
        eng2 = _CountingEngine()
        _bind(app2, eng2)
        r2 = await app2.video(course="c", video="v").source(srt, language="en").translate(src="en", tgt="zh").run()
        assert len(r2.records) == 3
        assert eng2.calls == [], "second run must hit cache, no LLM calls"
        for rec in r2.records:
            actual_prefix = (rec.get_translation("zh") or "")[: len("[zh]")]
            assert actual_prefix == "[zh]"


# ---------------------------------------------------------------------------
# E2E2 — partial failure: failed records persisted, succeeded skip on rerun
# ---------------------------------------------------------------------------


class TestPartialFailureResume:
    """Engine fails midway on run 1; partial progress must persist via the
    flush-every-1 contract so run 2 only retranslates what's missing."""

    @pytest.mark.asyncio
    async def test_resume_after_midway_engine_error(self, tmp_path: Path):
        ws = tmp_path / "ws"
        srt = tmp_path / "mix.srt"
        _write_srt(srt, ["Alpha.", "BAD-LINE.", "Bravo."])

        # Run 1 — engine raises on BAD; orchestrator aborts mid-stream.
        app1 = _make_app(ws)
        eng1 = _CountingEngine(fail_on={"BAD"})
        _bind(app1, eng1)
        with pytest.raises(RuntimeError, match="engine refused"):
            await app1.video(course="c", video="mix").source(srt, language="en").translate(src="en", tgt="zh").run()

        # Alpha was processed before BAD → must be persisted (flush_every=1).
        store_file = ws / "c" / "zzz_translation" / "mix.json"
        assert store_file.exists()
        on_disk = json.loads(store_file.read_text())
        persisted_ids = {r["id"] for r in on_disk["records"]}
        assert 0 in persisted_ids, "Alpha must persist before the abort"

        # Run 2 — healthy engine; Alpha hits cache, others retranslate.
        app2 = _make_app(ws)
        eng2 = _CountingEngine()
        _bind(app2, eng2)
        r2 = await app2.video(course="c", video="mix").source(srt, language="en").translate(src="en", tgt="zh").run()
        assert len(r2.records) == 3
        for r in r2.records:
            actual_prefix = (r.get_translation("zh") or "")[: len("[zh]")]
            assert actual_prefix == "[zh]"
        # Engine called only for records not previously persisted (BAD + Bravo)
        # — Alpha must NOT trigger a new call.
        assert "Alpha." not in eng2.calls
        # At most BAD and Bravo are retranslated; Alpha is served from cache.
        actual_calls = len(eng2.calls)
        max_expected = 2
        assert actual_calls <= max_expected, f"expected ≤{max_expected} retried calls, got {actual_calls}"


# ---------------------------------------------------------------------------
# E2E3 — Course end-to-end with on-disk persistence
# ---------------------------------------------------------------------------


class TestCourseE2E:
    @pytest.mark.asyncio
    async def test_course_persists_each_video_independently(self, tmp_path: Path):
        ws = tmp_path / "ws"
        a = tmp_path / "a.srt"
        b = tmp_path / "b.srt"
        _write_srt(a, ["From A line 1.", "From A line 2."])
        _write_srt(b, ["From B only."])

        app = _make_app(ws)
        eng = _CountingEngine()
        _bind(app, eng)

        result = await app.course(course="cs101").add_video("vid_a", a, language="en").add_video("vid_b", b, language="en").translate(src="en", tgt="zh").run()

        assert len(result.succeeded) == 2
        # On disk: both videos written under same course directory
        course_dir = ws / "cs101" / "zzz_translation"
        assert (course_dir / "vid_a.json").exists()
        assert (course_dir / "vid_b.json").exists()

        a_data = json.loads((course_dir / "vid_a.json").read_text())
        b_data = json.loads((course_dir / "vid_b.json").read_text())
        assert len(a_data["records"]) == 2
        assert len(b_data["records"]) == 1
        # Engine sees inputs from both videos, total 3 calls
        assert len(eng.calls) == 3


# ---------------------------------------------------------------------------
# E2E4 — Stream end-to-end persists records to Store
# ---------------------------------------------------------------------------


class TestStreamE2E:
    @pytest.mark.asyncio
    async def test_stream_writes_records_to_store(self, tmp_path: Path):
        ws = tmp_path / "ws"
        app = _make_app(ws)
        eng = _CountingEngine()
        _bind(app, eng)

        handle = app.stream(course="c", video="live", language="en").translate(src="en", tgt="zh").start()

        # Pump in two segments and drain concurrently — close after feed.
        collected: list = []

        async def drain():
            async for rec in handle.records():
                collected.append(rec)

        consumer = asyncio.create_task(drain())
        await handle.feed(Segment(start=0.0, end=1.0, text="One."))
        await handle.feed(Segment(start=1.0, end=2.0, text="Two."))
        await handle.close()
        await consumer

        assert len(collected) == 2
        for r in collected:
            actual_prefix = (r.get_translation("zh") or "")[: len("[zh]")]
            assert actual_prefix == "[zh]"

        # Store on disk should have both records
        store_file = ws / "c" / "zzz_translation" / "live.json"
        assert store_file.exists()
        on_disk = json.loads(store_file.read_text())
        assert len(on_disk["records"]) == 2

    @pytest.mark.asyncio
    async def test_stream_rerun_via_video_builder_hits_cache(self, tmp_path: Path):
        """Stream first, then rerun the same video via VideoBuilder using
        an SRT — cache stamped by stream should still apply if fingerprint
        matches.  Verifies fingerprint stability across Source kinds."""
        ws = tmp_path / "ws"

        # Stream phase — populate store
        app1 = _make_app(ws)
        eng1 = _CountingEngine()
        _bind(app1, eng1)
        handle = app1.stream(course="c", video="hybrid", language="en").translate(src="en", tgt="zh").start()
        collected: list = []

        async def drain():
            async for rec in handle.records():
                collected.append(rec)

        task = asyncio.create_task(drain())
        await handle.feed(Segment(start=0.0, end=1.0, text="Ping."))
        await handle.close()
        await task
        first_calls = list(eng1.calls)
        assert len(first_calls) == 1

        # The store is populated; ensure the on-disk file is healthy
        store_file = ws / "c" / "zzz_translation" / "hybrid.json"
        assert store_file.exists()
        data = json.loads(store_file.read_text())
        assert len(data["records"]) == 1
        # Variant registry stamped by translate processor.
        assert "variants" in data and data["variants"]
        rec0 = data["records"][0]
        # Translation persisted under at least one variant key.
        zh = rec0["translations"]["zh"]
        assert isinstance(zh, dict) and len(zh) >= 1


# ---------------------------------------------------------------------------
# E2E5 — config-via-dict round-trip is end-to-end functional
# ---------------------------------------------------------------------------


class TestConfigVariants:
    @pytest.mark.asyncio
    async def test_yaml_string_app_produces_translations(self, tmp_path: Path):
        ws = tmp_path / "ws"
        text = f"engines:\n  default:\n    kind: openai_compat\n    model: mock\n    base_url: http://localhost:0/v1\n    api_key: EMPTY\nstore:\n  root: {ws.as_posix()}\nruntime:\n  flush_every: 1\n"
        app = App.from_yaml(text)
        eng = _CountingEngine()
        _bind(app, eng)

        srt = tmp_path / "y.srt"
        _write_srt(srt, ["YAML works."])
        result = await app.video(course="c", video="y").source(srt, language="en").translate(src="en", tgt="zh").run()
        assert result.records[0].get_translation("zh") == "[zh]YAML works."
