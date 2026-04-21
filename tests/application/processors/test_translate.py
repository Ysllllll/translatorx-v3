"""Tests for :class:`runtime.processors.TranslateProcessor`.

Covers:

* fingerprint stability + sensitivity to config changes
* hit path (fingerprint matches + translation present) — no LLM call,
  window updated, record yielded unchanged
* miss path — LLM call, ``patch_video`` persists translation +
  ``terms_ready_at_translate`` flag
* ``output_is_stale`` transitions with TermsProvider readiness (D-068)
* buffered flush — records written in batches of ``flush_every``
* finally-shield — pending buffer + fingerprint flush happen on cancel
* direct_translate / skip_long / fake_process refinements inherit from
  the old :mod:`pipeline.nodes` behaviour
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pytest

from application.checker import CheckReport
from application.translate import Checker, ContextWindow, StaticTerms, TranslationContext
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult

from application.processors import TranslateProcessor
from ports.source import VideoKey
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace

from application.processors.prefix import EN_ZH_PREFIX_RULES, TranslateNodeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Simple mock — returns [翻译]<user> and counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.model = "mock-model-v1"

    async def complete(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"]
        return CompletionResult(text=f"[翻译]{user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source: str, translation: str, profile=None) -> CheckReport:
        return CheckReport.ok()


def _ctx(**overrides) -> TranslationContext:
    params = {
        "source_lang": "en",
        "target_lang": "zh",
        "window_size": 4,
        "terms_provider": StaticTerms({}),
    }
    params.update(overrides)
    return TranslationContext(**params)


class _NotReadyTerms(StaticTerms):
    """StaticTerms variant that reports ``ready=False`` for testing."""

    @property
    def ready(self) -> bool:  # type: ignore[override]
        return False


def _rec(rid: int, text: str, **extra) -> SentenceRecord:
    base = {"id": rid, **extra}
    return SentenceRecord(src_text=text, start=0.0, end=1.0, extra=base)


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    ws = Workspace(root=tmp_path, course="course_a")
    return JsonFileStore(ws)


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="course_a", video="lec1")


async def _drain(agen) -> list[SentenceRecord]:
    out = []
    async for item in agen:
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_stable(self):
        e = _RecordingEngine()
        p1 = TranslateProcessor(e, _PassChecker())
        p2 = TranslateProcessor(e, _PassChecker())
        assert p1.fingerprint() == p2.fingerprint()

    def test_sensitive_to_prompt(self):
        e = _RecordingEngine()
        p1 = TranslateProcessor(e, _PassChecker())
        p2 = TranslateProcessor(e, _PassChecker(), config=TranslateNodeConfig(system_prompt="diff"))
        assert p1.fingerprint() != p2.fingerprint()

    def test_sensitive_to_direct_map(self):
        e = _RecordingEngine()
        p1 = TranslateProcessor(e, _PassChecker())
        p2 = TranslateProcessor(
            e,
            _PassChecker(),
            config=TranslateNodeConfig(direct_translate={"hi": "你好"}),
        )
        assert p1.fingerprint() != p2.fingerprint()

    def test_sensitive_to_model(self):
        e1 = _RecordingEngine()
        e2 = _RecordingEngine()
        e2.model = "mock-model-v2"
        p1 = TranslateProcessor(e1, _PassChecker())
        p2 = TranslateProcessor(e2, _PassChecker())
        assert p1.fingerprint() != p2.fingerprint()


# ---------------------------------------------------------------------------
# hit path
# ---------------------------------------------------------------------------


class TestHitPath:
    @pytest.mark.asyncio
    async def test_hit_skips_llm(self, store, video_key):
        """Matching fingerprint + existing translation → no LLM call."""
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        # Seed the store with the current fingerprint.
        await store.patch_video(
            video_key.video,
            meta={"_fingerprints": {"translate": proc.fingerprint()}},
        )

        recs = [
            replace(_rec(0, "Hello."), translations={"zh": "你好。"}),
            replace(_rec(1, "Bye."), translations={"zh": "再见。"}),
        ]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 0
        assert [r.translations["zh"] for r in out] == ["你好。", "再见。"]

    @pytest.mark.asyncio
    async def test_hit_hydrates_from_store_records(self, store, video_key):
        """Round-2 scenario: upstream Source emits records with empty
        ``translations`` (like :class:`SrtSource`). The processor must
        hydrate them from the persisted ``records[]`` so the cache-hit
        branch fires — otherwise every rerun re-pays the LLM cost.
        """
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        # Seed the store as if a previous run had completed: records
        # persisted + fingerprint stamped.
        await store.patch_video(
            video_key.video,
            records={
                0: {"translations.zh": "你好。"},
                1: {"translations.zh": "再见。"},
            },
            meta={"_fingerprints": {"translate": proc.fingerprint()}},
        )

        # Upstream now emits *fresh* records (empty translations) — what
        # SrtSource does in practice.
        recs = [_rec(0, "Hello."), _rec(1, "Bye.")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 0
        assert [r.translations["zh"] for r in out] == ["你好。", "再见。"]

    @pytest.mark.asyncio
    async def test_miss_when_fingerprint_mismatch(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        # Stale fingerprint — should force miss even though translation is present.
        await store.patch_video(video_key.video, meta={"_fingerprints": {"translate": "stale"}})

        recs = [replace(_rec(0, "Hello."), translations={"zh": "旧译"})]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 1
        assert out[0].translations["zh"] == "[翻译]Hello."


# ---------------------------------------------------------------------------
# miss path + persistence
# ---------------------------------------------------------------------------


class TestMissPath:
    @pytest.mark.asyncio
    async def test_translation_persisted(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker(), flush_every=1, flush_interval_s=0.01)

        recs = [_rec(0, "Hello."), _rec(1, "Bye.")]

        async def src():
            for r in recs:
                yield r

        await _drain(proc.process(src(), ctx=_ctx(terms_provider=_NotReadyTerms({})), store=store, video_key=video_key))
        data = await store.load_video(video_key.video)
        by_id = {r["id"]: r for r in data["records"]}
        assert by_id[0]["translations"]["zh"] == "[翻译]Hello."
        assert by_id[1]["translations"]["zh"] == "[翻译]Bye."
        assert by_id[0]["extra"]["terms_ready_at_translate"] is False
        assert data["meta"]["_fingerprints"]["translate"] == proc.fingerprint()

    @pytest.mark.asyncio
    async def test_terms_ready_flag_reflects_provider(self, store, video_key):
        class _ReadyTerms(StaticTerms):
            @property
            def ready(self) -> bool:  # type: ignore[override]
                return True

        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        ctx = _ctx(terms_provider=_ReadyTerms({}))

        recs = [_rec(0, "Hello.")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=ctx, store=store, video_key=video_key))
        # Clean case: the marker is absent (terms ready → nothing to record).
        assert "terms_ready_at_translate" not in out[0].extra

    @pytest.mark.asyncio
    async def test_terms_not_ready_flag_marks_record(self, store, video_key):
        class _NotReadyTerms(StaticTerms):
            @property
            def ready(self) -> bool:  # type: ignore[override]
                return False

        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        ctx = _ctx(terms_provider=_NotReadyTerms({}))

        recs = [_rec(0, "Hello.")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=ctx, store=store, video_key=video_key))
        # Only the explicit False marker is recorded — enables future
        # retranslate when terms arrive.
        assert out[0].extra["terms_ready_at_translate"] is False


# ---------------------------------------------------------------------------
# refinements (direct_translate / skip_long / fake_process / prefix)
# ---------------------------------------------------------------------------


class TestRefinements:
    @pytest.mark.asyncio
    async def test_direct_translate_bypass(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(
            engine,
            _PassChecker(),
            config=TranslateNodeConfig(direct_translate={"hi": "你好"}),
        )

        async def src():
            yield _rec(0, "hi")
            yield _rec(1, "other")

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 1  # only "other" hit the engine
        assert out[0].translations["zh"] == "你好"

    @pytest.mark.asyncio
    async def test_skip_long_bypass(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(
            engine,
            _PassChecker(),
            config=TranslateNodeConfig(max_source_len=5),
        )

        async def src():
            yield _rec(0, "hello world")  # len 11 > 5

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 0
        assert out[0].translations["zh"] == "hello world"

    @pytest.mark.asyncio
    async def test_fingerprint_hit_skips_llm(self, store, video_key):
        """Record already translated + matching fingerprint → no LLM call."""
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        # Seed store with matching fingerprint so the hit path fires.
        await store.patch_video(
            video_key.video,
            meta={"_fingerprints": {"translate": proc.fingerprint()}},
        )

        async def src():
            yield replace(_rec(0, "Hello."), translations={"zh": "preloaded"})

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine.calls == 0
        assert out[0].translations["zh"] == "preloaded"

    @pytest.mark.asyncio
    async def test_prefix_strip_and_readd(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(
            engine,
            _PassChecker(),
            config=TranslateNodeConfig(prefix_rules=EN_ZH_PREFIX_RULES),
        )

        async def src():
            yield _rec(0, "ok, let's go")

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        # prefix readded → translation must start with the canonical "好的，"
        actual_prefix = out[0].translations["zh"][: len("好的，")]
        assert actual_prefix == "好的，"


# ---------------------------------------------------------------------------
# output_is_stale (D-068)
# ---------------------------------------------------------------------------


class TestStaleDetection:
    @pytest.mark.asyncio
    async def test_default_not_stale_without_provider(self, store, video_key):
        proc = TranslateProcessor(_RecordingEngine(), _PassChecker())
        rec = _rec(0, "hi")
        assert proc.output_is_stale(rec) is False

    @pytest.mark.asyncio
    async def test_stale_when_terms_became_ready(self, store, video_key):
        """Provider flips False→True mid-run: records translated before the
        flip are stale (the flag is False)."""

        class _Toggle(StaticTerms):
            def __init__(self):
                super().__init__({})
                self._ready = False

            @property
            def ready(self) -> bool:  # type: ignore[override]
                return self._ready

        provider = _Toggle()
        proc = TranslateProcessor(_RecordingEngine(), _PassChecker())
        ctx = _ctx(terms_provider=provider)

        recs = [_rec(0, "Hello."), _rec(1, "Bye.")]

        async def src():
            yield recs[0]
            # flip provider to ready between records
            provider._ready = True
            yield recs[1]

        out = await _drain(proc.process(src(), ctx=ctx, store=store, video_key=video_key))
        # First record was translated while provider not ready → stale
        # Second record was translated after flip → not stale
        assert proc.output_is_stale(out[0]) is True
        assert proc.output_is_stale(out[1]) is False


# ---------------------------------------------------------------------------
# buffered flush
# ---------------------------------------------------------------------------


class TestBufferedFlush:
    @pytest.mark.asyncio
    async def test_flush_by_count(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(
            engine,
            _PassChecker(),
            flush_every=2,
            flush_interval_s=3600,
        )

        async def src():
            for i in range(5):
                yield _rec(i, f"s{i}")

        await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        data = await store.load_video(video_key.video)
        assert len(data["records"]) == 5  # all landed (final flush + 2 batches)

    @pytest.mark.asyncio
    async def test_final_flush_happens_on_cancel(self, store, video_key):
        """Cancelling mid-stream still flushes the pending buffer."""
        engine = _RecordingEngine()
        proc = TranslateProcessor(
            engine,
            _PassChecker(),
            flush_every=100,
            flush_interval_s=3600,
        )

        async def src():
            yield _rec(0, "a")
            yield _rec(1, "b")
            raise asyncio.CancelledError  # cancel mid-stream

        with pytest.raises(asyncio.CancelledError):
            await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))

        data = await store.load_video(video_key.video)
        # Both records were computed before cancel → both must be persisted
        # by the finally-shielded flush (D-045).
        by_id = {r["id"]: r for r in data["records"]}
        assert by_id[0]["translations"]["zh"] == "[翻译]a"
        assert by_id[1]["translations"]["zh"] == "[翻译]b"
        assert data["meta"]["_fingerprints"]["translate"] == proc.fingerprint()


# ---------------------------------------------------------------------------
# records without id — pure-function mode (no persistence)
# ---------------------------------------------------------------------------


class TestNoIdRecords:
    @pytest.mark.asyncio
    async def test_records_without_id_not_persisted(self, store, video_key):
        """Orchestrator may feed ad-hoc records with no id; processor still
        translates them but does not write per-record patches."""
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        async def src():
            # No id in extra
            yield SentenceRecord(src_text="Hello.", start=0.0, end=1.0)

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out[0].translations["zh"] == "[翻译]Hello."

        data = await store.load_video(video_key.video)
        assert data["records"] == []  # nothing written, but meta fingerprint OK
        assert data["meta"]["_fingerprints"]["translate"] == proc.fingerprint()
