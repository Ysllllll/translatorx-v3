"""Tests for :class:`application.processors.TranslateProcessor` (variant-aware).

Cache decision is now made via the *variant key* derived from
``ctx.variant`` (model + prompt_id + config). The legacy provenance
scheme (``translation_meta``, ``_prev`` backup, ``edited`` flag,
``terms_ready_at_translate``, ``output_is_stale``) has been retired.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pytest

from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.checker import CheckReport, Checker
from application.processors import TranslateProcessor
from application.processors.prefix import EN_ZH_PREFIX_RULES, TranslateNodeConfig
from application.terminology import StaticTerms
from application.translate import TranslationContext, VariantSpec
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult
from ports.source import VideoKey


class _RecordingEngine:
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

    def check(self, source: str, translation: str, profile=None, **_) -> CheckReport:
        return CheckReport.ok()


def _variant(model: str = "mock-model-v1", *, alias: str = "", prompt: str = "", prompt_id: str = "default", **config) -> VariantSpec:
    return VariantSpec.create(model=model, prompt_id=prompt_id, prompt=prompt, config=config, alias=alias)


def _ctx(*, variant: VariantSpec | None = None, **overrides) -> TranslationContext:
    params = {"source_lang": "en", "target_lang": "zh", "window_size": 4, "terms_provider": StaticTerms({})}
    params.update(overrides)
    if variant is not None:
        params["variant"] = variant
    return TranslationContext(**params)


def _rec(rid: int, text: str, **extra) -> SentenceRecord:
    base = {"id": rid, **extra}
    return SentenceRecord(src_text=text, start=0.0, end=1.0, extra=base)


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="course_a"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="course_a", video="lec1")


async def _drain(agen) -> list[SentenceRecord]:
    out = []
    async for item in agen:
        out.append(item)
    return out


class TestHitPath:
    @pytest.mark.asyncio
    async def test_hit_when_variant_key_present(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        variant = _variant(alias="alpha")

        recs = [replace(_rec(0, "Hello."), translations={"zh": {"alpha": "你好。"}}), replace(_rec(1, "Bye."), translations={"zh": {"alpha": "再见。"}})]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(variant=variant), store=store, video_key=video_key))
        assert engine.calls == 0
        assert [r.get_translation("zh", default_variant_key="alpha") for r in out] == ["你好。", "再见。"]

    @pytest.mark.asyncio
    async def test_miss_when_variant_key_differs_keeps_old(self, store, video_key):
        """Switching variant computes a new translation while preserving
        the prior bucket entry for A/B comparison."""
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        rec = replace(_rec(0, "Hello."), translations={"zh": {"alpha": "旧译"}})

        async def src():
            yield rec

        out = await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="beta")), store=store, video_key=video_key))
        assert engine.calls == 1
        assert out[0].translations["zh"] == {"alpha": "旧译", "beta": "[翻译]Hello."}

    @pytest.mark.asyncio
    async def test_hydrates_from_store_records(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        variant = _variant(alias="alpha")

        await store.patch_video(video_key.video, records={0: {"translations.zh.alpha": "你好。"}, 1: {"translations.zh.alpha": "再见。"}})

        async def src():
            yield _rec(0, "Hello.")
            yield _rec(1, "Bye.")

        out = await _drain(proc.process(src(), ctx=_ctx(variant=variant), store=store, video_key=video_key))
        assert engine.calls == 0
        assert [r.get_translation("zh", default_variant_key="alpha") for r in out] == ["你好。", "再见。"]


class TestMissPath:
    @pytest.mark.asyncio
    async def test_translation_persisted_with_variant_registry(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())
        variant = _variant(alias="alpha", prompt_id="strict", prompt="be strict")

        async def src():
            yield _rec(0, "Hello.")
            yield _rec(1, "Bye.")

        await _drain(proc.process(src(), ctx=_ctx(variant=variant), store=store, video_key=video_key))
        data = await store.load_video(video_key.video)
        by_id = {r["id"]: r for r in data["records"]}
        assert by_id[0]["translations"]["zh"]["alpha"] == "[翻译]Hello."
        assert by_id[1]["translations"]["zh"]["alpha"] == "[翻译]Bye."
        assert "alpha" in data["variants"]
        assert data["variants"]["alpha"]["model"] == "mock-model-v1"
        assert data["variants"]["alpha"]["prompt_id"] == "strict"
        assert data["prompts"]["strict"] == "be strict"


class TestRefinements:
    @pytest.mark.asyncio
    async def test_direct_translate_bypass(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker(), config=TranslateNodeConfig(direct_translate={"hi": "你好"}))

        async def src():
            yield _rec(0, "hi")
            yield _rec(1, "other")

        out = await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))
        assert engine.calls == 1
        assert out[0].translations["zh"]["v"] == "你好"

    @pytest.mark.asyncio
    async def test_skip_long_bypass(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker(), config=TranslateNodeConfig(max_source_len=5))

        async def src():
            yield _rec(0, "hello world")

        out = await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))
        assert engine.calls == 0
        assert out[0].translations["zh"]["v"] == "hello world"

    @pytest.mark.asyncio
    async def test_prefix_strip_and_readd(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker(), config=TranslateNodeConfig(prefix_rules=EN_ZH_PREFIX_RULES))

        async def src():
            yield _rec(0, "ok, let's go")

        out = await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))
        translation = out[0].translations["zh"]["v"]
        assert translation.startswith("好的，")


class TestBufferedFlush:
    @pytest.mark.asyncio
    async def test_flush_by_count(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        async def src():
            for i in range(5):
                yield _rec(i, f"s{i}")

        await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))
        data = await store.load_video(video_key.video)
        assert len(data["records"]) == 5

    @pytest.mark.asyncio
    async def test_final_flush_happens_on_cancel(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        async def src():
            yield _rec(0, "a")
            yield _rec(1, "b")
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))

        data = await store.load_video(video_key.video)
        by_id = {r["id"]: r for r in data["records"]}
        assert by_id[0]["translations"]["zh"]["v"] == "[翻译]a"
        assert by_id[1]["translations"]["zh"]["v"] == "[翻译]b"


class TestNoIdRecords:
    @pytest.mark.asyncio
    async def test_records_without_id_not_persisted(self, store, video_key):
        engine = _RecordingEngine()
        proc = TranslateProcessor(engine, _PassChecker())

        async def src():
            yield SentenceRecord(src_text="Hello.", start=0.0, end=1.0)

        out = await _drain(proc.process(src(), ctx=_ctx(variant=_variant(alias="v")), store=store, video_key=video_key))
        assert out[0].translations["zh"]["v"] == "[翻译]Hello."

        data = await store.load_video(video_key.video)
        assert data["records"] == []


class TestSelectedOverride:
    def test_selected_wins_over_default_key(self):
        rec = SentenceRecord(src_text="Hello.", start=0.0, end=1.0, translations={"zh": {"alpha": "甲", "beta": "乙"}}, selected={"zh": "beta"})
        assert rec.get_translation("zh", default_variant_key="alpha") == "乙"

    def test_default_key_when_no_selected(self):
        rec = SentenceRecord(src_text="x", start=0.0, end=1.0, translations={"zh": {"alpha": "甲", "beta": "乙"}})
        assert rec.get_translation("zh", default_variant_key="alpha") == "甲"

    def test_first_when_default_missing(self):
        rec = SentenceRecord(src_text="x", start=0.0, end=1.0, translations={"zh": {"alpha": "甲"}})
        assert rec.get_translation("zh", default_variant_key="missing") == "甲"
