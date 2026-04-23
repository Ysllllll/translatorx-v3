"""Tests for :class:`application.processors.AlignProcessor`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.processors import AlignProcessor
from application.terminology import StaticTerms
from application.translate import TranslationContext
from domain.model import Segment, SentenceRecord
from domain.model.usage import CompletionResult
from ports.source import VideoKey


class _ScriptedEngine:
    def __init__(self, scripts: list[str]) -> None:
        self.scripts = list(scripts)
        self.calls = 0
        self.model = "mock"

    async def complete(self, messages, **_) -> CompletionResult:
        i = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return CompletionResult(text=self.scripts[i])

    async def stream(self, messages, **_):  # pragma: no cover
        yield (await self.complete(messages)).text


def _map(pieces: list[str]) -> str:
    return json.dumps({"mapping": [{"src": f"s{i}", "tgt": p} for i, p in enumerate(pieces)]}, ensure_ascii=False)


def _rec(rid: int, segs: list[str], translation: str) -> SentenceRecord:
    segments = [Segment(start=float(i), end=float(i + 1), text=t) for i, t in enumerate(segs)]
    return SentenceRecord(src_text=" ".join(segs), start=0.0, end=float(len(segs)), segments=segments, translations={"zh": translation} if translation else {}, extra={"id": rid})


def _ctx() -> TranslationContext:
    return TranslationContext(source_lang="en", target_lang="zh", window_size=4, terms_provider=StaticTerms({}))


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="c"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c", video="v1")


async def _drain(agen):
    return [x async for x in agen]


class TestFingerprint:
    def test_stable_across_instances(self):
        e = _ScriptedEngine([])
        a = AlignProcessor(e)
        b = AlignProcessor(e)
        assert a.fingerprint() == b.fingerprint()

    def test_sensitive_to_text_mode_toggle(self):
        e = _ScriptedEngine([])
        a = AlignProcessor(e, enable_text_mode=False)
        b = AlignProcessor(e, enable_text_mode=True)
        assert a.fingerprint() != b.fingerprint()

    def test_sensitive_to_ratio_thresholds(self):
        e = _ScriptedEngine([])
        a = AlignProcessor(e, json_accept_ratio=5.0)
        b = AlignProcessor(e, json_accept_ratio=3.0)
        assert a.fingerprint() != b.fingerprint()


class TestSkipPaths:
    @pytest.mark.asyncio
    async def test_skips_when_no_translation(self, store, video_key):
        e = _ScriptedEngine([])
        proc = AlignProcessor(e)
        recs = [_rec(0, ["a", "b"], "")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out[0].alignment == {}
        assert e.calls == 0

    @pytest.mark.asyncio
    async def test_single_segment_trivial(self, store, video_key):
        e = _ScriptedEngine([])
        proc = AlignProcessor(e)
        recs = [_rec(0, ["hello"], "你好")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out[0].alignment == {"zh": ["你好"]}
        assert e.calls == 0


class TestRecursiveBisect:
    @pytest.mark.asyncio
    async def test_two_segments_single_call(self, store, video_key):
        e = _ScriptedEngine([_map(["你好", "世界"])])
        proc = AlignProcessor(e)
        recs = [_rec(0, ["hello", "world"], "你好世界")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out[0].alignment == {"zh": ["你好", "世界"]}
        assert e.calls == 1

    @pytest.mark.asyncio
    async def test_four_segments_recurses(self, store, video_key):
        # First split: "你好世界天地乾坤" -> ("你好世界", "天地乾坤") over segs[:2]|segs[2:]
        # Then split "你好世界" -> ("你好", "世界") and "天地乾坤" -> ("天地", "乾坤")
        scripts = [_map(["你好世界", "天地乾坤"]), _map(["你好", "世界"]), _map(["天地", "乾坤"])]
        e = _ScriptedEngine(scripts)
        proc = AlignProcessor(e)
        recs = [_rec(0, ["one", "two", "three", "four"], "你好世界天地乾坤")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out[0].alignment["zh"] == ["你好", "世界", "天地", "乾坤"]

    @pytest.mark.asyncio
    async def test_persists_fingerprint(self, store, video_key):
        e = _ScriptedEngine([_map(["你好", "世界"])])
        proc = AlignProcessor(e)
        recs = [_rec(0, ["hello", "world"], "你好世界")]

        async def src():
            for r in recs:
                yield r

        await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        data = await store.load_video("v1")
        assert data["meta"]["_fingerprints"]["align"] == proc.fingerprint()


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_skip_on_fingerprint_match(self, store, video_key):
        e = _ScriptedEngine([_map(["你好", "世界"])])
        proc = AlignProcessor(e)

        async def src1():
            yield _rec(0, ["hello", "world"], "你好世界")

        await _drain(proc.process(src1(), ctx=_ctx(), store=store, video_key=video_key))
        calls_after_first = e.calls

        proc2 = AlignProcessor(e)

        async def src2():
            yield _rec(0, ["hello", "world"], "你好世界")

        out = await _drain(proc2.process(src2(), ctx=_ctx(), store=store, video_key=video_key))
        assert e.calls == calls_after_first
        assert out[0].alignment["zh"] == ["你好", "世界"]


class TestConfig:
    def test_requires_engine_or_agent(self):
        with pytest.raises(ValueError):
            AlignProcessor(None)
