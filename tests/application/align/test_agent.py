"""Tests for :class:`application.translate.align_agent.AlignAgent`."""

from __future__ import annotations

import json

import pytest

from application.align import AlignAgent, BisectResult
from domain.model.usage import CompletionResult


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


def _json_mapping(pieces: list[str]) -> str:
    return json.dumps({"mapping": [{"src": f"s{i}", "tgt": p} for i, p in enumerate(pieces)]}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# JSON mode — binary split
# ---------------------------------------------------------------------------


class TestJsonBisect:
    @pytest.mark.asyncio
    async def test_empty_translation(self):
        engine = _ScriptedEngine([])
        agent = AlignAgent(engine, "zh")
        r = await agent.bisect(["hello", "world"], "   ", norm_ratio=5, accept_ratio=5)
        assert not r.accepted
        assert r.pieces == ["", ""]
        assert engine.calls == 0

    @pytest.mark.asyncio
    async def test_rejects_non_binary_src(self):
        engine = _ScriptedEngine([])
        agent = AlignAgent(engine, "zh")
        with pytest.raises(ValueError):
            await agent.bisect(["a"], "xyz", norm_ratio=5, accept_ratio=5)

    @pytest.mark.asyncio
    async def test_successful_bisect(self):
        engine = _ScriptedEngine([_json_mapping(["你好", "世界"])])
        agent = AlignAgent(engine, "zh")
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert r.accepted and not r.need_rearrange
        assert r.pieces == ["你好", "世界"]
        assert engine.calls == 1

    @pytest.mark.asyncio
    async def test_retries_on_bad_shape(self):
        engine = _ScriptedEngine([_json_mapping(["你好世界"]), _json_mapping(["你好", "世界"])])
        agent = AlignAgent(engine, "zh", max_retries=2)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert r.accepted
        assert engine.calls == 2

    @pytest.mark.asyncio
    async def test_retries_on_concat_mismatch(self):
        engine = _ScriptedEngine(
            [
                _json_mapping(["完全", "不对"]),  # fails concat
                _json_mapping(["你好", "世界"]),
            ]
        )
        agent = AlignAgent(engine, "zh", max_retries=2)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert r.accepted
        assert engine.calls == 2

    @pytest.mark.asyncio
    async def test_fallback_on_exhaustion(self):
        engine = _ScriptedEngine([_json_mapping(["错", "了"])])
        agent = AlignAgent(engine, "zh", max_retries=1)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert not r.accepted
        assert r.pieces == ["", ""]

    @pytest.mark.asyncio
    async def test_json_fenced_response(self):
        fenced = "```json\n" + _json_mapping(["你好", "世界"]) + "\n```"
        engine = _ScriptedEngine([fenced])
        agent = AlignAgent(engine, "zh")
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert r.accepted
        assert r.pieces == ["你好", "世界"]

    @pytest.mark.asyncio
    async def test_retries_on_json_parse_error(self):
        engine = _ScriptedEngine(["not json at all", _json_mapping(["你好", "世界"])])
        agent = AlignAgent(engine, "zh", max_retries=2)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=5, accept_ratio=5)
        assert r.accepted

    @pytest.mark.asyncio
    async def test_rearrange_hint_when_ratio_between_norm_accept(self):
        # Make the split heavily unbalanced in length to trigger ratio branch.
        # src "a b c d e f g h i j" vs "k", tgt "你好" vs "你好世界世界世界"
        engine = _ScriptedEngine([_json_mapping(["你好", "世界天地乾坤"])])
        agent = AlignAgent(engine, "zh")
        r = await agent.bisect(["a b c d e", "f"], "你好世界天地乾坤", norm_ratio=1.1, accept_ratio=100.0)
        assert r.accepted
        assert r.need_rearrange


# ---------------------------------------------------------------------------
# Text mode — two-line output
# ---------------------------------------------------------------------------


class TestTextBisect:
    @pytest.mark.asyncio
    async def test_successful_two_line_split(self):
        engine = _ScriptedEngine(["你好\n世界"])
        agent = AlignAgent(engine, "zh", use_json=False, max_retries=0)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=3, accept_ratio=3)
        assert r.accepted
        assert r.pieces == ["你好", "世界"]

    @pytest.mark.asyncio
    async def test_text_mode_defaults_to_six_retries(self):
        agent = AlignAgent(_ScriptedEngine([""]), "zh", use_json=False)
        assert agent._max_retries == 6

    @pytest.mark.asyncio
    async def test_text_mode_rejects_one_line(self):
        engine = _ScriptedEngine(["只有一行", "你好\n世界"])
        agent = AlignAgent(engine, "zh", use_json=False, max_retries=2)
        r = await agent.bisect(["hello", "world"], "你好世界", norm_ratio=3, accept_ratio=3)
        assert r.accepted
        assert engine.calls == 2
