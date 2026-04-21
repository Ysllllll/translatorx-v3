"""Tests for :class:`application.translate.align_agent.AlignAgent`."""

from __future__ import annotations

import json
from typing import Any

import pytest

from application.translate.align_agent import AlignAgent
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


def _mapping(pieces: list[str]) -> str:
    return json.dumps({"mapping": [{"source": f"s{i}", "target": p} for i, p in enumerate(pieces)]}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_single_segment_short_circuits():
    engine = _ScriptedEngine([])
    agent = AlignAgent(engine, "zh")
    r = await agent.align(["hello"], "你好")
    assert r.accepted
    assert r.pieces == ["你好"]
    assert engine.calls == 0


@pytest.mark.asyncio
async def test_empty_translation_returns_blanks():
    engine = _ScriptedEngine([])
    agent = AlignAgent(engine, "zh")
    r = await agent.align(["a", "b", "c"], "   ")
    assert r.pieces == ["", "", ""]
    assert engine.calls == 0


@pytest.mark.asyncio
async def test_successful_alignment():
    engine = _ScriptedEngine([_mapping(["你好", "世界"])])
    agent = AlignAgent(engine, "zh")
    r = await agent.align(["hello", "world"], "你好世界")
    assert r.accepted
    assert r.pieces == ["你好", "世界"]
    assert engine.calls == 1


@pytest.mark.asyncio
async def test_retries_on_length_mismatch():
    engine = _ScriptedEngine(
        [
            _mapping(["你好世界"]),  # wrong length
            _mapping(["你好", "世界"]),  # good
        ]
    )
    agent = AlignAgent(engine, "zh", max_retries=2)
    r = await agent.align(["hello", "world"], "你好世界")
    assert r.accepted
    assert engine.calls == 2


@pytest.mark.asyncio
async def test_retries_on_concat_mismatch():
    # Different content AND different length → fails ratio check too.
    engine = _ScriptedEngine(
        [
            _mapping(["一", "二"]),  # concat length 2, expected length 4
            _mapping(["你好", "世界"]),
        ]
    )
    agent = AlignAgent(engine, "zh", max_retries=2, tolerate_ratio=0.1)
    r = await agent.align(["hello", "world"], "你好世界")
    assert r.accepted
    assert engine.calls == 2


@pytest.mark.asyncio
async def test_fallback_on_exhaustion():
    engine = _ScriptedEngine([_mapping(["wrong"])])
    agent = AlignAgent(engine, "zh", max_retries=1)
    r = await agent.align(["hello", "world"], "你好世界")
    assert not r.accepted
    assert r.pieces[0] == "你好世界"
    assert r.pieces[1] == ""


@pytest.mark.asyncio
async def test_tolerate_ratio_accepts_close_enough():
    # Concat "你好世界 " has whitespace — normalized away. If the mapping
    # produces "你好世 界", whitespace normalization makes them equal.
    engine = _ScriptedEngine([_mapping(["你好世", "界"])])
    agent = AlignAgent(engine, "zh", max_retries=0)
    r = await agent.align(["hello", "world"], "你好世界")
    assert r.accepted
    assert r.pieces == ["你好世", "界"]


@pytest.mark.asyncio
async def test_retries_on_json_parse_error():
    engine = _ScriptedEngine(["not json at all", _mapping(["你好", "世界"])])
    agent = AlignAgent(engine, "zh", max_retries=2)
    r = await agent.align(["hello", "world"], "你好世界")
    assert r.accepted
