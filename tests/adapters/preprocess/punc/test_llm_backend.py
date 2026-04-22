"""Tests for the ``llm`` punc backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from adapters.preprocess.punc.backends.llm import factory as llm_factory
from adapters.preprocess.punc.registry import PuncBackendRegistry


@dataclass
class _Completion:
    text: str


class _FakeEngine:
    def __init__(self, responses: list[str] | None = None, fail_times: int = 0):
        self._responses = responses or []
        self._fail_times = fail_times
        self.calls: list[list[dict]] = []

    async def complete(self, messages):  # noqa: D401
        self.calls.append(list(messages))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("transport failure")
        user_text = messages[-1]["content"]
        if self._responses:
            return _Completion(text=self._responses.pop(0))
        # Default: echo with a period.
        return _Completion(text=user_text + ".")


class TestLlmBackendFactory:
    def test_registered(self):
        assert PuncBackendRegistry.is_registered("llm")

    def test_basic_batch(self):
        engine = _FakeEngine()
        backend = llm_factory(engine=engine)  # type: ignore[arg-type]
        out = backend(["hello world", "goodbye moon"])
        assert out == ["hello world.", "goodbye moon."]
        assert len(engine.calls) == 2

    def test_retries_on_transport_failure(self):
        engine = _FakeEngine(fail_times=1)
        backend = llm_factory(engine=engine, max_retries=2)  # type: ignore[arg-type]
        out = backend(["hello world"])
        assert out == ["hello world."]

    def test_raises_after_max_retries(self):
        engine = _FakeEngine(fail_times=10)
        backend = llm_factory(engine=engine, max_retries=1)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="LLM punc failed"):
            backend(["hello world"])

    def test_retries_on_content_mismatch(self):
        # First response adds a word (should be rejected), second is clean.
        engine = _FakeEngine(responses=["hello world extra.", "hello world."])
        backend = llm_factory(engine=engine, max_retries=2)  # type: ignore[arg-type]
        out = backend(["hello world"])
        assert out == ["hello world."]
        assert len(engine.calls) == 2

    def test_raises_after_content_mismatch_exhausted(self):
        engine = _FakeEngine(responses=["bye bye.", "bye bye.", "bye bye."])
        backend = llm_factory(engine=engine, max_retries=1)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="LLM punc failed"):
            backend(["hello world"])

    def test_rejects_invalid_max_retries(self):
        with pytest.raises(ValueError):
            llm_factory(engine=_FakeEngine(), max_retries=-1)  # type: ignore[arg-type]

    def test_rejects_invalid_max_concurrent(self):
        with pytest.raises(ValueError):
            llm_factory(engine=_FakeEngine(), max_concurrent=0)  # type: ignore[arg-type]

    def test_runs_under_running_loop(self):
        engine = _FakeEngine()
        backend = llm_factory(engine=engine)  # type: ignore[arg-type]

        async def _run():
            # Backend is sync but handles running loop via ThreadPoolExecutor.
            return await asyncio.get_event_loop().run_in_executor(None, lambda: backend(["hi there"]))

        out = asyncio.run(_run())
        assert out == ["hi there."]
