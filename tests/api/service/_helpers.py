"""Shared helpers for service-layer tests."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from application.checker import CheckReport
from application.translate import Checker
from domain.model.usage import CompletionResult

from api.app import App


def write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n")
    path.write_text("\n".join(body), encoding="utf-8")


def make_app(root: Path) -> App:
    return App.from_dict({"engines": {"default": {"kind": "openai_compat", "model": "mock", "base_url": "http://localhost:0/v1", "api_key": "EMPTY"}}, "contexts": {"en_zh": {"src": "en", "tgt": "zh"}}, "store": {"kind": "json", "root": root.as_posix()}, "runtime": {"flush_every": 1, "max_concurrent_videos": 2}})


class ScriptedEngine:
    """Mock engine that returns ``[zh]<source>`` for every call."""

    def __init__(self) -> None:
        self.model = "mock"
        self.calls: list[str] = []

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        self.calls.append(user)
        return CompletionResult(text=f"[zh]{user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


def bind_mocks(app: App, engine: ScriptedEngine | None = None) -> ScriptedEngine:
    engine = engine or ScriptedEngine()
    app.engine = lambda name="default": engine  # type: ignore[assignment]
    app.checker = lambda s, t: PassChecker()  # type: ignore[assignment]
    return engine
