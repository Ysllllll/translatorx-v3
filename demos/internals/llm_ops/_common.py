"""Shared helpers + fake engines for the llm_ops walk-through chapters."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from dataclasses import dataclass

import httpx

from _print import banner as _banner, step as _step  # noqa: E402

from api import trx
from application.checker import CheckReport, Severity
from application.translate import ContextWindow
from domain.model.usage import CompletionResult, Usage
from ports.engine import Message


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"

SEP = "═" * 72
SUB = "─" * 72


def truncate(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def header(title: str) -> None:
    """Major heading — delegates to demos._print for unified styling."""
    _banner(title)


def sub(title: str) -> None:
    """Sub-section heading — delegates to demos._print.step()."""
    _step("·", title)


def print_messages(messages: list[Message], *, limit: int = 120) -> None:
    """Compact print of one engine.complete() messages list."""
    if not messages:
        print("    (空)")
        return
    for i, m in enumerate(messages):
        content = truncate(m["content"], limit)
        print(f"    [{i:2d}] {m['role']:9s} | {content}")


def print_system_prompt(prompt: str) -> None:
    for line in prompt.splitlines():
        print(f"    │ {line}")


def print_window(window: ContextWindow) -> None:
    if len(window) == 0:
        print("    (空)")
        return
    pairs = window.build_messages()
    for i in range(0, len(pairs), 2):
        print(f"    [{i // 2}] src: {truncate(pairs[i]['content'], 90)}")
        print(f"        tgt: {truncate(pairs[i + 1]['content'], 90)}")


def print_report(report: CheckReport) -> None:
    status = "✓ passed" if report.passed else "✗ failed"
    if not report.issues:
        print(f"    {status}  (no issues)")
        return
    print(f"    {status}  ({len(report.issues)} issue(s))")
    for iss in report.issues:
        marker = "!" if iss.severity == Severity.ERROR else "·"
        print(f"      {marker} [{iss.severity.value:7s}] {iss.rule}: {iss.message}")


@dataclass
class LoggingEngine:
    """Wrap a real engine, intercept complete()/stream() to record last messages."""

    inner: object
    last_messages: list[Message] | None = None

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.last_messages = list(messages)
        return await self.inner.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = list(messages)
        async for chunk in self.inner.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk


class ScriptedEngine:
    """Fake engine that replies from a queued list."""

    def __init__(self, scripted_replies: list[str]) -> None:
        self._replies = list(scripted_replies)
        self.call_log: list[list[Message]] = []

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        return CompletionResult(text=reply, usage=Usage())

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        for tok in reply:
            yield tok


def make_engine(*, max_tokens: int = 2048):
    return trx.create_engine(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


async def llm_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{LLM_BASE_URL}/models")
            return r.status_code == 200
    except Exception:
        return False
