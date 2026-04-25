"""STEP 6 — Prompt degradation demo (FlakyEngine).

Forces ``translate_with_verify`` to walk through the 4-level prompt degrade
ladder by wrapping a real engine in :class:`_FlakyEngine` that returns a
checker-failing response for the first N calls.
"""

from __future__ import annotations

from rich.table import Table

from _demo_shared import console, step, truncate
from api.trx import create_context
from application.checker import default_checker
from application.translate import ContextWindow, translate_with_verify
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult
from ports.engine import LLMEngine, Message


def _detect_prompt_level(messages: list[Message]) -> str:
    """Map message structure → degrade level (mirror translate.py builders)."""
    n = len(messages)
    if n == 1:
        return "L1" if messages[0]["role"] == "system" else "L3"
    if n == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        return "L2"
    return "L0"


class _FlakyEngine:
    """Engine wrapper that returns a checker-failing response for the first N calls.

    Wrapping a real engine means the final accepted attempt still produces a
    real translation, so the demo output is meaningful.
    """

    def __init__(self, real: LLMEngine, *, fail_n: int = 3, bad_text: str = "???") -> None:
        self._real = real
        self._fail_n = fail_n
        self._bad_text = bad_text
        self.attempts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return getattr(self._real, "model", "flaky")

    async def complete(self, messages: list[Message], **kwargs) -> CompletionResult:
        level = _detect_prompt_level(messages)
        if len(self.attempts) < self._fail_n:
            self.attempts.append((level, "BAD"))
            return CompletionResult(text=self._bad_text, usage=None)
        result = await self._real.complete(messages, **kwargs)
        self.attempts.append((level, "REAL"))
        return result

    async def stream(self, messages: list[Message], **kwargs):  # pragma: no cover
        async for chunk in self._real.stream(messages, **kwargs):
            yield chunk


async def step_degrade(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    terms: dict[str, str] | None,
) -> None:
    """Translate ONE record through a FlakyEngine. Expect attempts L0 BAD → L1 BAD → L2 BAD → L3 REAL."""
    step(
        "STEP 6",
        "Prompt degradation (FlakyEngine — 前 3 次返回坏译文)",
        "验证 4 级 prompt 降级路径：L0 → L1 → L2 → L3 fallback (bare)。",
    )
    flaky = _FlakyEngine(engine, fail_n=3, bad_text="**bad** translation with markdown artifacts")
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    win = ContextWindow(size=4)
    win.add("hello", "你好")
    win.add("world", "世界")

    target_rec = records[0]
    result = await translate_with_verify(target_rec.src_text, flaky, ctx, checker, win)
    console.print(
        f"  attempts={result.attempts}  accepted={result.accepted}  final_translation=[cyan]{truncate(result.translation, 120)}[/cyan]"
    )
    tbl = Table(
        title="attempt → prompt level",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("level", width=8)
    tbl.add_column("outcome", width=8)
    for i, (lvl, outcome) in enumerate(flaky.attempts, 1):
        color = "green" if outcome == "REAL" else "yellow"
        tbl.add_row(str(i), lvl, f"[{color}]{outcome}[/{color}]")
    console.print(tbl)
    expected_levels = ["L0", "L1", "L2", "L3"]
    actual_levels = [lvl for lvl, _ in flaky.attempts]
    if actual_levels[: len(expected_levels)] == expected_levels:
        console.print("  [bold green]✓ 4 级降级路径全部走过[/bold green]")
    else:
        console.print(f"  [yellow]⚠ levels seen: {actual_levels} (expected start with {expected_levels})[/yellow]")
