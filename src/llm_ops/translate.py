"""Translate-with-verify micro-loop.

Implements prompt degradation on retry (following the legacy TranslatorX
pattern) and integrates with :class:`~llm_ops.Checker` for quality gating.

The retry strategy degrades the *prompt structure* — not the checker
standard — on each attempt:

1. Full context (system prompt + history window + user message)
2. Compressed context (history folded into system prompt)
3. Minimal (no history, plain prompt)
4. Accept whatever the model returns (fallback result not added to history)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .context import ContextWindow, TranslationContext
from .protocol import LLMEngine, Message
from checker import CheckReport, Checker


# ---------------------------------------------------------------------------
# Prompt builders — one per degradation level
# ---------------------------------------------------------------------------

def _build_messages_full(
    system_prompt: str,
    context_messages: list[Message],
    user_text: str,
) -> list[Message]:
    """Level 0: system + few-shot context + user message."""
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(context_messages)
    messages.append({"role": "user", "content": user_text})
    return messages


def _build_messages_compressed(
    system_prompt: str,
    context_messages: list[Message],
    user_text: str,
) -> list[Message]:
    """Level 1: history compressed into system prompt (single-turn)."""
    parts = [system_prompt] if system_prompt else []
    if context_messages:
        pairs: list[str] = []
        for i in range(0, len(context_messages), 2):
            src = context_messages[i]["content"]
            dst = context_messages[i + 1]["content"] if i + 1 < len(context_messages) else ""
            pairs.append(f"  {src} → {dst}")
        parts.append("Reference translations:\n" + "\n".join(pairs))
    parts.append(f"Translate:\n{user_text}")
    combined = "\n\n".join(parts)
    return [{"role": "system", "content": combined}]


def _build_messages_minimal(
    system_prompt: str,
    _context_messages: list[Message],
    user_text: str,
) -> list[Message]:
    """Level 2: no history, but keep system prompt for task context."""
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


_PROMPT_LEVELS = [
    _build_messages_full,
    _build_messages_compressed,
    _build_messages_minimal,
]


# ---------------------------------------------------------------------------
# TranslateResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslateResult:
    """Outcome of a single translate_with_verify call."""

    translation: str
    report: CheckReport
    attempts: int
    accepted: bool  # True if checker passed; False if fallback-accepted
    skipped: bool = False  # True if bypassed LLM (direct_translate, skip_long, etc.)


# ---------------------------------------------------------------------------
# Core micro-loop
# ---------------------------------------------------------------------------

async def translate_with_verify(
    source: str,
    engine: LLMEngine,
    context: TranslationContext,
    checker: Checker,
    window: ContextWindow,
    *,
    system_prompt: str = "",
) -> TranslateResult:
    """Translate *source* with quality verification and prompt degradation.

    On each retry the prompt structure degrades while the checker
    standard stays the same.  If all retries are exhausted the last
    translation is accepted without entering the context window.

    Args:
        source: Source-language text to translate.
        engine: LLM backend.
        context: Immutable translation context (langs, terms, etc.).
        checker: Quality checker instance.
        window: Sliding history window (mutated on success).
        system_prompt: Optional system-level instruction.

    Returns:
        A :class:`TranslateResult` with the translation and metadata.
    """
    max_retries = context.max_retries
    context_messages = window.build_messages(context.frozen_pairs)

    last_translation = ""
    last_report = CheckReport.ok()

    for attempt in range(max_retries + 1):
        # Pick prompt builder for this degradation level
        level = min(attempt, len(_PROMPT_LEVELS) - 1)
        builder = _PROMPT_LEVELS[level]

        # Rebuild context messages only for level 0 (others don't use them
        # or have them embedded in system prompt).
        if attempt > 0 and level == 0:
            context_messages = window.build_messages(context.frozen_pairs)

        messages = builder(system_prompt, context_messages, source)
        translation = await engine.complete(messages)
        translation = translation.strip()

        report = checker.check(source, translation)
        last_translation = translation
        last_report = report

        if report.passed:
            window.add(source, translation)
            return TranslateResult(
                translation=translation,
                report=report,
                attempts=attempt + 1,
                accepted=True,
            )

    # Exhausted retries — accept without adding to history
    return TranslateResult(
        translation=last_translation,
        report=last_report,
        attempts=max_retries + 1,
        accepted=False,
    )
