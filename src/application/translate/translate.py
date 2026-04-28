"""Translate-with-verify micro-loop.

Implements prompt degradation on retry (following the legacy TranslatorX
pattern) and integrates with :class:`~llm_ops.Checker` for quality gating.

The retry strategy degrades the *prompt structure* — not the checker
standard — on each attempt:

1. Full context (system prompt + history window + user message)
2. Compressed context (history folded into system prompt)
3. Minimal (no history, system prompt + user)
4. Bare (single user message with inline "translate to <lang>" instruction,
   no system prompt) — last-resort fallback matching the legacy behaviour.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .context import ContextWindow, TranslationContext
from .prompts import get_default_system_prompt
from ports.engine import LLMEngine, Message
from ports.retries import retry_until_valid
from application.checker import CheckReport, Checker, SanitizerChain, default_sanitizer_chain


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


# Language code → display name used by the bare fallback prompt.
_TARGET_LANG_NAMES: dict[str, str] = {
    "zh": "简体中文",
    "zh-cn": "简体中文",
    "zh-hans": "简体中文",
    "zh-tw": "繁體中文",
    "zh-hant": "繁體中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "français",
    "de": "Deutsch",
    "es": "español",
    "ru": "русский",
    "pt": "português",
    "vi": "tiếng Việt",
}


def _lang_name(code: str) -> str:
    return _TARGET_LANG_NAMES.get(code.lower(), code)


def _build_messages_bare(target_lang: str, user_text: str) -> list[Message]:
    """Level 3: single user message, no system prompt, minimal instruction.

    Matches the legacy TranslatorX "last-resort" fallback — a single
    imperative user turn in the target language, e.g.
    ``请将以下内容翻译为简体中文：\\n\\n<text>``.
    """
    name = _lang_name(target_lang)
    # Localize the instruction when the target is Chinese; otherwise fall
    # back to English. This mirrors the old behaviour for zh-target.
    if target_lang.lower().startswith("zh"):
        instruction = f"请将以下内容翻译为{name}："
    else:
        instruction = f"Translate the following to {name}:"
    return [{"role": "user", "content": f"{instruction}\n{user_text}"}]


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
    sanitizer: SanitizerChain | None = None,
) -> TranslateResult:
    """Translate *source* with quality verification and prompt degradation.

    Prompt structure shrinks on each retry while the checker standard stays
    constant:

    * attempt 0 → Level 0 (system + history + user)
    * attempt 1 → Level 1 (compressed single-turn)
    * attempt 2 → Level 2 (system + user, no history)
    * attempt 3+ → Level 3 (bare one-message fallback)

    If every attempt fails the last translation is returned with
    ``accepted=False`` and NOT added to the history window.
    """
    max_retries = context.max_retries
    sanitize_chain = sanitizer if sanitizer is not None else default_sanitizer_chain()

    # Merge provider terms (if ready) in front of user-supplied frozen_pairs.
    provider = context.terms_provider
    if provider.ready:
        provider_terms = await provider.get_terms()
        effective_pairs = tuple(provider_terms.items()) + context.frozen_pairs
    else:
        effective_pairs = context.frozen_pairs

    # Optional system-prompt template interpolated from provider metadata.
    resolved_system_prompt = _resolve_system_prompt(system_prompt, context)

    context_messages = window.build_messages(effective_pairs)

    def _messages_for_attempt(attempt: int) -> list[Message]:
        """Return the message list used at a given degradation level."""
        if attempt >= len(_PROMPT_LEVELS):
            return _build_messages_bare(context.target_lang, source)
        builder = _PROMPT_LEVELS[attempt]
        return builder(resolved_system_prompt, context_messages, source)

    # Track the last (translation, report) so we can fall back on exhaustion.
    last_seen: dict[str, object] = {"translation": "", "report": CheckReport.ok()}

    async def _call(attempt: int) -> tuple[str, CheckReport]:
        messages = _messages_for_attempt(attempt)
        result = await engine.complete(messages)
        translation = sanitize_chain.sanitize(source, result.text.strip())
        report = checker.check(source, translation)
        last_seen["translation"] = translation
        last_seen["report"] = report
        return translation, report

    def _validate(pair: tuple[str, CheckReport]):
        translation, report = pair
        if report.passed:
            return True, (translation, report), ""
        return False, None, "checker rejected"

    outcome = await retry_until_valid(
        _call,
        validate=_validate,
        max_retries=max_retries,
    )

    if outcome.accepted:
        translation, report = outcome.value  # type: ignore[misc]
        window.add(source, translation)
        return TranslateResult(
            translation=translation,
            report=report,
            attempts=outcome.attempts,
            accepted=True,
        )

    # Exhausted retries — return the last seen translation without adding to history.
    return TranslateResult(
        translation=last_seen["translation"],  # type: ignore[arg-type]
        report=last_seen["report"],  # type: ignore[arg-type]
        attempts=outcome.attempts,
        accepted=False,
    )


def _resolve_system_prompt(system_prompt: str, context: TranslationContext) -> str:
    """Resolve the final system prompt used for translation.

    Priority (highest wins):

    1. ``context.system_prompt_template`` interpolated with provider
       metadata — application-level override that supersedes the
       per-call ``system_prompt`` (e.g. a runtime that rewrites the
       prompt per topic/field).
    2. Non-empty ``system_prompt`` passed by the caller (typically the
       ``TranslateNodeConfig.system_prompt`` on the translate processor).
    3. Language-pair default from :mod:`llm_ops.prompts`, interpolated
       with provider metadata (``topic`` / ``field`` / ``scope_line``).
       This guarantees the LLM always receives *some* instruction,
       matching the legacy TranslatorX behaviour (cf.
       ``get_system_prompt_with_topic``).
    """
    template = context.system_prompt_template
    if template:
        metadata = context.terms_provider.metadata if context.terms_provider.ready else {}
        return template.format_map(defaultdict(str, metadata))
    if system_prompt:
        return system_prompt
    return get_default_system_prompt(context)
