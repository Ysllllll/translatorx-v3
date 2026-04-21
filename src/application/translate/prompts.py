"""Default translation system prompts keyed by ``(src_lang, tgt_lang)``.

These defaults are applied by :func:`llm_ops.translate.translate_with_verify`
when the caller does not supply a ``system_prompt`` **and**
``context.system_prompt_template`` is empty.

Prompts follow the legacy TranslatorX "意译" style (cf.
``old/translatorx/prompt.py`` + ``translate.py::get_system_prompt_with_topic``)
adapted for runtime-layer use.

Topic/field are pulled from ``context.terms_provider.metadata`` (when the
provider is ready) and spliced into the prompt via a language-aware
*scope line*. Missing metadata keys render as empty strings so the prompt
gracefully degrades to the generic form.

Extending:

To add a new language pair register a template string with ``{scope_line}``
/ ``{src_lang}`` / ``{tgt_lang}`` / ``{topic}`` / ``{field}`` placeholders
via :func:`register_default_prompt`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from .context import TranslationContext


_ScopeLineFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Scope-line helpers (locale-aware)
# ---------------------------------------------------------------------------


def _zh_scope_line(topic: str, field_: str) -> str:
    if topic and field_:
        return f"本段属于 {field_} 领域关于 {topic} 的内容。\n\n"
    if topic:
        return f"本段是关于 {topic} 的内容。\n\n"
    if field_:
        return f"本段属于 {field_} 领域的内容。\n\n"
    return ""


def _en_scope_line(topic: str, field_: str) -> str:
    if topic and field_:
        return f"This segment is about {topic} in the {field_} domain.\n\n"
    if topic:
        return f"This segment is about {topic}.\n\n"
    if field_:
        return f"This segment belongs to the {field_} domain.\n\n"
    return ""


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


_EN_ZH_TEMPLATE = (
    "您是一位专业的英译中意译专家，请将提供的英文句子意译成流畅、自然的简体中文。\n"
    "{scope_line}"
    "要求：\n"
    "1. 意译结果流畅自然、通俗易懂，避免生硬的翻译腔；\n"
    "2. 在意译时修正明显错误的内容；\n"
    '3. 保持专业术语、人名、代码标识符原文不译，例如人名 "John"、专有名词 "token"；\n'
    "4. 数学口语表达请转换为 LaTeX 格式，例如 $\\sigma$；\n"
    "5. 结合上下文和背景知识提升翻译质量；\n"
    "6. 仅输出意译结果，不得附加任何解释、注释或说明。"
)


_GENERIC_TEMPLATE = (
    "You are a professional {src_lang}-to-{tgt_lang} translator. "
    "Translate the given text into fluent, natural {tgt_lang}.\n"
    "{scope_line}"
    "Requirements:\n"
    "1. Output must be fluent and idiomatic; avoid literal / mechanical "
    "translation.\n"
    "2. Preserve proper nouns, code identifiers, and domain-specific terms.\n"
    "3. Keep math expressions in LaTeX, e.g. $\\sigma$.\n"
    "4. Use the surrounding context to improve quality.\n"
    "5. Output ONLY the translation — no explanations, notes or prefixes."
)


_REGISTRY: dict[tuple[str, str], tuple[str, _ScopeLineFn]] = {
    ("en", "zh"): (_EN_ZH_TEMPLATE, _zh_scope_line),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_default_prompt(
    src_lang: str,
    tgt_lang: str,
    template: str,
    *,
    scope_line: _ScopeLineFn | None = None,
) -> None:
    """Register a default system prompt for a language pair.

    The template is filled via ``str.format_map`` and receives the keys
    ``src_lang``, ``tgt_lang``, ``topic``, ``field``, ``scope_line`` (all
    optional — missing metadata renders as an empty string).

    ``scope_line`` chooses the locale for the topic/field sentence. If
    omitted, Chinese targets default to :func:`_zh_scope_line`, everything
    else defaults to :func:`_en_scope_line`.
    """

    key = (src_lang.lower(), tgt_lang.lower())
    chosen = scope_line or (_zh_scope_line if tgt_lang.lower().startswith("zh") else _en_scope_line)
    _REGISTRY[key] = (template, chosen)


def get_default_system_prompt(context: "TranslationContext") -> str:
    """Return the default system prompt for *context*'s language pair.

    Falls back to :data:`_GENERIC_TEMPLATE` if the pair is not registered.
    Topic/field are sourced from ``context.terms_provider.metadata`` when
    the provider is ready.
    """

    src = context.source_lang.lower()
    tgt = context.target_lang.lower()
    entry = _REGISTRY.get((src, tgt))
    if entry is None:
        template = _GENERIC_TEMPLATE
        scope_fn = _en_scope_line
    else:
        template, scope_fn = entry

    metadata: dict[str, object] = {}
    provider = context.terms_provider
    if getattr(provider, "ready", False):
        metadata = dict(provider.metadata or {})

    topic = str(metadata.get("topic", "") or "")
    field_ = str(metadata.get("field", "") or "")

    filler = defaultdict(str)
    filler.update(
        {
            "src_lang": context.source_lang,
            "tgt_lang": context.target_lang,
            "topic": topic,
            "field": field_,
            "scope_line": scope_fn(topic, field_),
        }
    )
    return template.format_map(filler)


__all__ = [
    "get_default_system_prompt",
    "register_default_prompt",
]
