"""Translation use case — :func:`translate_with_verify` + context + prompts.

This package owns the *narrow* translation use case: turn a source-language
sentence into a target-language sentence, using LLM history, prompt
degradation on retry, and a system-prompt registry.

Sibling use cases live in their own packages:

* :mod:`application.terminology` — terminology / metadata extraction
* :mod:`application.summary`     — incremental summary
* :mod:`application.align`       — LLM-driven binary-split alignment
* :mod:`application.checker`     — translation quality validation rules

The :class:`TranslationContext` value object aggregates everything a
translate call needs (engine config, language pair, terminology provider,
chat history, prompt template) so processors stay stateless.
"""

from __future__ import annotations

from .context import (
    ContextWindow,
    StaticTerms,
    TermsProvider,
    TranslationContext,
    build_frozen_messages,
)
from .prompts import get_default_system_prompt, register_default_prompt
from .translate import TranslateResult, translate_with_verify
from .variant import VariantSpec, default_variant

__all__ = [
    "ContextWindow",
    "StaticTerms",
    "TermsProvider",
    "TranslateResult",
    "TranslationContext",
    "VariantSpec",
    "build_frozen_messages",
    "default_variant",
    "get_default_system_prompt",
    "register_default_prompt",
    "translate_with_verify",
]
