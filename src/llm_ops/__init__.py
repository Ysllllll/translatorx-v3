"""LLM operations: translation, term extraction, quality checking.

Provides the LLMEngine protocol, translation context management,
the translate-with-verify micro-loop, and the Checker subsystem.

For checker-specific types (Rule classes, LangProfile, ProfileOverrides),
import from ``checker`` directly::

    from checker import LengthRatioRule, LangProfile, get_profile
"""

from ._context import (
    ContextWindow,
    StaticTerms,
    TermsProvider,
    TranslationContext,
)
from ._protocol import LLMEngine, Message
from ._translate import TranslateResult, translate_with_verify
from checker import (
    CheckReport,
    Checker,
    Severity,
    default_checker,
)
from .engines import OpenAICompatEngine
from .engines._openai_compat import EngineConfig

__all__ = [
    # Protocol
    "LLMEngine",
    "Message",
    # Context
    "TermsProvider",
    "StaticTerms",
    "ContextWindow",
    "TranslationContext",
    # Translate
    "TranslateResult",
    "translate_with_verify",
    # Engine
    "OpenAICompatEngine",
    "EngineConfig",
    # Checker (high-level API only; for rules/profiles: from checker import ...)
    "Severity",
    "CheckReport",
    "Checker",
    "default_checker",
]

