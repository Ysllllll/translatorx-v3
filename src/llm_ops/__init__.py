"""LLM operations: translation, term extraction, quality checking.

Provides the LLMEngine protocol, translation context management,
the translate-with-verify micro-loop, and the Checker subsystem.

For checker-specific types (Rule classes, LangProfile, ProfileOverrides),
import from ``checker`` directly::

    from checker import LengthRatioRule, LangProfile, get_profile
"""

from .agents import TermsAgent, TermsAgentResult, parse_terms_response
from .context import (
    ContextWindow,
    StaticTerms,
    TermsProvider,
    TranslationContext,
)
from .protocol import LLMEngine, Message
from .prompts import get_default_system_prompt, register_default_prompt
from .providers import OneShotTerms, PreloadableTerms
from .translate import TranslateResult, translate_with_verify
from checker import (
    CheckReport,
    Checker,
    Severity,
    default_checker,
)
from model.usage import CompletionResult, Usage
from .engines import OpenAICompatEngine
from .engines.openai_compat import EngineConfig

__all__ = [
    # Protocol
    "LLMEngine",
    "Message",
    "CompletionResult",
    "Usage",
    # Context
    "TermsProvider",
    "StaticTerms",
    "PreloadableTerms",
    "OneShotTerms",
    "TermsAgent",
    "TermsAgentResult",
    "parse_terms_response",
    "ContextWindow",
    "TranslationContext",
    # Translate
    "TranslateResult",
    "translate_with_verify",
    # Prompt defaults
    "get_default_system_prompt",
    "register_default_prompt",
    # Engine
    "OpenAICompatEngine",
    "EngineConfig",
    # Checker (high-level API only; for rules/profiles: from checker import ...)
    "Severity",
    "CheckReport",
    "Checker",
    "default_checker",
]

