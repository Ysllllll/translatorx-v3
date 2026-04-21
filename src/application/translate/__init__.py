"""Application-layer translation use cases (translate + retries + context)."""

from .agents import (
    IncrementalSummaryAgent,
    IncrementalSummaryState,
    SummarySnapshot,
    TermsAgent,
    TermsAgentResult,
    parse_terms_response,
)
from .context import (
    ContextWindow,
    StaticTerms,
    TermsProvider,
    TranslationContext,
    build_frozen_messages,
)
from .prompts import get_default_system_prompt, register_default_prompt
from .providers import OneShotTerms, PreloadableTerms
from ports.retries import (
    AttemptOutcome,
    OnFailure,
    ValidateResult,
    resolve_on_failure,
    retry_until_valid,
)
from .translate import TranslateResult, translate_with_verify
from ports.engine import LLMEngine, Message
from domain.model import CompletionResult, Usage
from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine
from application.checker import CheckReport, Checker, Severity, default_checker

__all__ = [
    "LLMEngine",
    "Message",
    "CompletionResult",
    "Usage",
    "TermsProvider",
    "StaticTerms",
    "PreloadableTerms",
    "OneShotTerms",
    "TermsAgent",
    "TermsAgentResult",
    "parse_terms_response",
    "IncrementalSummaryAgent",
    "IncrementalSummaryState",
    "SummarySnapshot",
    "ContextWindow",
    "TranslationContext",
    "build_frozen_messages",
    "TranslateResult",
    "translate_with_verify",
    "retry_until_valid",
    "AttemptOutcome",
    "ValidateResult",
    "OnFailure",
    "resolve_on_failure",
    "get_default_system_prompt",
    "register_default_prompt",
    "OpenAICompatEngine",
    "EngineConfig",
    "Severity",
    "CheckReport",
    "Checker",
    "default_checker",
]
