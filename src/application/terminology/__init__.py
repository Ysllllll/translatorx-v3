"""Terminology extraction use case.

Public surface:

* :class:`TermsProvider` Protocol — async interface
* :class:`StaticTerms` — pre-defined glossary, always ready
* :class:`PreloadableTerms` — batch generation
* :class:`OneShotTerms` — streaming generation
* :class:`TermsAgent` / :class:`TermsAgentResult` / :func:`parse_terms_response` —
  raw LLM bridge consumed by the providers above
"""

from __future__ import annotations

from .agent import TermsAgent, TermsAgentResult, parse_terms_response
from .protocol import TermsProvider
from .providers import OneShotTerms, PreloadableTerms, TermsOnFailure
from .static import StaticTerms

__all__ = [
    "OneShotTerms",
    "PreloadableTerms",
    "StaticTerms",
    "TermsAgent",
    "TermsAgentResult",
    "TermsOnFailure",
    "TermsProvider",
    "parse_terms_response",
]
