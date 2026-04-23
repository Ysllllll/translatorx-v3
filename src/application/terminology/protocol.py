"""TermsProvider Protocol — async interface for supplying terminology.

The provider follows a simple **one-shot state transition** model: it is
either *not ready* (``ready=False``, ``get_terms()`` returns ``{}``) or
*ready* (``ready=True``, terms finalized — including the degraded "empty
terms" state after failure).  Once a provider becomes ready, its terms
do not change again.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TermsProvider(Protocol):
    """Async interface for supplying domain-specific terminology.

    Implementations live on a 2-state machine:
    - ``ready == False`` — terms not yet available; ``get_terms()`` returns ``{}``
    - ``ready == True``  — terms finalized (may still be empty on failure)

    ``metadata`` carries auxiliary information such as ``topic``, ``title``,
    and ``description`` that callers may interpolate into system prompts via
    :attr:`application.translate.context.TranslationContext.system_prompt_template`.
    """

    @property
    def ready(self) -> bool:
        """Whether terms have been finalized."""
        ...

    async def get_terms(self) -> dict[str, str]:
        """Return current ``{source_term: target_term}`` mapping."""
        ...

    async def request_generation(self, texts: list[str]) -> None:
        """Feed texts to the provider. Idempotent; may or may not trigger LLM work."""
        ...

    @property
    def metadata(self) -> dict[str, str]:
        """Auxiliary info (topic, title, description). Empty if not available."""
        ...
