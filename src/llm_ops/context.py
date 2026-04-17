"""Translation context — value objects that travel through the pipeline.

Defines :class:`TermsProvider` (Protocol), :class:`StaticTerms`
(always-ready implementation), :class:`ContextWindow` (sliding history),
and :class:`TranslationContext` (immutable carrier of all translation state).

Terms follow a simple **one-shot state transition** model: a provider is
either *not ready* (``ready=False``, no terms yet) or *ready*
(``ready=True``, terms finalized — including the degraded "empty terms"
state after failure). There is no version tracking; once a provider
becomes ready, its terms do not change again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .protocol import Message


# ---------------------------------------------------------------------------
# TermsProvider Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TermsProvider(Protocol):
    """Async interface for supplying domain-specific terminology.

    Implementations live on a 2-state machine:
    - ``ready == False`` — terms not yet available; ``get_terms()`` returns ``{}``
    - ``ready == True``  — terms finalized (may still be empty on failure)

    ``metadata`` carries auxiliary information such as ``topic``, ``title``,
    and ``description`` that callers may interpolate into system prompts via
    :attr:`TranslationContext.system_prompt_template`.
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


# ---------------------------------------------------------------------------
# StaticTerms — fixed glossary known upfront
# ---------------------------------------------------------------------------

class StaticTerms:
    """Pre-defined terminology that never changes.

    Always ``ready``; ``request_generation`` is a no-op.  Suitable for
    batch translation where the glossary is known upfront.
    """

    __slots__ = ("_terms", "_metadata")

    def __init__(
        self,
        terms: dict[str, str] | None = None,
        *,
        metadata: dict[str, str] | None = None,
    ):
        self._terms: dict[str, str] = dict(terms) if terms else {}
        self._metadata: dict[str, str] = dict(metadata) if metadata else {}

    @property
    def ready(self) -> bool:
        return True

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    async def request_generation(self, texts: list[str]) -> None:
        return None

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)


# ---------------------------------------------------------------------------
# ContextWindow — sliding translation-pair history
# ---------------------------------------------------------------------------

class ContextWindow:
    """Sliding window of recent (source, translation) pairs.

    Used to build few-shot context for the LLM so that consecutive
    sentences share a consistent translation style.
    """

    __slots__ = ("_size", "_history")

    def __init__(self, size: int = 4):
        self._size = size
        self._history: deque[tuple[str, str]] = deque(maxlen=size)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._history)

    def add(self, source: str, translation: str) -> None:
        """Append a translation pair to the window."""
        self._history.append((source, translation))

    def clear(self) -> None:
        self._history.clear()

    def build_messages(
        self,
        frozen_pairs: tuple[tuple[str, str], ...] = (),
    ) -> list[Message]:
        """Build context messages from frozen pairs + recent history.

        Frozen pairs appear first (high-quality reference translations),
        followed by the sliding window contents.  Each pair becomes a
        user→assistant message pair.
        """
        messages: list[Message] = []
        for src, dst in (*frozen_pairs, *self._history):
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        return messages


# ---------------------------------------------------------------------------
# TranslationContext — immutable carrier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslationContext:
    """Immutable value object carrying all state needed for translation.

    Constructed by the caller and passed into the pipeline.  The pipeline
    never mutates the context, though the ``terms_provider`` may have
    internal state that transitions from not-ready to ready exactly once.

    If ``system_prompt_template`` is non-empty, it overrides
    :attr:`pipeline.config.TranslateNodeConfig.system_prompt` and is filled
    via ``str.format_map`` with the provider's ``metadata`` (missing keys
    become empty strings) at each translate call.
    """

    source_lang: str
    target_lang: str
    terms_provider: TermsProvider = field(default_factory=StaticTerms)
    frozen_pairs: tuple[tuple[str, str], ...] = ()
    window_size: int = 4
    max_retries: int = 3
    system_prompt_template: str = ""
