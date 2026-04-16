"""Translation context — value objects that travel through the pipeline.

Defines :class:`TermsProvider` (Protocol), :class:`StaticTerms`
(simplest implementation), :class:`ContextWindow` (sliding history),
and :class:`TranslationContext` (immutable carrier of all translation state).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ._protocol import Message


# ---------------------------------------------------------------------------
# TermsProvider Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TermsProvider(Protocol):
    """Async interface for supplying domain-specific terminology."""

    @property
    def version(self) -> int:
        """Monotonically increasing version; bumped when terms change."""
        ...

    async def get_terms(self) -> dict[str, str]:
        """Return current ``{source_term: target_term}`` mapping."""
        ...

    async def update(self, text_batch: list[str]) -> bool:
        """Ingest new source texts; return True if terms changed (version bumped)."""
        ...


# ---------------------------------------------------------------------------
# StaticTerms — Phase 1 implementation
# ---------------------------------------------------------------------------

class StaticTerms:
    """Pre-defined terminology that never changes.

    Suitable for batch translation where the glossary is known upfront.
    ``version`` is always 1; ``update`` always returns False.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: dict[str, str] | None = None):
        self._terms: dict[str, str] = dict(terms) if terms else {}

    @property
    def version(self) -> int:
        return 1

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    async def update(self, text_batch: list[str]) -> bool:
        return False


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
    internal state that evolves (version increments).
    """

    source_lang: str
    target_lang: str
    terms_provider: TermsProvider = field(default_factory=StaticTerms)
    frozen_pairs: tuple[tuple[str, str], ...] = ()
    window_size: int = 4
    max_retries: int = 3
    retranslate_on_terms_update: bool = True
    retranslate_max_lookback: int = 20
