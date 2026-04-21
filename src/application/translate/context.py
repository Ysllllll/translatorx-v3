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

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ports.engine import Message


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
# ContextWindow — sliding translation-pair history (cache-friendly)
# ---------------------------------------------------------------------------

# Hardcoded primer demonstrating the "compact term list" format to the LLM
# and showing that spoken math should be rendered as LaTeX.  Ported from
# legacy TranslatorX ``ChatHistory.freeze`` (model.py:143-168).
_LATEX_PRIMER: tuple[tuple[str, str], ...] = (("Q equal gamma times beta", r"$Q = \gamma \times \beta$"),)


def build_frozen_messages(
    pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> list[Message]:
    """Pack a flat list of ``(src, tgt)`` term pairs into a compact few-shot
    block, matching legacy TranslatorX ``ChatHistory.freeze`` semantics.

    Structure::

        # fixed LaTeX primer pair (always included)
        user:      "Q equal gamma times beta"
        assistant: "$Q = \\gamma \\times \\beta$"

        # all term src-s joined with ", " in one user message,
        # all term tgt-s joined with "，" in one assistant message
        user:      "term1, term2, ..."
        assistant: "翻译1，翻译2，…"

    If there are 8 or more term pairs the payload is split into **two**
    consecutive user/assistant pairs (first 5 and rest) so the single
    message does not become unwieldy.

    The compact form is dramatically more prompt-cache friendly than
    emitting one user/assistant message pair per term: the LLM-facing
    prompt prefix stays the same length (4 or 6 messages) regardless of
    how many terms the provider returns.
    """
    messages: list[Message] = [
        {"role": "user", "content": _LATEX_PRIMER[0][0]},
        {"role": "assistant", "content": _LATEX_PRIMER[0][1]},
    ]
    if not pairs:
        return messages

    srcs = [s for s, _ in pairs]
    tgts = [t for _, t in pairs]
    if len(pairs) >= 8:
        messages.extend(
            [
                {"role": "user", "content": ", ".join(srcs[:5])},
                {"role": "assistant", "content": "，".join(tgts[:5])},
                {"role": "user", "content": ", ".join(srcs[5:])},
                {"role": "assistant", "content": "，".join(tgts[5:])},
            ]
        )
    else:
        messages.extend(
            [
                {"role": "user", "content": ", ".join(srcs)},
                {"role": "assistant", "content": "，".join(tgts)},
            ]
        )
    return messages


class ContextWindow:
    """Sliding window of recent (source, translation) pairs.

    Used to build few-shot context for the LLM so that consecutive
    sentences share a consistent translation style.

    **Bulk eviction (prompt-cache friendly).**  When the window overflows
    it drops ``max(int(size * evict_rate), 1)`` pairs at once instead of
    one, so the prompt prefix stays identical across several consecutive
    calls.  With the default ``evict_rate=0.5`` and ``size=4`` two
    consecutive translations reuse the exact same 3-pair prefix, which
    allows KV-cache-capable backends (vLLM prefix caching, Anthropic
    prompt caching, …) to actually hit.  Ported from legacy
    ``ChatHistory(max_size, evict_rate)``.
    """

    __slots__ = ("_size", "_evict_num", "_history")

    def __init__(self, size: int = 4, *, evict_rate: float = 0.5):
        self._size = size
        self._evict_num = max(int(size * evict_rate), 1)
        self._history: list[tuple[str, str]] = []

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._history)

    def add(self, source: str, translation: str) -> None:
        """Append a translation pair to the window.

        When ``len > size`` after append, drop :attr:`_evict_num` oldest
        pairs in a single slice (bulk eviction, see class doc).
        """
        self._history.append((source, translation))
        if len(self._history) > self._size:
            self._history = self._history[self._evict_num :]

    def clear(self) -> None:
        self._history.clear()

    def build_messages(
        self,
        frozen_pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        *,
        compact_frozen: bool = True,
    ) -> list[Message]:
        """Build context messages from frozen pairs + recent history.

        Frozen pairs appear first (high-quality reference translations),
        followed by the sliding window contents.

        Args:
            frozen_pairs: Terminology pairs injected as few-shot examples.
            compact_frozen: When ``True`` (default), pack all frozen
                pairs into the compact form produced by
                :func:`build_frozen_messages` (primer + 1 or 2
                concatenated pairs — cache friendly).  When ``False``,
                emit one user/assistant pair per term (legacy shape,
                kept for tests and debugging).
        """
        messages: list[Message] = []
        if frozen_pairs:
            if compact_frozen:
                messages.extend(build_frozen_messages(tuple(frozen_pairs)))
            else:
                for src, dst in frozen_pairs:
                    messages.append({"role": "user", "content": src})
                    messages.append({"role": "assistant", "content": dst})
        for src, dst in self._history:
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
