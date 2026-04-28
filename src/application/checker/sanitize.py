"""Output sanitizers — pure text→text transforms run *before* checker rules.

Backfills the cleanup logic that the legacy ``Agent.check_response`` and
``metadata.beautify_text`` performed inline, but which the new
:func:`translate_with_verify` had dropped (only ``.strip()`` was kept).

Each sanitizer is a tiny class conforming to :class:`Sanitizer`::

    class MySanitizer:
        name: str
        def sanitize(self, source: str, translation: str) -> str: ...

A :class:`SanitizerChain` runs them in order. By design sanitizers do
not produce :class:`Issue` objects — failed cleanup is silently skipped;
quality validation is the checker's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Sanitizer(Protocol):
    """Pure text transform applied before checker rules run."""

    @property
    def name(self) -> str: ...

    def sanitize(self, source: str, translation: str) -> str: ...


# -------------------------------------------------------------------
# Concrete sanitizers
# -------------------------------------------------------------------


class BackticksStrip:
    """Strip leading/trailing backticks and code-fence newlines.

    Mirrors ``re.sub(r"^`+|`+$|^\\n+|\\n+$", "", response)`` from the
    legacy ``Agent.check_response``.  Also removes any in-string
    backticks (LLMs often wrap a single token in ``\\`...\\```).
    """

    @property
    def name(self) -> str:
        return "backticks_strip"

    def sanitize(self, source: str, translation: str) -> str:
        t = re.sub(r"^`+|`+$|^\n+|\n+$", "", translation)
        return t.replace("`", "")


class TrailingAnnotationStrip:
    """Strip LLM-added trailing parenthesised annotations.

    The legacy regex was ``（注.*）$|（说明.*）$``; we generalise to any
    full-width parenthesised note ending the string when its content
    starts with one of the configured prefixes.
    """

    __slots__ = ("_prefixes",)

    def __init__(self, prefixes: tuple[str, ...] | None = None) -> None:
        self._prefixes = prefixes or ("注", "说明", "注释")

    @property
    def name(self) -> str:
        return "trailing_annotation_strip"

    def sanitize(self, source: str, translation: str) -> str:
        for prefix in self._prefixes:
            translation = re.sub(
                rf"（{prefix}[^（）]*?）\s*[,.?;!，。？；！]*$",
                "",
                translation,
            )
        return translation.strip()


class ColonToPunctuation:
    """Replace a trailing ``：`` with the source-mirrored punctuation.

    If the source ends with one of ``. , ! ?`` and the translation ends
    with ``：`` (LLM artefact), swap to the matching CJK punctuation.
    """

    _MAP = {".": "。", ",": "，", "!": "！", "?": "？"}

    @property
    def name(self) -> str:
        return "colon_to_punctuation"

    def sanitize(self, source: str, translation: str) -> str:
        src = source.rstrip()
        tgt = translation.rstrip()
        if not tgt.endswith("：") or not src:
            return translation
        last_src = src[-1]
        if last_src in self._MAP and last_src != ":":
            return tgt[:-1] + self._MAP[last_src]
        return translation


class QuoteStrip:
    """Strip a single matched layer of surrounding quote characters.

    Handles full-width ``“…”`` ``‘…’`` and half-width ``"…"`` ``'…'``.
    Repeats once if multiple layers (CJK + ASCII) are wrapping.
    """

    _PATTERNS: tuple[tuple[str, str], ...] = (
        ("“", "”"),
        ("‘", "’"),
        ('"', '"'),
        ("'", "'"),
    )

    @property
    def name(self) -> str:
        return "quote_strip"

    def sanitize(self, source: str, translation: str) -> str:
        for _ in range(2):
            stripped = translation
            for opener, closer in self._PATTERNS:
                if len(stripped) >= 2 and stripped.startswith(opener) and stripped.endswith(closer):
                    stripped = stripped[len(opener) : -len(closer)]
                    break
            if stripped == translation:
                break
            translation = stripped
        return translation


class LeadingPunctStrip:
    """Strip leading ``，`` / ``、`` / ``。`` artifacts.

    LLMs occasionally start a translation with a stray comma when the
    sentence is a continuation of a previous turn.
    """

    @property
    def name(self) -> str:
        return "leading_punct_strip"

    def sanitize(self, source: str, translation: str) -> str:
        return re.sub(r"^[，、。\s]+", "", translation)


# -------------------------------------------------------------------
# Chain
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SanitizerChain:
    """Apply a sequence of :class:`Sanitizer` instances in order."""

    sanitizers: tuple[Sanitizer, ...]

    def sanitize(self, source: str, translation: str) -> str:
        for s in self.sanitizers:
            translation = s.sanitize(source, translation)
        return translation


def default_sanitizer_chain() -> SanitizerChain:
    """Return the default chain mirroring the legacy cleanup order."""
    return SanitizerChain(
        sanitizers=(
            BackticksStrip(),
            TrailingAnnotationStrip(),
            ColonToPunctuation(),
            QuoteStrip(),
            LeadingPunctStrip(),
        )
    )
