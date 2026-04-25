"""Fence registry — protect inline ``[? ... ?]`` style markers.

Some subtitle conventions use paired markers like ``[? proceed ?]`` or
``[! unsure !]`` to flag uncertain or negative segments. These markers
contain sentence-terminating punctuation but should be treated as a
single opaque token by:

* The SRT cleaner — so ``C7`` does not eat the leading space inside the
  closing ``?]``, leaving the marker symmetric.
* The sentence splitter — so the inner ``?`` is not seen as a sentence
  boundary and the marker is not torn apart.

The registry is intentionally string-based and language-agnostic: a
``Fence`` is just an open/close string pair. Callers may extend the
default set or pass their own list at the call site.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Fence:
    """An open/close string pair that protects an inline span."""

    open: str
    close: str

    def __post_init__(self) -> None:
        if not self.open or not self.close:
            raise ValueError("Fence open/close must be non-empty")


# Default registry — covers ``[? ... ?]`` (uncertain) and ``[! ... !]``
# (notable / negative) markers commonly found in transcribed subtitles.
DEFAULT_FENCES: tuple[Fence, ...] = (
    Fence("[?", "?]"),
    Fence("[!", "!]"),
)


def _build_finder(fences: Sequence[Fence]) -> re.Pattern[str]:
    """Build a single regex that matches any registered fence span.

    The pattern uses ``re.escape`` and a non-greedy body so nested or
    sequential markers do not over-match. The longest open string is
    tried first to support future fence pairs that share a prefix.
    """

    if not fences:
        # Pattern that never matches — keeps the API uniform.
        return re.compile(r"(?!x)x")
    sorted_fences = sorted(fences, key=lambda f: -len(f.open))
    parts = [f"(?:{re.escape(f.open)}.*?{re.escape(f.close)})" for f in sorted_fences]
    return re.compile("|".join(parts), re.DOTALL)


def find_fence_spans(
    text: str,
    fences: Sequence[Fence] = DEFAULT_FENCES,
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, text)`` triples for every fenced region.

    Spans are non-overlapping and ordered left-to-right. Empty input
    yields an empty list.
    """

    if not text or not fences:
        return []
    pattern = _build_finder(fences)
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


_SENTINEL_OPEN = "\u27e6"  # ⟦ MATHEMATICAL LEFT WHITE SQUARE BRACKET (cat Ps)
_SENTINEL_CLOSE = "\u27e7"  # ⟧ MATHEMATICAL RIGHT WHITE SQUARE BRACKET (cat Pe)


def mask_fences(
    text: str,
    fences: Sequence[Fence] = DEFAULT_FENCES,
) -> tuple[str, list[str]]:
    """Replace each fenced span with an opaque PUA sentinel.

    Returns ``(masked_text, mapping)`` where ``mapping[i]`` is the
    original text of the *i*-th fence. Sentinels use Unicode Private
    Use Area characters (``\\uE000``..``\\uE001``) so they don't collide
    with normal text and survive nearly all tokenizers as discrete
    tokens.

    The string-level inverse is :func:`unmask_fences`.
    """

    spans = find_fence_spans(text, fences)
    if not spans:
        return text, []
    parts: list[str] = []
    mapping: list[str] = []
    cursor = 0
    for start, end, original in spans:
        parts.append(text[cursor:start])
        parts.append(f"{_SENTINEL_OPEN}{len(mapping)}{_SENTINEL_CLOSE}")
        mapping.append(original)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), mapping


_SENTINEL_RE = re.compile(rf"{_SENTINEL_OPEN}(\d+){_SENTINEL_CLOSE}")


def unmask_fences(text: str, mapping: Sequence[str]) -> str:
    """Restore fenced spans replaced by :func:`mask_fences`."""

    if not mapping or _SENTINEL_OPEN not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(mapping):
            return mapping[idx]
        return match.group(0)

    return _SENTINEL_RE.sub(_sub, text)


def split_with_fences(
    text: str,
    splitter: Callable[[str], list[str]],
    fences: Sequence[Fence] = DEFAULT_FENCES,
) -> list[str]:
    """Tokenize ``text`` while keeping each fenced span as one token.

    The non-fence regions are passed to ``splitter`` (typically
    ``ops.split``) and concatenated with the fenced spans inserted as
    single tokens. This keeps the boundary finder from seeing the
    inner ``?`` as a sentence terminator.
    """

    spans = find_fence_spans(text, fences)
    if not spans:
        return splitter(text) if text else []
    out: list[str] = []
    cursor = 0
    for start, end, original in spans:
        if cursor < start:
            chunk = text[cursor:start]
            if chunk:
                out.extend(splitter(chunk))
        out.append(original)
        cursor = end
    if cursor < len(text):
        out.extend(splitter(text[cursor:]))
    return out


__all__ = [
    "Fence",
    "DEFAULT_FENCES",
    "find_fence_spans",
    "mask_fences",
    "unmask_fences",
    "split_with_fences",
]


def _ensure_iterable(fences: Iterable[Fence] | None) -> Sequence[Fence]:
    """Coerce ``fences`` to a tuple, falling back to the default set."""

    if fences is None:
        return DEFAULT_FENCES
    if isinstance(fences, tuple):
        return fences
    return tuple(fences)
