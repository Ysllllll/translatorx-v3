"""Length-based text splitter."""

from __future__ import annotations

from typing import Protocol

from lang_ops._core._types import Span


class _HasSplitJoin(Protocol):
    """Protocol for the lang_ops methods we need."""

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]: ...
    def join(self, tokens: list[str]) -> str: ...
    def length(self, text: str, **kwargs: int) -> int: ...


def split_by_length(
    text: str,
    ops: _HasSplitJoin,
    max_length: int,
) -> list[Span]:
    """Split *text* into chunks whose ``ops.length()`` ≤ *max_length*.

    Always tokenises with ``ops.split()`` (word mode) and accumulates
    tokens until the joined length would exceed the limit.  If a single
    token already exceeds *max_length* it is emitted as-is (the minimum
    unit is one token — we never break a token to preserve readability).

    Returns Span objects with ``start=-1, end=-1`` because tokenise+join
    may alter whitespace, making character offsets unreliable.
    """
    if not text:
        return []

    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[Span] = []
    chunk_tokens: list[str] = []

    for token in tokens:
        if chunk_tokens:
            joined_len = ops.length(ops.join(chunk_tokens + [token]))
            if joined_len > max_length:
                result.append(Span(ops.join(chunk_tokens), -1, -1))
                chunk_tokens = []

        chunk_tokens.append(token)

    if chunk_tokens:
        result.append(Span(ops.join(chunk_tokens), -1, -1))

    return result
