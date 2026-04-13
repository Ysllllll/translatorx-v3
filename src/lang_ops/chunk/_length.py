"""Length-based text splitter."""

from __future__ import annotations

from typing import Protocol


class _HasSplitJoin(Protocol):
    """Protocol for the lang_ops methods we need."""

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]: ...
    def join(self, tokens: list[str]) -> str: ...
    def length(self, text: str, **kwargs: int) -> int: ...


def split_tokens_by_length(
    tokens: list[str],
    ops: _HasSplitJoin,
    max_length: int,
) -> list[list[str]]:
    """Split a token array into groups whose joined length ≤ *max_length*.

    Accumulates tokens until ``ops.length(ops.join(chunk))`` would exceed
    the limit.  A single token exceeding *max_length* is emitted as-is
    (minimum unit = one token, never break a token).

    Args:
        tokens: Pre-tokenized array from ``ops.split()``.
        ops: Language ops providing ``join()`` and ``length()``.
        max_length: Upper bound on ``ops.length()`` per chunk.

    Returns:
        List of token groups.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    if not tokens:
        return []

    result: list[list[str]] = []
    chunk: list[str] = []

    for token in tokens:
        if chunk:
            joined_len = ops.length(ops.join(chunk + [token]))
            if joined_len > max_length:
                result.append(chunk)
                chunk = []

        chunk.append(token)

    if chunk:
        result.append(chunk)

    return result


