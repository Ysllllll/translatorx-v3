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
    unit: str = "character",
) -> list[Span]:
    """Split text into chunks that don't exceed max_length.

    Returns Span objects with start=-1, end=-1 because tokenize+join
    may alter whitespace, making character offsets unreliable.
    """
    if not text:
        return []

    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if unit not in ("character", "word"):
        raise ValueError(f"unit must be 'character' or 'word', got {unit!r}")

    if unit == "word":
        return _split_by_word_count(text, ops, max_length)
    return _split_by_char_count(text, ops, max_length)


def _split_by_char_count(
    text: str,
    ops: _HasSplitJoin,
    max_length: int,
) -> list[Span]:
    """Split by character count, breaking at word/token boundaries."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[Span] = []
    chunk_tokens: list[str] = []
    chunk_len = 0

    for token in tokens:
        token_len = ops.length(token)

        if chunk_tokens and chunk_len + token_len > max_length:
            result.append(Span(ops.join(chunk_tokens), -1, -1))
            chunk_tokens = []
            chunk_len = 0

        if token_len > max_length:
            if chunk_tokens:
                result.append(Span(ops.join(chunk_tokens), -1, -1))
                chunk_tokens = []
                chunk_len = 0
            i = 0
            while i < len(token):
                result.append(Span(token[i : i + max_length], -1, -1))
                i += max_length
        else:
            chunk_tokens.append(token)
            chunk_len += token_len

    if chunk_tokens:
        result.append(Span(ops.join(chunk_tokens), -1, -1))

    return result


def _split_by_word_count(
    text: str,
    ops: _HasSplitJoin,
    max_length: int,
) -> list[Span]:
    """Split by word/token count."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[Span] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + max_length]
        result.append(Span(ops.join(chunk), -1, -1))
        i += max_length

    return result
