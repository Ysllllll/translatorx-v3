"""Length-based text splitter."""

from __future__ import annotations

from typing import Protocol


class _HasSplitJoin(Protocol):
    """Protocol for the text_ops methods we need."""

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]: ...
    def join(self, tokens: list[str]) -> str: ...
    def length(self, text: str, **kwargs: int) -> int: ...


def split_by_length(
    text: str,
    ops: _HasSplitJoin,
    max_length: int,
    unit: str = "character",
) -> list[str]:
    """Split text into chunks that don't exceed max_length."""
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
) -> list[str]:
    """Split by character count, breaking at word/token boundaries."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[str] = []
    chunk_tokens: list[str] = []
    chunk_len = 0

    for token in tokens:
        token_len = ops.length(token)

        if chunk_tokens and chunk_len + token_len > max_length:
            result.append(ops.join(chunk_tokens))
            chunk_tokens = []
            chunk_len = 0

        if token_len > max_length:
            if chunk_tokens:
                result.append(ops.join(chunk_tokens))
                chunk_tokens = []
                chunk_len = 0
            i = 0
            while i < len(token):
                result.append(token[i : i + max_length])
                i += max_length
        else:
            chunk_tokens.append(token)
            chunk_len += token_len

    if chunk_tokens:
        result.append(ops.join(chunk_tokens))

    return result


def _split_by_word_count(
    text: str,
    ops: _HasSplitJoin,
    max_length: int,
) -> list[str]:
    """Split by word/token count."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[str] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + max_length]
        result.append(ops.join(chunk))
        i += max_length

    return result
