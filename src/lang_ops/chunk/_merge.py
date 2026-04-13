"""Length-based chunk merging — the inverse of splitting."""

from __future__ import annotations

from typing import Protocol


class _HasJoinLength(Protocol):
    """Protocol for the lang_ops methods merge needs."""

    def join(self, tokens: list[str]) -> str: ...
    def length(self, text: str, **kwargs: int) -> int: ...

    @property
    def is_cjk(self) -> bool: ...


def merge_token_groups(
    groups: list[list[str]],
    ops: _HasJoinLength,
    max_length: int,
) -> list[list[str]]:
    """Greedily merge adjacent token groups whose joined length ≤ *max_length*.

    Iterates left-to-right.  Each group is appended to the current
    accumulator; if the combined ``ops.length(ops.join(...))`` would
    exceed *max_length*, the accumulator is flushed first.

    A single group exceeding *max_length* is emitted as-is (minimum
    unit = one group, never break within a group).

    Args:
        groups: Token groups (e.g. from ``split_tokens_by_length``).
        ops: Language ops providing ``join()`` and ``length()``.
        max_length: Upper bound on ``ops.length()`` per merged chunk.

    Returns:
        List of merged token groups.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    if not groups:
        return []

    result: list[list[str]] = []
    current: list[str] = []

    for group in groups:
        if current:
            candidate = current + group
            if ops.length(ops.join(candidate)) > max_length:
                result.append(current)
                current = list(group)
            else:
                current = candidate
        else:
            current = list(group)

    if current:
        result.append(current)

    return result


def merge_chunks_by_length(
    chunks: list[str],
    ops: _HasJoinLength,
    max_length: int,
) -> list[str]:
    """Greedily merge adjacent text chunks whose combined length ≤ *max_length*.

    Uses a language-appropriate separator (empty for CJK, space otherwise).
    When a chunk already starts with the separator, no extra separator is
    added (avoids double spaces).

    Args:
        chunks: Text fragments to merge.
        ops: Language ops providing ``length()`` and ``is_cjk``.
        max_length: Upper bound on ``ops.length()`` per merged chunk.

    Returns:
        List of merged text chunks.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    if not chunks:
        return []

    sep = "" if ops.is_cjk else " "
    result: list[str] = []
    current_parts: list[str] = []
    current_text = ""

    for chunk in chunks:
        if current_parts:
            if sep and chunk.startswith(sep):
                candidate = current_text + chunk
            else:
                candidate = current_text + sep + chunk
            if ops.length(candidate) > max_length:
                result.append(current_text)
                current_parts = [chunk]
                current_text = chunk
            else:
                current_parts.append(chunk)
                current_text = candidate
        else:
            current_parts.append(chunk)
            current_text = chunk

    if current_parts:
        result.append(current_text)

    return result
