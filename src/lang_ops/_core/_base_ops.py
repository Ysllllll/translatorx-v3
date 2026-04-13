"""Shared base class for all language operations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ._chars import STRIP_PUNCT, decompose_token
from ._types import Span


# Mode shorthand: "c" = "character", "w" = "word"
_MODE_SHORTHAND = {"c": "character", "w": "word"}
_VALID_MODES = {"character", "word"}


def normalize_mode(mode: str) -> str:
    """Normalize mode shorthand to full name."""
    return _MODE_SHORTHAND.get(mode, mode)


class _BaseOps(ABC):
    """Abstract base for all language-specific text operations.

    Subclasses must implement:
        - split, join, length, normalize
        - sentence_terminators, clause_separators, abbreviations, is_cjk
    """

    # -- Abstract properties (override in subclass) -------------------------

    @property
    @abstractmethod
    def sentence_terminators(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def clause_separators(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def abbreviations(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def is_cjk(self) -> bool: ...

    @property
    def strip_spaces(self) -> bool:
        """Whether to strip inter-sentence/clause leading spaces. True for CJK except Korean."""
        return self.is_cjk

    # -- Abstract methods (override in subclass) ----------------------------

    @abstractmethod
    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]: ...

    @abstractmethod
    def join(self, tokens: list[str]) -> str: ...

    @abstractmethod
    def length(self, text: str, **kwargs: int) -> int: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    # -- Shared concrete methods --------------------------------------------

    def plength(self, text: str, font_path: str, font_size: int) -> int:
        from PIL import ImageFont
        left, _, right, _ = ImageFont.truetype(font_path, font_size).getbbox(text)
        return max(0, int(right - left))

    def strip(self, text: str, chars: str | None = None) -> str:
        return text.strip(chars)

    def lstrip(self, text: str, chars: str | None = None) -> str:
        return text.lstrip(chars)

    def rstrip(self, text: str, chars: str | None = None) -> str:
        return text.rstrip(chars)

    def strip_punc(self, text: str) -> str:
        return text.strip(STRIP_PUNCT)

    def lstrip_punc(self, text: str) -> str:
        return text.lstrip(STRIP_PUNCT)

    def rstrip_punc(self, text: str) -> str:
        return text.rstrip(STRIP_PUNCT)

    def restore_punc(self, text_a: str, text_b: str) -> str:
        tokens_a = self.split(text_a)
        tokens_b = self.split(text_b)
        if len(tokens_a) != len(tokens_b):
            raise ValueError(
                f"Token count mismatch: text_a has {len(tokens_a)}, "
                f"text_b has {len(tokens_b)}"
            )
        result: list[str] = []
        for ta, tb in zip(tokens_a, tokens_b):
            _, content_a, _ = decompose_token(ta)
            lead_b, _, trail_b = decompose_token(tb)
            result.append(lead_b + content_a + trail_b)
        return self.join(result)

    # -- Segment-level shortcuts --------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (token-based)."""
        from lang_ops.splitter._boundary import find_boundaries, split_tokens_by_boundaries
        tokens = self.split(text)
        if not tokens:
            return []
        boundaries = find_boundaries(tokens, self.sentence_terminators, self.abbreviations)
        groups = split_tokens_by_boundaries(tokens, boundaries)
        return [self.join(g) for g in groups]

    def split_clauses(self, text: str) -> list[str]:
        """Split text into clauses (sentence boundaries included, token-based)."""
        from lang_ops.splitter._boundary import find_boundaries, split_tokens_by_boundaries
        tokens = self.split(text)
        if not tokens:
            return []
        boundaries = find_boundaries(
            tokens, self.sentence_terminators, self.abbreviations, self.clause_separators,
        )
        groups = split_tokens_by_boundaries(tokens, boundaries)
        return [self.join(g) for g in groups]

    def split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        from lang_ops.splitter._paragraph import split_paragraphs as _split
        return Span.to_texts(_split(text))

    def split_by_length(self, text: str, max_length: int) -> list[str]:
        """Split text into chunks whose length ≤ *max_length* (token-based)."""
        from lang_ops.splitter._length import split_tokens_by_length
        tokens = self.split(text)
        if not tokens:
            return []
        groups = split_tokens_by_length(tokens, self, max_length)
        return [self.join(g) for g in groups]

    def chunk(self, text: str) -> "ChunkPipeline":
        """Create a ChunkPipeline for chainable splitting."""
        from lang_ops.splitter._pipeline import ChunkPipeline
        return ChunkPipeline(text, ops=self)
