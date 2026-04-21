"""Shared base class for all language operations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ._chars import STRIP_PUNCT, decompose_token


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

    def transfer_punc(self, text_a: str, text_b: str) -> str:
        """Transfer punctuation from *text_b* onto words of *text_a*.

        Token-level punctuation transfer: takes the word content from
        *text_a* and wraps it with the leading/trailing punctuation
        found in *text_b*.

        Not to be confused with LLM-based punctuation *restoration*
        (see :class:`~preprocess.LlmPuncRestorer`).
        """
        tokens_a = self.split(text_a)
        tokens_b = self.split(text_b)
        if len(tokens_a) != len(tokens_b):
            raise ValueError(f"Token count mismatch: text_a has {len(tokens_a)}, text_b has {len(tokens_b)}")
        result: list[str] = []
        for ta, tb in zip(tokens_a, tokens_b):
            _, content_a, _ = decompose_token(ta)
            lead_b, _, trail_b = decompose_token(tb)
            result.append(lead_b + content_a + trail_b)
        return self.join(result)

    # -- Segment-level shortcuts --------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).sentences().result()

    def split_clauses(self, text: str) -> list[str]:
        """Split text into clauses (sentence boundaries included, token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).clauses().result()

    def split_by_length(self, text: str, max_len: int) -> list[str]:
        """Split text into chunks whose length ≤ *max_len* (token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).split(max_len).result()

    def merge_by_length(self, chunks: list[str], max_len: int) -> list[str]:
        """Greedily merge adjacent chunks whose combined length ≤ *max_len*."""
        from domain.lang.chunk._merge import merge_chunks_by_length

        return merge_chunks_by_length(chunks, self, max_len)

    def chunk(self, text: str) -> "TextPipeline":
        """Create a TextPipeline for chainable text structuring."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self)
