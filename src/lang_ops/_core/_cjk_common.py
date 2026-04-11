"""Shared CJK mechanism base class and helpers."""

from __future__ import annotations

from ._chars import (
    is_east_asian,
    is_cjk_ideograph,
    is_hangul,
    is_hiragana,
    is_katakana,
    is_opening_punct_char,
    is_attach_to_prev_char,
    cjk_needs_space,
    CONTENT_LIKE_CHARS,
    STRIP_PUNCT,
    decompose_token,
)
from ._mode import normalize_mode, _VALID_MODES


def _is_cjk_or_kana(ch: str) -> bool:
    return is_cjk_ideograph(ch) or is_hiragana(ch) or is_katakana(ch) or is_hangul(ch)


def _is_full_width_char(ch: str) -> bool:
    if is_east_asian(ch):
        return True
    cp = ord(ch)
    if 0x3000 <= cp <= 0x303F:
        return True
    if 0xFF01 <= cp <= 0xFF5E:
        return True
    if ch in CONTENT_LIKE_CHARS:
        return True
    return False


def _cjk_length(text: str, cjk_width: int = 1) -> int:
    total = 0
    latin_count = 0

    def flush() -> None:
        nonlocal total, latin_count
        if latin_count > 0:
            total += (latin_count + cjk_width - 1) // cjk_width
            latin_count = 0

    for ch in text:
        if _is_full_width_char(ch):
            flush()
            total += 1
        elif ch.isspace():
            flush()
        else:
            latin_count += 1

    flush()
    return total


def _parse_characters(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in CONTENT_LIKE_CHARS:
            tokens.append(ch)
            i += 1
        elif _is_cjk_or_kana(ch):
            tokens.append(ch)
            i += 1
        elif ch.isalnum():
            j = i
            while j < n and text[j].isalnum() and not _is_cjk_or_kana(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
        elif ch == ".":
            j = i
            while j < n and text[j] == ".":
                j += 1
            tokens.append(text[i:j])
            i = j
        elif ch.isspace():
            i += 1
        else:
            tokens.append(ch)
            i += 1

    return tokens


def _is_opening_token(token: str) -> bool:
    return len(token) == 1 and is_opening_punct_char(token)


def _is_trailing_or_closing_token(token: str, multi_dot_attaches: bool = True) -> bool:
    if len(token) == 1 and is_attach_to_prev_char(token):
        return True
    if multi_dot_attaches and len(token) > 1 and all(c == "." for c in token):
        return True
    return False


def _is_content_token(token: str) -> bool:
    return not _is_opening_token(token) and not _is_trailing_or_closing_token(token)


def _attach_tokens(raw_tokens: list[str], multi_dot_attaches: bool = True) -> list[str]:
    result: list[str] = []
    current = ""
    pending_open = ""

    for token in raw_tokens:
        if _is_opening_token(token):
            if current:
                result.append(current)
                current = ""
            pending_open += token
        elif _is_trailing_or_closing_token(token, multi_dot_attaches):
            if current:
                current += token
            elif pending_open:
                pending_open += token
            else:
                if result:
                    result[-1] += token
                else:
                    current = token
        else:
            if current:
                result.append(current)
            current = pending_open + token
            pending_open = ""

    if current:
        result.append(current)
    if pending_open:
        result.append(pending_open)

    return result


def _cjk_join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    parts = [tokens[0]]
    for token in tokens[1:]:
        prev_last = parts[-1][-1]
        curr_first = token[0]
        if cjk_needs_space(prev_last, curr_first):
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


def _determine_kind(token: str, mode: str) -> str:
    if not any(c.isalnum() for c in token):
        return "punctuation"
    if mode == "character" and len(token) == 1 and is_east_asian(token):
        return "character"
    return "word"


class _BaseCjkOps:
    """Base class for CJK text operations.

    Subclasses must implement ``_word_tokenize``.
    Override ``split`` and ``join`` if the language requires
    special handling (e.g. Korean eojeol tracking).
    """

    @property
    def sentence_terminators(self) -> frozenset[str]:
        return frozenset({"。", "！", "？"})

    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({"，", "、", "；", "："})

    @property
    def abbreviations(self) -> frozenset[str]:
        return frozenset()

    @property
    def is_cjk(self) -> bool:
        return True

    def _word_tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")

        if mode == "character":
            raw = _parse_characters(text)
        else:
            raw = self._word_tokenize(text)

        if attach_punctuation:
            return _attach_tokens(raw, multi_dot_attaches=(mode == "character"))
        return raw

    def join(self, tokens: list[str]) -> str:
        return _cjk_join_tokens(tokens)

    def length(self, text: str, cjk_width: int = 1) -> int:
        return _cjk_length(text, cjk_width)

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

    def normalize(self, text: str) -> str:
        return text

    # -- Segment-level shortcuts ----------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        from lang_ops.splitter._sentence import split_sentences as _split
        return _split(text, self.sentence_terminators, self.abbreviations, is_cjk=self.is_cjk)

    def split_clauses(self, text: str) -> list[str]:
        """Split text into clauses."""
        from lang_ops.splitter._clause import split_clauses as _split
        return _split(text, self.clause_separators)

    def split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        from lang_ops.splitter._paragraph import split_paragraphs as _split
        return _split(text)

    def chunk(self, text: str) -> "ChunkPipeline":
        """Create a ChunkPipeline for chainable splitting."""
        from lang_ops.splitter._pipeline import ChunkPipeline
        return ChunkPipeline(text, ops=self)
