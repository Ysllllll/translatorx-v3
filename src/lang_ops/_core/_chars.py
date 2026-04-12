"""Character classification utilities for multilingual text processing."""

from __future__ import annotations


def is_cjk_ideograph(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return (
        0xAC00 <= cp <= 0xD7AF
        or 0x1100 <= cp <= 0x11FF
        or 0x3130 <= cp <= 0x318F
        or 0xA960 <= cp <= 0xA97F
        or 0xD7B0 <= cp <= 0xD7FF
    )


def is_hiragana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3040 <= cp <= 0x309F


def is_katakana(ch: str) -> bool:
    cp = ord(ch)
    return 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF


def is_east_asian(ch: str) -> bool:
    return is_cjk_ideograph(ch) or is_hangul(ch) or is_hiragana(ch) or is_katakana(ch)


# Punctuation sets for CJK attachment
TRAILING_PUNCT = set(",.!?:;，。！？：；、")
CLOSING_PUNCT = set(")]}）》”’")  # ）》”’
OPENING_PUNCT = set("([{（《“‘")   # （《“‘

ATTACH_TO_PREV = TRAILING_PUNCT | CLOSING_PUNCT

# Comprehensive punctuation for strip_punc operations
STRIP_PUNCT = "".join(sorted(
    TRAILING_PUNCT | CLOSING_PUNCT | OPENING_PUNCT
    | set("¡¿<>\"'—–‐…·「」『』【】")
))


def is_opening_punct_char(ch: str) -> bool:
    return ch in OPENING_PUNCT


def is_attach_to_prev_char(ch: str) -> bool:
    return ch in ATTACH_TO_PREV


def cjk_needs_space(prev_last: str, curr_first: str) -> bool:
    if not prev_last.isalnum() or not curr_first.isalnum():
        return False
    return not (is_east_asian(prev_last) and is_east_asian(curr_first))


# Characters treated as content (not punctuation) in CJK mode
CONTENT_LIKE_CHARS = {"…", "・"}


def decompose_token(token: str) -> tuple[str, str, str]:
    """Decompose a token into (leading_punct, content, trailing_punct).

    Uses STRIP_PUNCT to identify punctuation characters.
    """
    i = 0
    while i < len(token) and token[i] in STRIP_PUNCT:
        i += 1
    j = len(token)
    while j > i and token[j - 1] in STRIP_PUNCT:
        j -= 1
    return (token[:i], token[i:j], token[j:])
