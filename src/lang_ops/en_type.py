"""EN-type ops for space-delimited languages (en, ru, es, fr, de, pt, vi)."""

from __future__ import annotations

import re

from ._core._mode import normalize_mode, _VALID_MODES
from ._core._chars import STRIP_PUNCT, decompose_token


class EnTypeOps:

    def __init__(self, language: str = "en") -> None:
        self._language = language

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        return text.split()

    def join(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        parts: list[tuple[bool, str]] = []  # (space_before, text)
        open_double = False
        skip_next = False

        for i, token in enumerate(tokens):
            is_first = i == 0
            is_single = len(token) == 1

            space = not is_first

            if skip_next:
                space = False
                skip_next = False

            if is_single and token == '"':
                if open_double:
                    space = False
                    if (parts and parts[-1][0]
                            and len(parts[-1][1]) == 1
                            and not parts[-1][1].isalnum()):
                        parts[-1] = (False, parts[-1][1])
                    open_double = False
                else:
                    skip_next = True
                    open_double = True
            elif is_single and token == "'":
                space = False
                skip_next = True
            elif is_single and token in (",", "."):
                space = False
            elif is_single and token in (")", "]", "}"):
                space = False
            elif is_single and token in ("(", "[", "{", "¡", "¿"):
                skip_next = True

            parts.append((space, token))

        result: list[str] = []
        for sp, text in parts:
            if sp:
                result.append(" ")
            result.append(text)
        return "".join(result)

    def length(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> int:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        return len(text)

    def plength(self, text: str, font_path: str, font_size: int) -> int:
        from PIL import ImageFont
        left, _, right, _ = ImageFont.truetype(font_path, font_size).getbbox(text)
        return max(0, int(right - left))

    def normalize(self, text: str) -> str:
        if self._language == "fr":
            text = re.sub(r' +([.,])', r'\1', text)
            text = re.sub(r' +([!?;:])', r' \1', text)
            text = re.sub(r'([^\s!?;:,.])([!?;])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z\u00C0-\u024F])(:)', r'\1 \2', text)
        else:
            text = re.sub(r' +([.,!?);:}\]])', r'\1', text)
            text = re.sub(r'([(\[{¡¿]) +', r'\1', text)
        return text

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