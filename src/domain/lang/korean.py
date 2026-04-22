"""Korean text operations using Kiwi."""

from __future__ import annotations

import unicodedata

from ._core._cjk_common import (
    _BaseCjkOps,
    _attach_tokens,
    _cjk_join_tokens,
    _iter_script_segments,
    _protect_latin_fragments,
    _restore_protected_tokens,
)
from ._core._base_ops import normalize_mode, _VALID_MODES

_SPACE_MARKER = " "


class KoreanOps(_BaseCjkOps):
    def __init__(self):
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()

    @property
    def sentence_terminators(self) -> frozenset[str]:
        return frozenset({".", "。", "!", "?"})

    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({",", "；"})

    @property
    def strip_spaces(self) -> bool:
        return False

    def _word_tokenize(self, text: str) -> list[str]:
        # Called by base class when script segmentation hands us a CJK
        # segment from a single eojeol; no internal spaces expected.
        return self._tokenize_eojeol(text)

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")

        protected, mapping = _protect_latin_fragments(text)
        eojeols = protected.split(_SPACE_MARKER)
        all_tokens: list[str] = []
        for i, eojeol in enumerate(eojeols):
            if i > 0:
                all_tokens.append(_SPACE_MARKER)
            raw: list[str] = []
            for kind, seg in _iter_script_segments(eojeol):
                if kind == "cjk":
                    if mode == "character":
                        raw.extend(list(seg))
                    else:
                        raw.extend(self._tokenize_eojeol(seg))
                else:
                    raw.append(seg)
            raw = _restore_protected_tokens(raw, mapping)
            if attach_punctuation:
                tokens = _attach_tokens(raw, multi_dot_attaches=(mode == "character"))
            else:
                tokens = raw
            all_tokens.extend(tokens)
        return all_tokens

    def _tokenize_eojeol(self, eojeol: str) -> list[str]:
        return [token.form for token in self._kiwi.tokenize(eojeol)]

    def join(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        groups: list[str] = []
        current: list[str] = []
        for tok in tokens:
            if tok == _SPACE_MARKER:
                if current:
                    groups.append(_cjk_join_tokens(current))
                    current = []
            else:
                current.append(tok)
        if current:
            groups.append(_cjk_join_tokens(current))

        # NFC-normalize: kiwi may return NFD jamo sequences
        return unicodedata.normalize("NFC", _SPACE_MARKER.join(groups))
