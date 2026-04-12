"""Korean text operations using Kiwi."""

from __future__ import annotations

from ._core._cjk_common import (
    _BaseCjkOps,
    _parse_characters,
    _attach_tokens,
    _cjk_join_tokens,
)
from ._core._base_ops import normalize_mode, _VALID_MODES

_SPACE_MARKER = " "


class KoreanOps(_BaseCjkOps):

    @property
    def sentence_terminators(self) -> frozenset[str]:
        return frozenset({".", "。", "!", "?"})

    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({",", "；"})

    def _word_tokenize(self, text: str) -> list[str]:
        eojeols = text.split(_SPACE_MARKER)
        tokens: list[str] = []
        for i, eojeol in enumerate(eojeols):
            if i > 0:
                tokens.append(_SPACE_MARKER)
            tokens.extend(self._tokenize_eojeol(eojeol))
        return tokens

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")

        eojeols = text.split(_SPACE_MARKER)
        all_tokens: list[str] = []
        for i, eojeol in enumerate(eojeols):
            if i > 0:
                all_tokens.append(_SPACE_MARKER)
            if mode == "character":
                raw = _parse_characters(eojeol)
            else:
                raw = self._tokenize_eojeol(eojeol)

            if attach_punctuation:
                tokens = _attach_tokens(raw, multi_dot_attaches=(mode == "character"))
            else:
                tokens = raw
            all_tokens.extend(tokens)
        return all_tokens

    def _tokenize_eojeol(self, eojeol: str) -> list[str]:
        from kiwipiepy import Kiwi
        if not hasattr(self, "_kiwi"):
            self._kiwi = Kiwi()
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

        return _SPACE_MARKER.join(groups)