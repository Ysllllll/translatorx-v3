"""Chinese text operations using jieba."""

from __future__ import annotations

from ._core._cjk_common import _BaseCjkOps


class ChineseOps(_BaseCjkOps):

    def _word_tokenize(self, text: str) -> list[str]:
        import jieba
        tokens: list[str] = []
        for tok in jieba.lcut(text):
            if tok.isspace():
                continue
            tokens.append(tok)
        return tokens
