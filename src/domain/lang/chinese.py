"""Chinese text operations using jieba."""

from __future__ import annotations

from ._core._cjk_common import _BaseCjkOps


_CONNECTIVES: frozenset[str] = frozenset(
    {
        "因为",
        "所以",
        "但是",
        "然而",
        "虽然",
        "如果",
        "即使",
        "尽管",
        "不过",
        "否则",
        "除非",
        "而且",
        "并且",
        "当",
        "一旦",
        "只要",
        "只有",
        "由于",
        "因此",
        "因而",
        "然后",
        "接着",
        "此外",
        "另外",
    }
)


class ChineseOps(_BaseCjkOps):
    @property
    def connectives(self) -> frozenset[str]:
        return _CONNECTIVES

    def _word_tokenize(self, text: str) -> list[str]:
        import jieba

        tokens: list[str] = []
        for tok in jieba.lcut(text):
            if tok.isspace():
                continue
            tokens.append(tok)
        return tokens
