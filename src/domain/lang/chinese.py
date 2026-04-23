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
        "可是",
        "不仅",
        "不但",
        "而是",
        "或者",
        "要是",
        "假如",
        "倘若",
        "既然",
        "既",
        "况且",
        "再者",
        "反之",
        "否则",
        "不然",
        "除了",
        "与此同时",
        "同时",
        "于是",
        "结果",
        "最终",
        "况且",
        "何况",
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
