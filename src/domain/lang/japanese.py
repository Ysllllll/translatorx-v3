"""Japanese text operations using MeCab."""

from __future__ import annotations

from ._core._cjk_common import _BaseCjkOps


_CONNECTIVES: frozenset[str] = frozenset(
    {
        "けれども",
        "しかし",
        "だから",
        "それで",
        "なぜなら",
        "もし",
        "ので",
        "のに",
        "ため",
        "しかしながら",
        "ところが",
        "ただし",
        "そして",
        "また",
        "または",
    }
)


class JapaneseOps(_BaseCjkOps):
    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({"、", "；", ","})

    @property
    def connectives(self) -> frozenset[str]:
        return _CONNECTIVES

    def _word_tokenize(self, text: str) -> list[str]:
        import MeCab

        tagger = MeCab.Tagger()
        node = tagger.parseToNode(text)
        tokens: list[str] = []
        while node:
            if node.surface:
                tokens.append(node.surface)
            node = node.next
        return tokens
