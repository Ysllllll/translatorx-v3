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
        "だが",
        "でも",
        "けれど",
        "それでも",
        "それなら",
        "それに",
        "さらに",
        "その上",
        "そのため",
        "その結果",
        "したがって",
        "ゆえに",
        "および",
        "かつ",
        "もしくは",
        "あるいは",
        "なお",
        "ちなみに",
        "ところで",
        "つまり",
        "すなわち",
        "たとえば",
        "一方",
        "他方",
        "一方で",
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
