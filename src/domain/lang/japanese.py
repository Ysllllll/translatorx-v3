"""Japanese text operations using MeCab."""

from __future__ import annotations

from threading import Lock
from typing import Any

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
        tagger = _get_tagger()
        node = tagger.parseToNode(text)
        tokens: list[str] = []
        while node:
            if node.surface:
                tokens.append(node.surface)
            node = node.next
        return tokens


# C5 — MeCab.Tagger() construction is non-trivial (loads dictionaries
# from disk). Re-using a single Tagger across calls cuts per-call cost
# from O(dict-load) to O(parse). The tagger is documented thread-safe
# for parseToNode; we still gate behind a lock for first-time creation
# to avoid duplicate loads under concurrent imports.
_TAGGER: Any = None
_TAGGER_LOCK = Lock()


def _get_tagger() -> Any:
    global _TAGGER
    if _TAGGER is None:
        with _TAGGER_LOCK:
            if _TAGGER is None:
                import MeCab

                _TAGGER = MeCab.Tagger()
    return _TAGGER
