"""Japanese text operations using MeCab."""

from __future__ import annotations

from ._core._cjk_common import _BaseCjkOps


class JapaneseOps(_BaseCjkOps):
    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({"、", "；", ","})

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
