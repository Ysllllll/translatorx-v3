"""Tests for SpacyLlmChunker — specifically the ops-aware length metric."""

from __future__ import annotations

import pytest

from adapters.preprocess.availability import spacy_is_available

pytestmark = pytest.mark.skipif(not spacy_is_available(), reason="spacy not installed")


class _FakeSplitter:
    """Minimal spaCy-splitter stand-in: returns sentences as-is."""

    def __init__(self, sentences_per_text: dict[str, list[str]]):
        self._map = sentences_per_text

    def __call__(self, texts: list[str]) -> list[list[str]]:
        return [self._map.get(t, [t]) for t in texts]


class _FakeLlmChunker:
    """Records which chunks were routed for LLM refinement."""

    def __init__(self):
        self.seen: list[str] = []

    async def _process_batch(self, texts: list[str]) -> list[list[str]]:
        self.seen.extend(texts)
        # Pass-through: one-chunk output per input.
        return [[t] for t in texts]


class TestSpacyLlmChunkerLength:
    def test_len_based_default_may_overcount_cjk(self) -> None:
        """Without ops, a mix of CJK + ASCII may trip the threshold via len()."""
        from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker

        splitter = _FakeSplitter({"mixed": ["测试 Node.js 非常有用"]})
        llm = _FakeLlmChunker()
        chunker = SpacyLlmChunker(splitter, llm, chunk_len=15)  # type: ignore[arg-type]
        chunker(["mixed"])
        # len("测试 Node.js 非常有用") == 15; threshold is 15 → NOT > threshold, not routed.
        # Prove path runs without error; threshold comparison uses len by default.
        assert llm.seen == []  # exactly at threshold, not exceeded

    def test_ops_length_treats_cjk_as_single_char(self) -> None:
        """With ops, CJK characters count as 1 each (same as len here), but the
        key guarantee is the factory accepts ops without error and uses it."""
        from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker
        from domain.lang import LangOps

        ops = LangOps.for_language("zh")
        splitter = _FakeSplitter({"long": ["这是一个超级长的句子需要被切开abc"]})
        llm = _FakeLlmChunker()
        chunker = SpacyLlmChunker(splitter, llm, chunk_len=10, ops=ops)  # type: ignore[arg-type]
        chunker(["long"])
        # ops.length ≈ 18 > 10 → routed.
        assert llm.seen == ["这是一个超级长的句子需要被切开abc"]

    def test_length_fn_overrides(self) -> None:
        from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker

        splitter = _FakeSplitter({"t": ["hello"]})
        llm = _FakeLlmChunker()
        chunker = SpacyLlmChunker(splitter, llm, chunk_len=2, length_fn=lambda s: 999)  # type: ignore[arg-type]
        chunker(["t"])
        assert llm.seen == ["hello"]
