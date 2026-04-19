"""Tests for NerPuncRestorer."""

from __future__ import annotations

import pytest

from preprocess._availability import punc_model_is_available

pytestmark = pytest.mark.skipif(
    not punc_model_is_available(),
    reason="deepmultilingualpunctuation not installed",
)


class TestNerPuncRestorer:
    def test_singleton(self) -> None:
        from preprocess import NerPuncRestorer

        a = NerPuncRestorer.get_instance()
        b = NerPuncRestorer.get_instance()
        assert a is b

    def test_basic_restore(self) -> None:
        from preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer(["hello world this is a test"])
        assert len(result) == 1
        assert len(result[0]) == 1
        restored = result[0][0]
        # Should contain at least one punctuation mark
        assert any(c in restored for c in ".,!?;:")

    def test_empty_input(self) -> None:
        from preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer([""])
        assert result == [[""]]

    def test_batch_processing(self) -> None:
        from preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        texts = ["hello world", "this is great", "how are you"]
        result = restorer(texts)
        assert len(result) == 3
        for r in result:
            assert len(r) == 1  # 1:1 mapping

    def test_omit_punct_stripping(self) -> None:
        """The _OMIT_PUNCT_RE should strip unusual punctuation."""
        from preprocess._ner_punc import _OMIT_PUNCT_RE

        assert _OMIT_PUNCT_RE.sub("", "hello@world") == "helloworld"
        assert _OMIT_PUNCT_RE.sub("", "test{value}") == "testvalue"
        # Normal punctuation should be preserved
        assert _OMIT_PUNCT_RE.sub("", "Hello, world.") == "Hello, world."

    def test_applyfn_conformance(self) -> None:
        """NerPuncRestorer should conform to the ApplyFn protocol."""
        from preprocess import NerPuncRestorer
        from preprocess._protocol import ApplyFn

        restorer = NerPuncRestorer.get_instance()
        # Structural check: callable with correct signature
        result = restorer(["test"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)
