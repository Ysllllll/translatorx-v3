"""Tests for NerPuncRestorer."""

from __future__ import annotations

import pytest

from adapters.preprocess.availability import punc_model_is_available

pytestmark = pytest.mark.skipif(not punc_model_is_available(), reason="deepmultilingualpunctuation not installed")


class TestNerPuncRestorer:
    def test_singleton(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        a = NerPuncRestorer.get_instance()
        b = NerPuncRestorer.get_instance()
        assert a is b

    def test_basic_restore(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer(["hello world this is a test"])
        assert len(result) == 1
        assert len(result[0]) == 1
        restored = result[0][0]
        # Should contain at least one punctuation mark
        assert any(c in restored for c in ".,!?;:")

    def test_empty_input(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer([""])
        assert result == [[""]]

    def test_batch_processing(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        texts = ["hello world", "this is great", "how are you"]
        result = restorer(texts)
        assert len(result) == 3
        for r in result:
            assert len(r) == 1  # 1:1 mapping

    def test_omit_punct_stripping(self) -> None:
        """The _OMIT_PUNCT_RE should strip unusual punctuation."""
        from adapters.preprocess.ner_punc import _OMIT_PUNCT_RE

        assert _OMIT_PUNCT_RE.sub("", "hello@world") == "helloworld"
        assert _OMIT_PUNCT_RE.sub("", "test{value}") == "testvalue"
        # Normal punctuation should be preserved
        assert _OMIT_PUNCT_RE.sub("", "Hello, world.") == "Hello, world."

    def test_applyfn_conformance(self) -> None:
        """NerPuncRestorer should conform to the ApplyFn protocol."""
        from adapters.preprocess import NerPuncRestorer
        from ports.apply_fn import ApplyFn

        restorer = NerPuncRestorer.get_instance()
        # Structural check: callable with correct signature
        result = restorer(["test"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)


class TestProtectDottedWords:
    """Unit tests for _protect_dotted_words (no model needed)."""

    def test_node_js_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _protect_dotted_words

        result = _protect_dotted_words("you have Node.js version eighteen", "you have Node. Js version eighteen,")
        assert "Node.js" in result

    def test_eg_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _protect_dotted_words

        result = _protect_dotted_words("use e.g. something here", "use e. G. Something here,")
        assert "e.g." in result

    def test_no_dotted_words_passthrough(self) -> None:
        from adapters.preprocess.ner_punc import _protect_dotted_words

        text = "Hello, world."
        assert _protect_dotted_words("Hello world", text) == text

    def test_multiple_dotted_words(self) -> None:
        from adapters.preprocess.ner_punc import _protect_dotted_words

        result = _protect_dotted_words("use Node.js and Vue.js here", "use Node. Js and Vue. Js here,")
        assert "Node.js" in result
        assert "Vue.js" in result


class TestPreserveTrailingPunc:
    """Unit tests for _preserve_trailing_punc (no model needed)."""

    def test_period_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _preserve_trailing_punc

        result = _preserve_trailing_punc("hello world.", "Hello, world")
        assert result.endswith(".")

    def test_ellipsis_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _preserve_trailing_punc

        result = _preserve_trailing_punc("hello world...", "Hello, world.")
        assert result.endswith("...")

    def test_exclamation_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _preserve_trailing_punc

        result = _preserve_trailing_punc("hello world!", "Hello, world.")
        assert result.endswith("!")

    def test_no_trailing_punc_passthrough(self) -> None:
        from adapters.preprocess.ner_punc import _preserve_trailing_punc

        result = _preserve_trailing_punc("hello world", "Hello, world.")
        # No trailing punc in source, so restored text is unchanged
        assert result == "Hello, world."

    def test_question_mark_preserved(self) -> None:
        from adapters.preprocess.ner_punc import _preserve_trailing_punc

        result = _preserve_trailing_punc("how are you?", "How are you.")
        assert result.endswith("?")


class TestRestoreOneIntegration:
    """Integration tests that exercise _restore_one with the actual model."""

    def test_dotted_word_not_corrupted(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer(["you have Node.js version eighteen or above installed on your machine"])
        restored = result[0][0]
        assert "Node.js" in restored

    def test_trailing_ellipsis_preserved(self) -> None:
        from adapters.preprocess import NerPuncRestorer

        restorer = NerPuncRestorer.get_instance()
        result = restorer(["and that is how we do it..."])
        restored = result[0][0]
        assert restored.rstrip().endswith("...")
