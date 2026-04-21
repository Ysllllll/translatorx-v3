"""Tests for SpacySplitter."""

from __future__ import annotations

import pytest

from adapters.preprocess.availability import spacy_is_available

pytestmark = pytest.mark.skipif(
    not spacy_is_available(),
    reason="spacy not installed",
)


class TestSpacySplitter:
    def test_singleton_per_model(self) -> None:
        from adapters.preprocess import SpacySplitter

        a = SpacySplitter.get_instance()
        b = SpacySplitter.get_instance()
        assert a is b

    def test_basic_split(self) -> None:
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["Hello world. This is a test. How are you?"])
        assert len(result) == 1
        sentences = result[0]
        assert len(sentences) >= 2  # At least 2 sentences

    def test_empty_input(self) -> None:
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter([""])
        assert result == [[""]]

    def test_single_sentence(self) -> None:
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["Hello world."])
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_batch_processing(self) -> None:
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        texts = [
            "First text. Second sentence.",
            "Another text here.",
        ]
        result = splitter(texts)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_applyfn_conformance(self) -> None:
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["Test sentence."])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)

    def test_dotted_word_not_split(self) -> None:
        """Node.js should be one token and not cause a sentence split."""
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["You need Node.js version eighteen installed on your machine."])
        sentences = result[0]
        # Should be a single sentence — the dot in Node.js is not a boundary.
        assert len(sentences) == 1
        assert "Node.js" in sentences[0]

    def test_multiple_dotted_words(self) -> None:
        """Multiple dotted words should be preserved as single tokens."""
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["Use Node.js and Vue.js for your project. They work well."])
        # "Node.js" and "Vue.js" must appear intact in the output.
        joined = " ".join(result[0])
        assert "Node.js" in joined
        assert "Vue.js" in joined

    def test_eg_not_split(self) -> None:
        """e.g. should not cause a sentence break."""
        from adapters.preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance()
        result = splitter(["Use a framework e.g. React or Vue for this project."])
        sentences = result[0]
        assert len(sentences) == 1
        assert "e.g." in sentences[0]
