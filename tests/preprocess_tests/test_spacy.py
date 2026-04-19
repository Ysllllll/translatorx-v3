"""Tests for SpacySplitter."""

from __future__ import annotations

import pytest

from preprocess._availability import spacy_is_available

pytestmark = pytest.mark.skipif(
    not spacy_is_available(),
    reason="spacy not installed",
)


class TestSpacySplitter:
    def test_singleton_per_model(self) -> None:
        from preprocess import SpacySplitter

        a = SpacySplitter.get_instance("en_core_web_md")
        b = SpacySplitter.get_instance("en_core_web_md")
        assert a is b

    def test_basic_split(self) -> None:
        from preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance("en_core_web_md")
        result = splitter(["Hello world. This is a test. How are you?"])
        assert len(result) == 1
        sentences = result[0]
        assert len(sentences) >= 2  # At least 2 sentences

    def test_empty_input(self) -> None:
        from preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance("en_core_web_md")
        result = splitter([""])
        assert result == [[""]]

    def test_single_sentence(self) -> None:
        from preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance("en_core_web_md")
        result = splitter(["Hello world."])
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_batch_processing(self) -> None:
        from preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance("en_core_web_md")
        texts = [
            "First text. Second sentence.",
            "Another text here.",
        ]
        result = splitter(texts)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_applyfn_conformance(self) -> None:
        from preprocess import SpacySplitter

        splitter = SpacySplitter.get_instance("en_core_web_md")
        result = splitter(["Test sentence."])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)
