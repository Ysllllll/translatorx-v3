"""Black-box tests for the ``spacy`` chunk backend and SpacySplitter singleton."""

from __future__ import annotations

import pytest

from adapters.preprocess.availability import spacy_is_available

pytestmark = pytest.mark.skipif(not spacy_is_available(), reason="spacy not installed")


class TestSpacyBackend:
    def test_basic_split(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["Hello world. This is a test. How are you?"])
        assert result == [["Hello world.", "This is a test.", "How are you?"]]

    def test_empty_input(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        assert backend([""]) == [[""]]

    def test_batch_processing(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["First text. Second sentence.", "Another text here."])
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_applyfn_conformance(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["Test sentence."])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)

    def test_dotted_word_not_split(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["You need Node.js version eighteen installed on your machine."])
        assert len(result[0]) == 1
        assert "Node.js" in result[0][0]

    def test_multiple_dotted_words(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["Use Node.js and Vue.js for your project. They work well."])
        joined = " ".join(result[0])
        assert "Node.js" in joined
        assert "Vue.js" in joined

    def test_eg_not_split(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend

        backend = spacy_backend(language="en")
        result = backend(["Use a framework e.g. React or Vue for this project."])
        assert len(result[0]) == 1
        assert "e.g." in result[0][0]


class TestSpacySplitterSingleton:
    def test_singleton_per_model(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import SpacySplitter

        a = SpacySplitter.get_instance()
        b = SpacySplitter.get_instance()
        assert a is b

    def test_for_language_english(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import SpacySplitter

        splitter = SpacySplitter.for_language("en")
        assert splitter is SpacySplitter.get_instance("en_core_web_md")

    def test_explicit_model_overrides_language(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import SpacySplitter

        splitter = SpacySplitter.for_language("zh", model="en_core_web_md")
        assert splitter is SpacySplitter.get_instance("en_core_web_md")

    def test_unknown_language_falls_back_to_multilingual(self) -> None:
        from adapters.preprocess.chunk.backends.spacy import DEFAULT_MODELS_BY_LANG, FALLBACK_MODEL

        assert DEFAULT_MODELS_BY_LANG.get("sv", FALLBACK_MODEL) == FALLBACK_MODEL
