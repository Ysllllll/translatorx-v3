"""Tests for the ``pos_connective`` chunk backend (spaCy POS-aware)."""

from __future__ import annotations

import pytest

from adapters.preprocess.availability import spacy_is_available

pytestmark = pytest.mark.skipif(not spacy_is_available(), reason="spacy not installed")


class TestPosConnectiveBackend:
    def test_registered(self) -> None:
        from adapters.preprocess.chunk.registry import ChunkBackendRegistry

        assert "pos_connective" in ChunkBackendRegistry.names()

    def test_invalid_min_words(self) -> None:
        from adapters.preprocess.chunk.backends.connective import pos_connective_backend

        with pytest.raises(ValueError):
            pos_connective_backend(language="en", min_words=0)

    def test_en_split_at_but(self) -> None:
        from adapters.preprocess.chunk.backends.connective import pos_connective_backend

        try:
            be = pos_connective_backend(language="en", min_words=3)
        except OSError:
            pytest.skip("spaCy en_core_web_md not installed")
        out = be(["I walked to the store but she stayed at home alone"])
        assert len(out[0]) >= 2

    def test_en_no_split_short(self) -> None:
        from adapters.preprocess.chunk.backends.connective import pos_connective_backend

        try:
            be = pos_connective_backend(language="en", min_words=5)
        except OSError:
            pytest.skip("spaCy en_core_web_md not installed")
        out = be(["I ran but she slept"])
        assert out == [["I ran but she slept"]]

    def test_en_that_as_determiner_not_split(self) -> None:
        """English `that` as a determiner (`that book`) must not split."""
        from adapters.preprocess.chunk.backends.connective import pos_connective_backend

        try:
            be = pos_connective_backend(language="en", min_words=2)
        except OSError:
            pytest.skip("spaCy en_core_web_md not installed")
        # "that book" is determiner — should not split here
        out = be(["I want to buy that book for my sister tomorrow"])
        # rule_connective would never split (that not in lexicon); POS variant
        # also must refuse because dep=det head=NOUN
        for piece in out[0]:
            assert not piece.strip().startswith("that book")

    def test_empty_input(self) -> None:
        from adapters.preprocess.chunk.backends.connective import pos_connective_backend

        try:
            be = pos_connective_backend(language="en")
        except OSError:
            pytest.skip("spaCy en_core_web_md not installed")
        assert be([""]) == [[""]]
