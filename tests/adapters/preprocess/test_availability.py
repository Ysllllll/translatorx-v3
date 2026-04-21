"""Tests for preprocess._availability guards."""

from adapters.preprocess.availability import (
    langdetect_is_available,
    punc_model_is_available,
    spacy_is_available,
)


def test_guards_return_bool() -> None:
    assert isinstance(punc_model_is_available(), bool)
    assert isinstance(spacy_is_available(), bool)
    assert isinstance(langdetect_is_available(), bool)
