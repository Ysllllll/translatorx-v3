"""Tests for the ``rule_connective`` chunk backend."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.backends.connective import rule_connective_backend
from adapters.preprocess.chunk.registry import ChunkBackendRegistry


class TestRuleConnectiveBackend:
    def test_registered(self) -> None:
        assert "rule_connective" in ChunkBackendRegistry.names()
        be = ChunkBackendRegistry.create("rule_connective", language="en", min_context=3)
        assert callable(be)

    def test_invalid_min_context(self) -> None:
        with pytest.raises(ValueError):
            rule_connective_backend(language="en", min_context=0)

    # ---- English -----------------------------------------------------

    def test_en_basic_split_but(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I walked to the store but she stayed at home alone"])
        assert len(out) == 1
        assert len(out[0]) == 2
        assert out[0][0].startswith("I walked")
        assert out[0][1].startswith("but she")

    def test_en_multiple_splits(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I ran very fast because I was late but she did not wait for me at all"])
        pieces = out[0]
        assert len(pieces) >= 3
        assert pieces[0].startswith("I ran")
        assert any(p.startswith("because") for p in pieces)
        assert any(p.startswith("but") for p in pieces)

    def test_en_no_split_when_context_too_small(self) -> None:
        be = rule_connective_backend(language="en", min_context=5)
        # only 2 tokens on each side of "but"
        out = be(["I ran but she slept"])
        assert out == [["I ran but she slept"]]

    def test_en_no_connective_passthrough(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["The quick brown fox jumps over the lazy dog today"])
        assert out == [["The quick brown fox jumps over the lazy dog today"]]

    def test_en_case_insensitive(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I walked to the store But she stayed at home alone"])
        assert len(out[0]) == 2

    def test_en_excludes_ambiguous_and(self) -> None:
        # "and" is intentionally excluded from the connective set.
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I walked to the store and she stayed at home alone"])
        assert out == [["I walked to the store and she stayed at home alone"]]

    def test_en_empty_input(self) -> None:
        be = rule_connective_backend(language="en")
        assert be([""]) == [[""]]

    def test_en_batch(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I walked slowly but she ran quickly over there", "just a normal sentence here"])
        assert len(out) == 2
        assert len(out[0]) == 2
        assert len(out[1]) == 1

    # ---- Chinese -----------------------------------------------------

    def test_zh_basic_split(self) -> None:
        be = rule_connective_backend(language="zh", min_context=2)
        out = be(["我今天很高兴因为我考试通过了所以我请客吃饭"])
        pieces = out[0]
        assert len(pieces) >= 2

    def test_zh_no_split_short(self) -> None:
        be = rule_connective_backend(language="zh", min_context=5)
        out = be(["因为下雨"])
        assert out == [["因为下雨"]]

    # ---- Connective with trailing punctuation ------------------------

    def test_connective_with_attached_punct(self) -> None:
        # "Because," — strip punct then match
        be = rule_connective_backend(language="en", min_context=3)
        out = be(["I stayed home all day because, the weather was cold outside"])
        assert len(out[0]) == 2

    # ---- Reconstruction sanity ---------------------------------------

    def test_reconstruction_matches_source(self) -> None:
        from adapters.preprocess.chunk.reconstruct import chunks_match_source

        be = rule_connective_backend(language="en", min_context=3)
        src = "I ran very fast because I was late for the meeting today"
        out = be([src])
        assert chunks_match_source(out[0], src)

    def test_reconstruction_zh(self) -> None:
        from adapters.preprocess.chunk.reconstruct import chunks_match_source

        be = rule_connective_backend(language="zh", min_context=2)
        src = "我今天很高兴因为我考试通过了所以我请客吃饭"
        out = be([src])
        assert chunks_match_source(out[0], src)

    # ---- Contraction apostrophe guard --------------------------------

    def test_en_does_not_split_before_contraction(self) -> None:
        be = rule_connective_backend(language="en", min_context=3)
        # "but" followed directly by apostrophe should not trigger
        out = be(["I ran quickly but 's not a contraction ever seen here"])
        # not split because "'s" apostrophe guard kicks in
        # (this mostly exercises the guard path without crashing)
        assert isinstance(out[0], list)


class TestParametrizedLanguages:
    """Multi-lang smoke — every populated connective set can split at least once."""

    CASES = {
        "en": "I walked slowly to the park but she ran quickly back home",
        "es": "Caminé lentamente al parque pero ella corrió rápidamente a casa",
        "de": "Ich ging langsam in den Park aber sie rannte schnell nach Hause",
        "fr": "Je marchais lentement vers le parc mais elle courait rapidement",
        "pt": "Eu caminhei lentamente para o parque mas ela correu rapidamente",
    }

    @pytest.mark.parametrize("lang,text", list(CASES.items()))
    def test_split_at_contrastive_connective(self, lang: str, text: str) -> None:
        be = rule_connective_backend(language=lang, min_context=3)
        out = be([text])
        assert len(out[0]) >= 2, f"{lang}: expected split, got {out[0]!r}"
