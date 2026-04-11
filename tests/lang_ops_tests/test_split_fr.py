"""Sentence and clause splitting tests for French (fr)."""

import pytest

from lang_ops import TextOps
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences
from lang_ops import ChunkPipeline


def _ops(language: str) -> TextOps:
    return TextOps.for_language(language)


def _split_sentences(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return split_sentences(
        text,
        ops.sentence_terminators,
        ops.abbreviations,
        is_cjk=ops.is_cjk,
    )


def _split_clauses(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return split_clauses(text, ops.clause_separators)


# 541 characters. Topic: Parisian culture and architecture.
#
# Abbreviations: Mme., av., bd., éd., réf., no., etc., janv. — all in fr set.
# Number dots: 3.2 — should not split.
# No ellipsis in this sample.
#
# Sentence split points (10 total):
#   1. Paris.      — period (not abbreviation)
#   2. décennies.  — period (not abbreviation)
#   3. superbe!    — exclamation
#   4. unique.     — period (not abbreviation)
#   5. année?      — question
#   6. cités.      — period (not abbreviation)
#   7. monde.      — period (not abbreviation)
#   8. merveilleux! — exclamation
#   9. avenir?     — question
#  10. promet.     — period (not abbreviation)

TEXT_SAMPLE = (
    "Mme. Dupont habite au 15 av. des Champs-Élysées à Paris. "
    "Elle se promène souvent sur le bd. Haussmann; elle adore "
    "l'architecture haussmannienne, éd. originaire du XIXe siècle, "
    "réf. classée depuis 3.2 décennies. C'est vraiment superbe! "
    "Les marchés, les cafés, les librairies etc. rendent la ville "
    "unique. Avez-vous visité le no. 1 de la Place cette année? "
    "Chaque quartier offre des perspectives fascinantes sur les "
    "villes cités. En janv., les lumières illuminent le monde. "
    "Quel merveilleux! N'est-ce pas un bel avenir? La culture "
    "française le promet."
)

SENTENCE_COUNT = 10

CLAUSE_TEXT = "Paris, la capitale; est magnifique: voir pour croire."
# Comma: 1 → split after "Paris,"
# Semicolon: 1 → split after "capitale;"
# Colon: 1 → split after "magnifique:"
# Remainder: " voir pour croire."
# Total: 4 clauses

MULTI_PARAGRAPH = (
    "Premier paragraphe. Deux phrases.\n\n"
    "Deuxième paragraphe. Avec trois. Phrases courtes.\n\n"
    "Troisième et dernier paragraphe."
)


class TestSentenceSplitFr:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "fr")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "fr")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "fr")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "fr")
        joined = "".join(result)
        # "Mme." is a French abbreviation.
        assert "Mme. " in joined
        # "av." is a French abbreviation.
        assert "av. " in joined
        # "bd." is a French abbreviation.
        assert "bd. " in joined
        # "éd." is a French abbreviation.
        assert "éd. " in joined
        # "réf." is a French abbreviation.
        assert "réf. " in joined
        # "etc." is shared abbreviation.
        assert "etc. " in joined
        # "no." is a French abbreviation.
        assert "no. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "fr")
        joined = "".join(result)
        # "3.2" should stay together.
        assert "3.2" in joined
        for s in result:
            assert not s.startswith("2 ")

    def test_ellipsis_not_in_sample_but_reconstruction_works(self) -> None:
        # The sample does not have ellipsis, but verify reconstruction anyway.
        result = _split_sentences(TEXT_SAMPLE, "fr")
        assert "".join(result) == TEXT_SAMPLE

    def test_exclamation_and_question(self) -> None:
        result = _split_sentences("Super! Non? Oui.", "fr")
        assert result == ["Super!", " Non?", " Oui."]


class TestClauseSplitFr:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "fr")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "fr")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Paris, Lyon, Marseille", "fr")
        assert result == ["Paris,", " Lyon,", " Marseille"]

    def test_colon_split(self) -> None:
        result = _split_clauses("Attention: ceci est important", "fr")
        assert result == ["Attention:", " ceci est important"]

    def test_em_dash_split(self) -> None:
        result = _split_clauses("Début\u2014milieu\u2014fin", "fr")
        assert result == ["Début\u2014", "milieu\u2014", "fin"]


class TestPipelineFr:

    def test_sentences_then_clauses(self) -> None:
        result = (
            ChunkPipeline("Bonjour, monde. Au revoir, monde.", language="fr")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Bonjour,", " monde.", " Au revoir,", " monde."]

    def test_multi_paragraph(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="fr")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Bonjour. Monde.", language="fr")
        _derived = original.sentences().clauses()
        assert original.result() == ["Bonjour. Monde."]

    def test_sentences_on_sample(self) -> None:
        result = (
            ChunkPipeline(TEXT_SAMPLE, language="fr")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="fr")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
