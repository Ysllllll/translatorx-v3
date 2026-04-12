"""French (fr) splitter tests."""

from ._base import SplitterTestBase


TEXT_SAMPLE: str = "Mme. Dupont habite au 15 av. des Champs-Élysées à Paris. Elle se promène souvent sur le bd. Haussmann; elle adore l'architecture haussmannienne, éd. originaire du XIXe siècle, réf. classée depuis 3.2 décennies. C'est vraiment superbe! Les marchés, les cafés, les librairies etc. rendent la ville unique. Avez-vous visité le no. 1 de la Place cette année? Chaque quartier offre des perspectives fascinantes sur les villes cités. En janv., les lumières illuminent le monde. Quel merveilleux! N'est-ce pas un bel avenir? La culture française le promet."

PARAGRAPH_TEXT: str = 'Premier paragraphe. Deux phrases.\n\nDeuxième paragraphe. Avec trois. Phrases courtes.\n\nTroisième et dernier paragraphe.'


class TestFrenchSplitter(SplitterTestBase):
    LANGUAGE = "fr"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            'Mme. Dupont habite au 15 av. des Champs-Élysées à Paris.',
            " Elle se promène souvent sur le bd. Haussmann; elle adore l'architecture haussmannienne, éd. originaire du XIXe siècle, réf. classée depuis 3.2 décennies.",
            " C'est vraiment superbe!",
            ' Les marchés, les cafés, les librairies etc. rendent la ville unique.',
            ' Avez-vous visité le no. 1 de la Place cette année?',
            ' Chaque quartier offre des perspectives fascinantes sur les villes cités.',
            ' En janv., les lumières illuminent le monde.',
            ' Quel merveilleux!',
            " N'est-ce pas un bel avenir?",
            ' La culture française le promet.',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            'Mme. Dupont habite au 15 av. des Champs-Élysées à Paris.',
            ' Elle se promène souvent sur le bd. Haussmann;',
            " elle adore l'architecture haussmannienne,",
            ' éd. originaire du XIXe siècle,',
            ' réf. classée depuis 3.2 décennies.',
            " C'est vraiment superbe!",
            ' Les marchés,',
            ' les cafés,',
            ' les librairies etc. rendent la ville unique.',
            ' Avez-vous visité le no. 1 de la Place cette année?',
            ' Chaque quartier offre des perspectives fascinantes sur les villes cités.',
            ' En janv.,',
            ' les lumières illuminent le monde.',
            ' Quel merveilleux!',
            " N'est-ce pas un bel avenir?",
            ' La culture française le promet.',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            'Mme. Dupont habite au 15 av. des Champs-Élysées à Paris.',
            ' Elle se promène souvent sur le bd. Haussmann;',
            " elle adore l'architecture haussmannienne,",
            ' éd. originaire du XIXe siècle,',
            ' réf. classée depuis 3.2 décennies.',
            " C'est vraiment superbe!",
            ' Les marchés,',
            ' les cafés,',
            ' les librairies etc. rendent la ville unique.',
            ' Avez-vous visité le no. 1 de la Place cette année?',
            ' Chaque quartier offre des perspectives fascinantes sur les villes cités.',
            ' En janv.,',
            ' les lumières illuminent le monde.',
            ' Quel merveilleux!',
            " N'est-ce pas un bel avenir?",
            ' La culture française le promet.',
        ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            'Premier paragraphe.',
            ' Deux phrases.',
            'Deuxième paragraphe.',
            ' Avec trois.',
            ' Phrases courtes.',
            'Troisième et dernier paragraphe.',
        ]
