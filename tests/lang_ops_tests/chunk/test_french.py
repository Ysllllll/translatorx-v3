"""French (fr) splitter tests."""

from lang_ops import LangOps, TextPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = "Mme. Dupont habite au 15 av. des Champs-Élysées à Paris. Elle se promène souvent sur le bd. Haussmann; elle adore l'architecture haussmannienne, éd. originaire du XIXe siècle, réf. classée depuis 3.2 décennies. C'est vraiment superbe! Les marchés, les cafés, les librairies etc. rendent la ville unique. Avez-vous visité le no. 1 de la Place cette année? Chaque quartier offre des perspectives fascinantes sur les villes cités. En janv., les lumières illuminent le monde. Quel merveilleux! N'est-ce pas un bel avenir? La culture française le promet."

_ops = LangOps.for_language("fr")


class TestFrenchSplitter(SplitterTestBase):
    LANGUAGE = "fr"
    TEXT_SAMPLE = TEXT_SAMPLE

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Bonjour monde. Comment vas-tu?") == ["Bonjour monde.", "Comment vas-tu?"]
        assert _ops.split_sentences("Incroyable! Vraiment? Oui.") == ["Incroyable!", "Vraiment?", "Oui."]

        # Consecutive terminators
        assert _ops.split_sentences("Attends!! Vraiment???") == ["Attends!!", "Vraiment???"]

        # Abbreviation
        assert _ops.split_sentences("Mme. Dupont est partie.") == ["Mme. Dupont est partie."]

        # Ellipsis
        assert _ops.split_sentences("Attends... Continue.") == ["Attends... Continue."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Pas de terminateurs") == ["Pas de terminateurs"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Bonjour, monde.") == ["Bonjour,", "monde."]
        assert _ops.split_clauses("Premier; deuxième: troisième.") == ["Premier;", "deuxième:", "troisième."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Pas de séparateurs") == ["Pas de séparateurs"]

    def test_split_by_length(self) -> None:
        # Character split
        # Multi-word split
        assert _ops.split_by_length("Bonjour le monde entier", max_len=12) == ["Bonjour le", "monde entier"]

        # Fit / empty / edge
        assert _ops.split_by_length("Bonjour", max_len=20) == ["Bonjour"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("Bonjour", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Bonjour", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Bonjour", max_len=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test. Another one.").sentences().split(20).result() == [
            "Hello world.", "This is a test.", "Another one.",
        ]
        assert _ops.chunk("First clause, second clause, and third.").clauses().split(20).result() == [
            "First clause,", "second clause,", "and third.",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            'Mme. Dupont habite au 15 av. des Champs-Élysées à Paris.',
            "Elle se promène souvent sur le bd. Haussmann; elle adore l'architecture haussmannienne, éd. originaire du XIXe siècle, réf. classée depuis 3.2 décennies.",
            "C'est vraiment superbe!",
            'Les marchés, les cafés, les librairies etc. rendent la ville unique.',
            'Avez-vous visité le no. 1 de la Place cette année?',
            'Chaque quartier offre des perspectives fascinantes sur les villes cités.',
            'En janv., les lumières illuminent le monde.',
            'Quel merveilleux!',
            "N'est-ce pas un bel avenir?",
            'La culture française le promet.',
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            'Mme. Dupont habite au 15 av. des Champs-Élysées à Paris.',
            'Elle se promène souvent sur le bd. Haussmann;',
            "elle adore l'architecture haussmannienne,",
            'éd. originaire du XIXe siècle,',
            'réf. classée depuis 3.2 décennies.',
            "C'est vraiment superbe!",
            'Les marchés,',
            'les cafés,',
            'les librairies etc. rendent la ville unique.',
            'Avez-vous visité le no. 1 de la Place cette année?',
            'Chaque quartier offre des perspectives fascinantes sur les villes cités.',
            'En janv.,',
            'les lumières illuminent le monde.',
            'Quel merveilleux!',
            "N'est-ce pas un bel avenir?",
            'La culture française le promet.',
        ]

        # long text chunk chain equivalence
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
