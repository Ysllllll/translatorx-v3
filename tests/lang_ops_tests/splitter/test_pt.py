"""Portuguese (pt) splitter tests."""

from lang_ops import TextOps, ChunkPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo. Na pág. 87 do relatório, tel. +55-11-98765-4321, descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte, ciência, música, etc. O projeto está pronto? É simplesmente incrível! A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte. A ciência transforma o mundo. Que futuro maravilhoso! Não é um grande futuro? A inovação brasileira promete.'

PARAGRAPH_TEXT: str = 'Primeiro parágrafo. Duas frases.\n\nSegundo parágrafo. Com três. Frases curtas.\n\nTerceiro e último parágrafo.'

_ops = TextOps.for_language("pt")


class TestPortugueseSplitter(SplitterTestBase):
    LANGUAGE = "pt"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Olá mundo. Como vai?") == ["Olá mundo.", "Como vai?"]
        assert _ops.split_sentences("Incrível! Mesmo? Sim.") == ["Incrível!", "Mesmo?", "Sim."]

        # Consecutive terminators
        assert _ops.split_sentences("Espere!! Mesmo???") == ["Espere!!", "Mesmo???"]

        # Abbreviation
        assert _ops.split_sentences("Dra. Santos saiu.") == ["Dra. Santos saiu."]

        # Ellipsis
        assert _ops.split_sentences("Espere... Continue.") == ["Espere... Continue."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Sem terminadores") == ["Sem terminadores"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Olá, mundo.") == ["Olá,", "mundo."]
        assert _ops.split_clauses("Primeiro; segundo: terceiro.") == ["Primeiro;", "segundo:", "terceiro."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Sem separadores") == ["Sem separadores"]

    def test_split_by_length(self) -> None:
        # Character split
        # Multi-word split
        assert _ops.split_by_length("Olá mundo como vai", max_length=10) == ["Olá mundo", "como vai"]

        # Fit / empty / edge
        assert _ops.split_by_length("Olá", max_length=20) == ["Olá"]
        assert _ops.split_by_length("", max_length=10) == []

        # Errors
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("Olá", max_length=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Olá", max_length=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Olá", max_length=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test. Another one.").sentences().by_length(20).result() == [
            "Hello world.", "This is a test.", "Another one.",
        ]
        assert _ops.chunk("First clause, second clause, and third.").clauses().by_length(20).result() == [
            "First clause,", "second clause,", "and third.",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo.',
            'Na pág. 87 do relatório, tel. +55-11-98765-4321, descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte, ciência, música, etc. O projeto está pronto?',
            'É simplesmente incrível!',
            'A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte.',
            'A ciência transforma o mundo.',
            'Que futuro maravilhoso!',
            'Não é um grande futuro?',
            'A inovação brasileira promete.',
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo.',
            'Na pág. 87 do relatório,',
            'tel. +55-11-98765-4321,',
            'descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte,',
            'ciência,',
            'música,',
            'etc. O projeto está pronto?',
            'É simplesmente incrível!',
            'A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte.',
            'A ciência transforma o mundo.',
            'Que futuro maravilhoso!',
            'Não é um grande futuro?',
            'A inovação brasileira promete.',
        ]

        # long text chunk chain equivalence
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)

        # long text pipeline_paragraphs_sentences()
        assert ChunkPipeline(self.PARAGRAPH_TEXT, language=self.LANGUAGE).paragraphs().sentences().result() == [
            'Primeiro parágrafo.',
            'Duas frases.',
            'Segundo parágrafo.',
            'Com três.',
            'Frases curtas.',
            'Terceiro e último parágrafo.',
        ]
