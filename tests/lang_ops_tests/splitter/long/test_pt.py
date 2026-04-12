"""Portuguese (pt) long-text splitting tests."""

from ._base import LongTextTestBase
from lang_ops._core._types import Span


TEXT_SAMPLE: str = 'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo. Na pág. 87 do relatório, tel. +55-11-98765-4321, descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte, ciência, música, etc. O projeto está pronto? É simplesmente incrível! A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte. A ciência transforma o mundo. Que futuro maravilhoso! Não é um grande futuro? A inovação brasileira promete.'

PARAGRAPH_TEXT: str = 'Primeiro parágrafo. Duas frases.\n\nSegundo parágrafo. Com três. Frases curtas.\n\nTerceiro e último parágrafo.'


class TestLongTextPortuguese(LongTextTestBase):
    LANGUAGE = "pt"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        assert self._split_sentences() == [
        'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo.',
        ' Na pág. 87 do relatório, tel. +55-11-98765-4321, descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte, ciência, música, etc. O projeto está pronto?',
        ' É simplesmente incrível!',
        ' A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte.',
        ' A ciência transforma o mundo.',
        ' Que futuro maravilhoso!',
        ' Não é um grande futuro?',
        ' A inovação brasileira promete.',
    ]

    def test_split_clauses(self) -> None:
        assert self._split_clauses() == [
        'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo. Na pág. 87 do relatório,',
        ' tel. +55-11-98765-4321,',
        ' descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte,',
        ' ciência,',
        ' música,',
        ' etc. O projeto está pronto? É simplesmente incrível! A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte. A ciência transforma o mundo. Que futuro maravilhoso! Não é um grande futuro? A inovação brasileira promete.',
    ]

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
        'Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São Paulo.',
        ' Na pág. 87 do relatório,',
        ' tel. +55-11-98765-4321,',
        ' descreve-se um projeto de aprox. 3.6 milhões... Os resultados incluem arte,',
        ' ciência,',
        ' música,',
        ' etc. O projeto está pronto?',
        ' É simplesmente incrível!',
        ' A Profa. Lima disse que o Sr. Oliveira visitou a exposição de arte.',
        ' A ciência transforma o mundo.',
        ' Que futuro maravilhoso!',
        ' Não é um grande futuro?',
        ' A inovação brasileira promete.',
    ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
        'Primeiro parágrafo.',
        ' Duas frases.',
        'Segundo parágrafo.',
        ' Com três.',
        ' Frases curtas.',
        'Terceiro e último parágrafo.',
    ]
