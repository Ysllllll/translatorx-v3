"""Spanish (es) splitter tests."""

from domain.lang import LangOps, TextPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = "Dra. García y la Sra. López caminan por la av. Reforma en Madrid. En pág. 42 del informe, tel. +34-91-555-0100, se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte, ciencia, música, etc. ¿Ha terminado? ¡Es increíble! La Profa. Ruiz preguntó si Ud. conoce la exposición de arte. Cultura y ciencia transforman el mundo. ¡Qué maravilloso! No es un gran futuro? La tradición española lo promete."

_ops = LangOps.for_language("es")


class TestSpanishSplitter(SplitterTestBase):
    LANGUAGE = "es"
    TEXT_SAMPLE = TEXT_SAMPLE

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Hola mundo. ¿Cómo estás?") == ["Hola mundo.", "¿Cómo estás?"]
        assert _ops.split_sentences("¡Increíble! ¿De verdad? Sí.") == ["¡Increíble!", "¿De verdad?", "Sí."]

        # Consecutive terminators
        assert _ops.split_sentences("¡¡Espera!! ¿¿¿De verdad???") == ["¡¡Espera!!", "¿¿¿De verdad???"]

        # Abbreviation
        assert _ops.split_sentences("Dra. García se fue.") == ["Dra. García se fue."]

        # Ellipsis
        assert _ops.split_sentences("Espera... Continúa.") == ["Espera... Continúa."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Sin terminadores") == ["Sin terminadores"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Hola, mundo.") == ["Hola,", "mundo."]
        assert _ops.split_clauses("Primero; segundo: tercero.") == ["Primero;", "segundo:", "tercero."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Sin separadores") == ["Sin separadores"]

    def test_split_by_length(self) -> None:
        # Character split
        # Multi-word split
        assert _ops.split_by_length("Hola mundo como estas", max_len=12) == ["Hola mundo", "como estas"]

        # Fit / empty / edge
        assert _ops.split_by_length("Hola", max_len=20) == ["Hola"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest

        with pytest.raises(ValueError):
            _ops.split_by_length("Hola", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Hola", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Hola", max_len=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test. Another one.").sentences().split(20).result() == [
            "Hello world.",
            "This is a test.",
            "Another one.",
        ]
        assert _ops.chunk("First clause, second clause, and third.").clauses().split(20).result() == [
            "First clause,",
            "second clause,",
            "and third.",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            "Dra. García y la Sra. López caminan por la av. Reforma en Madrid.",
            "En pág. 42 del informe, tel. +34-91-555-0100, se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte, ciencia, música, etc. ¿Ha terminado?",
            "¡Es increíble!",
            "La Profa. Ruiz preguntó si Ud. conoce la exposición de arte.",
            "Cultura y ciencia transforman el mundo.",
            "¡Qué maravilloso!",
            "No es un gran futuro?",
            "La tradición española lo promete.",
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            "Dra. García y la Sra. López caminan por la av. Reforma en Madrid.",
            "En pág. 42 del informe,",
            "tel. +34-91-555-0100,",
            "se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte,",
            "ciencia,",
            "música,",
            "etc. ¿Ha terminado?",
            "¡Es increíble!",
            "La Profa. Ruiz preguntó si Ud. conoce la exposición de arte.",
            "Cultura y ciencia transforman el mundo.",
            "¡Qué maravilloso!",
            "No es un gran futuro?",
            "La tradición española lo promete.",
        ]

        # long text chunk chain equivalence
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
