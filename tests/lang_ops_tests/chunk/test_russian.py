"""Russian (ru) splitter tests."""

from lang_ops import LangOps, ChunkPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Доктор Иванов живёт на ул. Пушкина; его оклад составляет ок. 95 тыс. руб. в месяц. Он работает в НИИ, который получил 3.7 млн. рублей на исследование... Dr. Петров спросил: «Где хранятся данные?» Как удивительно! Команда из пятнадцати человек завершила работу в декабре. Это выдающийся результат! Просто невероятно? Да, технологии меняют мир.'

_ops = LangOps.for_language("ru")


class TestRussianSplitter(SplitterTestBase):
    LANGUAGE = "ru"
    TEXT_SAMPLE = TEXT_SAMPLE

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Привет мир. Как дела?") == ["Привет мир.", "Как дела?"]
        assert _ops.split_sentences("Ура! Правда? Да.") == ["Ура!", "Правда?", "Да."]

        # Consecutive terminators
        assert _ops.split_sentences("Подожди!! Правда???") == ["Подожди!!", "Правда???"]
        assert _ops.split_sentences("Что?! Да.") == ["Что?!", "Да."]

        # Abbreviation
        assert _ops.split_sentences("Доктор Иванов ушёл.") == ["Доктор Иванов ушёл."]

        # Ellipsis
        assert _ops.split_sentences("Подожди... Продолжай.") == ["Подожди... Продолжай."]

        # Number dot
        assert _ops.split_sentences("Значение 3.14 примерно.") == ["Значение 3.14 примерно."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Без терминаторов") == ["Без терминаторов"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Привет, мир.") == ["Привет,", "мир."]
        assert _ops.split_clauses("Первый; второй: третий.") == ["Первый;", "второй:", "третий."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]
        assert _ops.split_clauses("Привет,,, мир") == ["Привет,,,", "мир"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Без разделителей") == ["Без разделителей"]

    def test_split_by_length(self) -> None:
        # Character split
        assert _ops.split_by_length("Привет мир как дела", max_len=12) == ["Привет мир", "как дела"]

        # Fit / empty / edge
        assert _ops.split_by_length("Привет", max_len=20) == ["Привет"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("Привет", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Привет", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Привет", max_len=5, unit="sentence")

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
            'Доктор Иванов живёт на ул. Пушкина; его оклад составляет ок. 95 тыс. руб. в месяц.',
            'Он работает в НИИ, который получил 3.7 млн. рублей на исследование... Dr. Петров спросил: «Где хранятся данные?» Как удивительно!',
            'Команда из пятнадцати человек завершила работу в декабре.',
            'Это выдающийся результат!',
            'Просто невероятно?',
            'Да, технологии меняют мир.',
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            'Доктор Иванов живёт на ул. Пушкина;',
            'его оклад составляет ок. 95 тыс. руб. в месяц.',
            'Он работает в НИИ,',
            'который получил 3.7 млн. рублей на исследование... Dr. Петров спросил:',
            '«Где хранятся данные?» Как удивительно!',
            'Команда из пятнадцати человек завершила работу в декабре.',
            'Это выдающийся результат!',
            'Просто невероятно?',
            'Да,',
            'технологии меняют мир.',
        ]

        # long text chunk chain equivalence
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
