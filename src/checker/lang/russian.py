from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "вот перевод", "перевод:", "переведу",
        "перевод текста", "оригинал",
        "```",
    ],
    hallucination_starts=[
        (r"^конечно[,.]?\s", None),
        (r"^понятно[,.]?\s", None),
        (r"^хорошо[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["перевод", "переводить"],
        "subtitle": ["субтитры", "субтитр"],
        "polish": ["редактировать", "редактирование"],
        "context": ["контекст"],
    },
)
