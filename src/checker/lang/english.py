from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "here's the translation",
        "here is the translation",
        "i'll translate",
        "i will translate",
        "the translation is",
        "translated version",
        "```",
    ],
    hallucination_starts=[
        (r"^sure[,.]?\s", None),
        (r"^of course[,.]?\s", None),
        (r"^i understand", None),
        (r"^certainly[,.]?\s", None),
        (r"^no problem", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["translate", "translation", "translating"],
        "subtitle": ["subtitle", "subtitles"],
        "polish": ["polish", "polishing"],
        "context": ["context"],
    },
)
