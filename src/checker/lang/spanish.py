from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "aquí está la traducción", "la traducción es",
        "voy a traducir", "texto original",
        "```",
    ],
    hallucination_starts=[
        (r"^por supuesto[,.]?\s", None),
        (r"^claro[,.]?\s", None),
        (r"^entendido[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["traducción", "traducir"],
        "subtitle": ["subtítulo", "subtítulos"],
        "polish": ["pulir"],
        "context": ["contexto"],
    },
)
