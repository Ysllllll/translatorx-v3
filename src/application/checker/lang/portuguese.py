from . import LangProfile

PROFILE = LangProfile(
    script_family="latin",
    forbidden_terms=[
        "aqui está a tradução",
        "a tradução é",
        "vou traduzir",
        "texto original",
        "```",
    ],
    hallucination_starts=[
        (r"^claro[,.]?\s", None),
        (r"^entendido[,.]?\s", None),
        (r"^com certeza[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["tradução", "traduzir"],
        "subtitle": ["legenda", "legendas"],
        "polish": ["polir"],
        "context": ["contexto"],
    },
)
