from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "hier ist die übersetzung",
        "die übersetzung lautet",
        "ich werde übersetzen",
        "originaltext",
        "```",
    ],
    hallucination_starts=[
        (r"^natürlich[,.]?\s", None),
        (r"^verstanden[,.]?\s", None),
        (r"^klar[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["übersetzung", "übersetzen"],
        "subtitle": ["untertitel"],
        "polish": ["überarbeiten"],
        "context": ["kontext"],
    },
)
