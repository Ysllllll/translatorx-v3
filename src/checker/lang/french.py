from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "voici la traduction",
        "la traduction est",
        "je vais traduire",
        "texte original",
        "```",
    ],
    hallucination_starts=[
        (r"^bien sûr[,.]?\s", None),
        (r"^d'accord[,.]?\s", None),
        (r"^compris[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["traduction", "traduire"],
        "subtitle": ["sous-titre", "sous-titres"],
        "polish": ["peaufiner"],
        "context": ["contexte"],
    },
)
