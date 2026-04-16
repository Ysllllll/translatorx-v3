from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "đây là bản dịch", "bản dịch là",
        "tôi sẽ dịch", "văn bản gốc",
        "```",
    ],
    hallucination_starts=[
        (r"^tất nhiên[,.]?\s", None),
        (r"^được rồi[,.]?\s", None),
        (r"^hiểu rồi[,.]?\s", None),
    ],
    question_marks=["?"],
    concept_words={
        "translate": ["dịch", "bản dịch"],
        "subtitle": ["phụ đề"],
        "polish": ["chau chuốt"],
        "context": ["ngữ cảnh"],
    },
)
