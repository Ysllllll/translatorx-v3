from . import LangProfile

PROFILE = LangProfile(
    forbidden_terms=[
        "번역해 주세요", "번역해주세요", "번역 결과",
        "원문", "다음은 번역입니다",
        "도움이 되셨", "도와드릴",
        "```",
    ],
    hallucination_starts=[
        (r"^알겠습니다", None),
        (r"^네[,.]?\s", None),
        (r"^물론이죠", None),
    ],
    question_marks=["?", "？"],
    concept_words={
        "translate": ["번역"],
        "subtitle": ["자막"],
        "polish": ["다듬기", "윤색"],
        "context": ["문맥", "맥락"],
    },
)
