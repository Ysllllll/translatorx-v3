from . import LangProfile

PROFILE = LangProfile(
    script_family="cjk",
    forbidden_terms=[
        "请翻译",
        "原句",
        "意译",
        "精译",
        "精翻",
        "保留原文",
        "可意译为",
        "可翻译为",
        "可精译为",
        "不客气",
        "效劳",
        "能帮到您",
        "能帮到你",
        "```",
    ],
    hallucination_starts=[
        (r"^明白了", r"吗"),
        (r"^知道了", r"吗"),
        (r"^没关系", None),
        (r"^好的[，,]", None),
        (r"^当然[，,]", r"(了|可以|是|啦)"),
    ],
    question_marks=["？", "?"],
    concept_words={
        "translate": ["翻译"],
        "subtitle": ["字幕"],
        "polish": ["润色"],
        "context": ["语境", "上下文"],
    },
)
