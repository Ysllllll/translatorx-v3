from . import LangProfile

PROFILE = LangProfile(
    script_family="cjk",
    forbidden_terms=[
        "翻訳してください",
        "翻訳します",
        "翻訳結果",
        "以下は翻訳です",
        "原文",
        "お役に立てて",
        "お手伝い",
        "```",
    ],
    hallucination_starts=[
        (r"^わかりました", None),
        (r"^了解しました", None),
        (r"^承知しました", None),
        (r"^もちろん[、,]", None),
    ],
    question_marks=["？", "?"],
    concept_words={
        "translate": ["翻訳"],
        "subtitle": ["字幕"],
        "polish": ["推敲", "添削"],
        "context": ["文脈", "コンテキスト"],
    },
)
