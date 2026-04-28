"""各语言检查器配置档案与注册表。

新增语言时，在本包中创建 ``{language}.py``（英文全名）文件，
定义模块级别的 ``PROFILE`` 变量，类型为 :class:`LangProfile`。
然后在下方 ``_LANG_TO_MODULE`` 中添加映射即可。
注册表在首次访问时自动发现。

示例（``_lang/arabic.py``）::

    from . import LangProfile

    PROFILE = LangProfile(
        script_family="other",
        forbidden_terms=["..."],
        hallucination_starts=[...],
        question_marks=["?", "؟"],
        concept_words={"translate": ["ترجمة"]},
    )
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal

ScriptFamily = Literal["latin", "cjk", "other"]


@dataclass(frozen=True)
class LangProfile:
    """单一目标语言的质量检查数据。

    属性
    ----------
    script_family :
        粗粒度文字系统分类（``latin`` / ``cjk`` / ``other``）。
        驱动 :mod:`application.checker.factory` 中的跨文字系统长度比阈值。
    forbidden_terms :
        不得出现在以该语言为目标的译文中的子串（不区分大小写）。
    hallucination_starts :
        ``(regex_pattern, exclude_pattern_or_None)`` 元组。
        如果 *exclude* 在主模式之后立即匹配，则跳过该规则。
    question_marks :
        该语言中视为有效问号的字符。
    concept_words :
        ``{concept_name: [surface_form, ...]}`` 映射。
        用于构建跨语言关键词一致性对：如果译文包含某个概念词但源文不包含，
        则模型可能生成了幻觉元响应。
    """

    script_family: ScriptFamily = "latin"
    forbidden_terms: list[str] = field(default_factory=list)
    hallucination_starts: list[tuple[str, str | None]] = field(default_factory=list)
    question_marks: list[str] = field(default_factory=lambda: ["?"])
    concept_words: dict[str, list[str]] = field(default_factory=dict)


# -------------------------------------------------------------------
# 注册表
# -------------------------------------------------------------------

# ISO 639-1 语言代码 → 模块名（英文全名）
_LANG_TO_MODULE: dict[str, str] = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "ru": "russian",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "pt": "portuguese",
    "vi": "vietnamese",
}

_registry: dict[str, LangProfile] = {}
_loaded = False

_EMPTY = LangProfile()


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    for lang, module_name in _LANG_TO_MODULE.items():
        try:
            mod = importlib.import_module(f".{module_name}", __package__)
            profile: LangProfile = getattr(mod, "PROFILE")
            _registry[lang] = profile
        except (ModuleNotFoundError, AttributeError):
            pass
    _loaded = True


def get_profile(lang: str) -> LangProfile:
    """返回指定语言的 :class:`LangProfile`，未注册则返回空档案。"""
    _ensure_loaded()
    return _registry.get(lang, _EMPTY)


def registered_langs() -> list[str]:
    """返回已注册配置档案的语言代码列表。"""
    _ensure_loaded()
    return sorted(_registry)
