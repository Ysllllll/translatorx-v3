"""Per-language checker profile and registry.

To add a new language, create a file ``{language}.py`` (English full name)
in this package that defines a module-level ``PROFILE`` of type
:class:`LangProfile`.  Then add a mapping in ``_LANG_TO_MODULE`` below.
The registry discovers it automatically on first access.

Example (``_lang/arabic.py``)::

    from . import LangProfile

    PROFILE = LangProfile(
        forbidden_terms=["..."],
        hallucination_starts=[...],
        question_marks=["?", "؟"],
        concept_words={"translate": ["ترجمة"]},
    )
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LangProfile:
    """Quality-check data for a single target language.

    Attributes
    ----------
    forbidden_terms :
        Substrings that must NOT appear in a translation targeting
        this language (case-insensitive).
    hallucination_starts :
        ``(regex_pattern, exclude_pattern_or_None)`` tuples.
        If *exclude* matches immediately after the main pattern the
        rule is skipped.
    question_marks :
        Characters considered valid question marks in this language.
    concept_words :
        ``{concept_name: [surface_form, ...]}`` mapping.  Used to
        build cross-language keyword-consistency pairs: if the target
        translation contains a concept word but the source does not,
        the model likely hallucinated a meta-response.
    """

    forbidden_terms: list[str] = field(default_factory=list)
    hallucination_starts: list[tuple[str, str | None]] = field(default_factory=list)
    question_marks: list[str] = field(default_factory=lambda: ["?"])
    concept_words: dict[str, list[str]] = field(default_factory=dict)


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

# ISO 639-1 code → module name (English full name)
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
    """Return the :class:`LangProfile` for *lang*, or an empty one."""
    _ensure_loaded()
    return _registry.get(lang, _EMPTY)


def registered_langs() -> list[str]:
    """Return the list of language codes with registered profiles."""
    _ensure_loaded()
    return sorted(_registry)
