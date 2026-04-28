"""Checker factory — build a :class:`Checker` for a language pair.

Loads :class:`LangProfile` data and constructs a per-language-pair scene
that extends ``builtin.translate.strict`` with profile-derived parameter
overrides (length-ratio thresholds, expected question marks, forbidden
terms, keyword pairs, target-lang for CJK content).
"""

from __future__ import annotations

from .checkers import Checker
from .lang import LangProfile, ScriptFamily, get_profile
from .scene import SceneConfig

# -------------------------------------------------------------------
# Script family helpers
# -------------------------------------------------------------------

_SAME_SCRIPT = dict(short=5.0, medium=3.0, long=2.0, very_long=1.6)
_CJK_TO_LATIN = dict(short=8.0, medium=5.0, long=3.5, very_long=2.5)
_LATIN_TO_CJK = dict(short=4.0, medium=2.5, long=1.8, very_long=1.4)


def _ratio_thresholds(src_family: ScriptFamily, tgt_family: ScriptFamily) -> dict[str, float]:
    """Pick length-ratio thresholds based on cross-script direction."""
    if src_family == "cjk" and tgt_family != "cjk":
        return dict(_CJK_TO_LATIN)
    if src_family != "cjk" and tgt_family == "cjk":
        return dict(_LATIN_TO_CJK)
    return dict(_SAME_SCRIPT)


def _build_keyword_pairs(
    src: LangProfile,
    tgt: LangProfile,
) -> list[tuple[list[str], list[str]]]:
    """Build cross-language keyword pairs from concept intersections."""
    pairs: list[tuple[list[str], list[str]]] = []
    for concept, src_words in src.concept_words.items():
        tgt_words = tgt.concept_words.get(concept)
        if tgt_words:
            pairs.append((list(src_words), list(tgt_words)))
    return pairs


def _build_translate_scene(
    name: str,
    *,
    tgt_lang: str,
    src_profile: LangProfile,
    tgt_profile: LangProfile,
    base: str = "builtin.translate.strict",
) -> SceneConfig:
    """Build a scene that wires per-language profile data into the
    builtin translate preset via parameter overrides."""
    thresholds = _ratio_thresholds(src_profile.script_family, tgt_profile.script_family)
    keyword_pairs = _build_keyword_pairs(src_profile, tgt_profile)

    overrides: dict[str, dict] = {
        "length_ratio": {"params": thresholds},
        "format_artifacts": {
            "params": {
                "hallucination_starts": list(tgt_profile.hallucination_starts),
            }
        },
        "question_mark": {
            "params": {
                "source_marks": list(src_profile.question_marks),
                "expected_marks": list(tgt_profile.question_marks),
            }
        },
        "keywords": {
            "params": {
                "forbidden_terms": list(tgt_profile.forbidden_terms),
                "keyword_pairs": keyword_pairs,
            }
        },
        "cjk_content": {
            "params": {"target_lang": tgt_lang},
        },
    }
    return SceneConfig(name=name, extends=(base,), overrides=overrides)


def default_checker(source_lang: str, target_lang: str) -> Checker:
    """Build the default :class:`Checker` for a language pair.

    Returns a Checker bound to a per-language-pair scene
    ``translate.<src>.<tgt>`` that extends
    ``builtin.translate.strict`` with overrides drawn from the
    :class:`LangProfile` of each side.
    """
    src_profile = get_profile(source_lang)
    tgt_profile = get_profile(target_lang)

    scene_name = f"translate.{source_lang}.{target_lang}"
    return Checker(
        source_lang=source_lang,
        target_lang=target_lang,
        scenes={
            scene_name: _build_translate_scene(
                scene_name,
                tgt_lang=target_lang,
                src_profile=src_profile,
                tgt_profile=tgt_profile,
            ),
        },
        default_scene=scene_name,
    )
