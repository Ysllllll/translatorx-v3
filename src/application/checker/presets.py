"""Builtin scene presets.

These scenes are registered into the preset registry on import; they
become resolvable by :func:`resolve_scene` even when the user passes
no ``scenes`` mapping. Users may still:

- ``extends: [builtin.translate.strict]`` to start from a preset.
- Re-define a scene with the same name to shadow the preset.

Available presets:

- ``builtin.translate.strict``  — full translation gate (10 rules,
                                   5 sanitize steps).
- ``builtin.translate.lenient`` — same rule set but ``length_ratio`` /
                                   ``length_bounds`` / ``cjk_content``
                                   are downgraded to ``warning``.
- ``builtin.subtitle.line``     — minimal: ``non_empty`` only, plus the
                                   ``leading_punct_strip`` /
                                   ``strip_backticks`` sanitizers.
- ``builtin.llm.response``      — output-side only: ``format_artifacts``
                                   + ``output_tokens`` + sanitize of
                                   markdown noise.
"""

from __future__ import annotations

from ._scene import SceneConfig, register_preset_scene
from .types import RuleSpec, Severity


# ---------------------------------------------------------------------------
# Translation — strict
# ---------------------------------------------------------------------------


_TRANSLATE_STRICT = SceneConfig(
    name="builtin.translate.strict",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="trailing_annotation_strip"),
        RuleSpec(name="colon_to_punctuation"),
        RuleSpec(name="quote_strip"),
        RuleSpec(name="leading_punct_strip"),
    ),
    rules=(
        RuleSpec(name="non_empty", severity=Severity.ERROR),
        RuleSpec(name="length_bounds", severity=Severity.ERROR),
        RuleSpec(name="length_ratio", severity=Severity.ERROR),
        RuleSpec(name="format_artifacts", severity=Severity.ERROR),
        RuleSpec(name="cjk_content", severity=Severity.ERROR),
        RuleSpec(name="trailing_annotation", severity=Severity.ERROR),
        RuleSpec(name="keywords", severity=Severity.ERROR),
        RuleSpec(name="question_mark", severity=Severity.WARNING),
        RuleSpec(name="output_tokens", severity=Severity.WARNING),
        RuleSpec(name="pixel_width", severity=Severity.WARNING),
    ),
)


# ---------------------------------------------------------------------------
# Translation — lenient (downgrade hard length / cjk gates to warning)
# ---------------------------------------------------------------------------


_TRANSLATE_LENIENT = SceneConfig(
    name="builtin.translate.lenient",
    extends=("builtin.translate.strict",),
    overrides={
        "length_ratio": {"severity": Severity.WARNING},
        "length_bounds": {"severity": Severity.WARNING},
        "cjk_content": {"severity": Severity.WARNING},
    },
)


# ---------------------------------------------------------------------------
# Subtitle line — minimal
# ---------------------------------------------------------------------------


_SUBTITLE_LINE = SceneConfig(
    name="builtin.subtitle.line",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="leading_punct_strip"),
    ),
    rules=(RuleSpec(name="non_empty", severity=Severity.ERROR),),
)


# ---------------------------------------------------------------------------
# LLM response — output hygiene
# ---------------------------------------------------------------------------


_LLM_RESPONSE = SceneConfig(
    name="builtin.llm.response",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="quote_strip"),
    ),
    rules=(
        RuleSpec(name="non_empty", severity=Severity.ERROR),
        RuleSpec(name="format_artifacts", severity=Severity.WARNING),
        RuleSpec(name="output_tokens", severity=Severity.WARNING),
    ),
)


for _scene in (_TRANSLATE_STRICT, _TRANSLATE_LENIENT, _SUBTITLE_LINE, _LLM_RESPONSE):
    register_preset_scene(_scene)


__all__ = [
    "register_preset_scene",
]
