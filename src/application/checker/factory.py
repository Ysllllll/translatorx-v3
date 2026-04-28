"""Checker factory — build a :class:`Checker` from language codes.

Loads per-language :class:`LangProfile` data and constructs the
default rule list with language-appropriate parameters.  Also builds
profile-specific rule sets (lenient, minimal) for runtime switching.
"""

from __future__ import annotations

from .checkers import Checker
from .config import (
    CheckerConfig,
    PROFILES,
    ProfileOverrides,
)
from .lang import LangProfile, get_profile
from .rules import (
    LengthBounds,
    OutputTokenLimits,
    PixelWidthLimits,
    Rule,
    RatioThresholds,
    build_default_rules,
)
from .sanitize import (
    SanitizerChain,
    default_sanitizer_chain,
    BackticksStrip,
    TrailingAnnotationStrip,
    ColonToPunctuation,
    QuoteStrip,
    LeadingPunctStrip,
)
from .types import Severity


# -------------------------------------------------------------------
# Script family helpers
# -------------------------------------------------------------------

_CJK_LANGS = frozenset({"zh", "ja", "ko"})

_SAME_SCRIPT = dict(short=5.0, medium=3.0, long=2.0, very_long=1.6)
_CJK_TO_LATIN = dict(short=8.0, medium=5.0, long=3.5, very_long=2.5)
_LATIN_TO_CJK = dict(short=4.0, medium=2.5, long=1.8, very_long=1.4)


def _ratio_thresholds(src_lang: str, tgt_lang: str) -> RatioThresholds:
    src_cjk = src_lang in _CJK_LANGS
    tgt_cjk = tgt_lang in _CJK_LANGS
    if src_cjk and not tgt_cjk:
        return RatioThresholds(**_CJK_TO_LATIN)
    if not src_cjk and tgt_cjk:
        return RatioThresholds(**_LATIN_TO_CJK)
    return RatioThresholds(**_SAME_SCRIPT)


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


# -------------------------------------------------------------------
# Build rules with profile overrides applied
# -------------------------------------------------------------------


def _apply_profile_overrides(
    base_kwargs: dict,
    profile: ProfileOverrides,
) -> dict:
    """Merge profile overrides into base rule construction kwargs."""
    merged = dict(base_kwargs)
    if profile.ratio_severity is not None:
        merged["ratio_severity"] = profile.ratio_severity
    if profile.format_severity is not None:
        merged["format_severity"] = profile.format_severity
    if profile.keyword_severity is not None:
        merged["keyword_severity"] = profile.keyword_severity
    if profile.question_mark_severity is not None:
        merged["question_mark_severity"] = profile.question_mark_severity
    if profile.annotation_severity is not None:
        merged["annotation_severity"] = profile.annotation_severity

    # Threshold overrides — build new RatioThresholds
    base_thresholds: RatioThresholds = merged.get("ratio_thresholds") or RatioThresholds()
    t_overrides = {}
    if profile.ratio_thresholds_short is not None:
        t_overrides["short"] = profile.ratio_thresholds_short
    if profile.ratio_thresholds_medium is not None:
        t_overrides["medium"] = profile.ratio_thresholds_medium
    if profile.ratio_thresholds_long is not None:
        t_overrides["long"] = profile.ratio_thresholds_long
    if profile.ratio_thresholds_very_long is not None:
        t_overrides["very_long"] = profile.ratio_thresholds_very_long
    if t_overrides:
        merged["ratio_thresholds"] = RatioThresholds(
            short=t_overrides.get("short", base_thresholds.short),
            medium=t_overrides.get("medium", base_thresholds.medium),
            long=t_overrides.get("long", base_thresholds.long),
            very_long=t_overrides.get("very_long", base_thresholds.very_long),
        )

    return merged


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------


def default_checker(
    source_lang: str,
    target_lang: str,
    *,
    config_overrides: dict | None = None,
) -> Checker:
    """Build the default :class:`Checker` for a language pair.

    1. Selects ratio thresholds by script family.
    2. Loads :class:`LangProfile` data for rule parameters.
    3. Builds the base rule list and profile-specific variants.
    4. Returns a :class:`Checker` with profile switching support.

    Parameters
    ----------
    source_lang / target_lang :
        ISO 639-1 codes (e.g. ``"en"``, ``"zh"``).
    config_overrides :
        Partial dict to override any rule construction parameter.
        Keys match :func:`build_default_rules` keyword arguments.
    """
    src_profile = get_profile(source_lang)
    tgt_profile = get_profile(target_lang)

    # Base construction kwargs
    base_kwargs: dict = dict(
        ratio_thresholds=_ratio_thresholds(source_lang, target_lang),
        hallucination_starts=list(tgt_profile.hallucination_starts),
        expected_question_marks=list(tgt_profile.question_marks),
        forbidden_terms=list(tgt_profile.forbidden_terms),
        keyword_pairs=_build_keyword_pairs(src_profile, tgt_profile),
        target_lang=target_lang,
    )

    if config_overrides:
        base_kwargs.update(config_overrides)

    # Build base rules
    base_rules = build_default_rules(**base_kwargs)

    # Build profile-specific rule sets
    profile_rules: dict[str, list[Rule]] = {}
    for profile_name, profile_overrides in PROFILES.items():
        merged = _apply_profile_overrides(base_kwargs, profile_overrides)
        profile_rules[profile_name] = build_default_rules(**merged)

    return Checker(
        rules=base_rules,
        source_lang=source_lang,
        target_lang=target_lang,
        profile_rules=profile_rules,
    )


# -------------------------------------------------------------------
# Factory: from CheckerConfig (YAML)
# -------------------------------------------------------------------


def from_config(
    source_lang: str,
    target_lang: str,
    config: CheckerConfig,
) -> Checker:
    """Build a :class:`Checker` from a :class:`CheckerConfig` (YAML).

    Maps every section of :class:`CheckerConfig` to ``build_default_rules``
    keyword arguments, layering on top of the language-pair defaults
    produced by :func:`default_checker`.
    """
    src_profile = get_profile(source_lang)
    tgt_profile = get_profile(target_lang)

    # Resolve ratio thresholds — config overrides script-family defaults.
    base_ratio = _ratio_thresholds(source_lang, target_lang)
    rt = config.ratio_thresholds
    ratio_thresholds = RatioThresholds(
        short=rt.short if rt.short is not None else base_ratio.short,
        medium=rt.medium if rt.medium is not None else base_ratio.medium,
        long=rt.long if rt.long is not None else base_ratio.long,
        very_long=rt.very_long if rt.very_long is not None else base_ratio.very_long,
    )

    length_bounds = LengthBounds(
        abs_max=config.length_bounds.abs_max,
        short_target_max=config.length_bounds.short_target_max,
        short_target_inverse_ratio=config.length_bounds.short_target_inverse_ratio,
    )

    output_token_limits = OutputTokenLimits(
        max_output=config.output_tokens.max_output,
        short_input_threshold=config.output_tokens.short_input_threshold,
        short_input_max_output=config.output_tokens.short_input_max_output,
        output_input_ratio_max=config.output_tokens.output_input_ratio_max,
    )

    pixel_width_limits = PixelWidthLimits(
        font_path=config.pixel_width.font_path if config.pixel_width.enabled else "",
        font_size=config.pixel_width.font_size,
        max_ratio=config.pixel_width.max_ratio,
    )

    qm_severity_map = {
        "error": Severity.ERROR,
        "warning": Severity.WARNING,
        "info": Severity.INFO,
    }

    base_kwargs: dict = dict(
        ratio_thresholds=ratio_thresholds,
        length_bounds=length_bounds,
        hallucination_starts=list(tgt_profile.hallucination_starts),
        expected_question_marks=list(tgt_profile.question_marks),
        question_mark_whitelist=list(config.question_marks.whitelist_suffixes),
        forbidden_terms=list(tgt_profile.forbidden_terms),
        keyword_pairs=_build_keyword_pairs(src_profile, tgt_profile),
        target_lang=target_lang,
        output_token_limits=output_token_limits,
        pixel_width_limits=pixel_width_limits,
    )

    base_rules = build_default_rules(**base_kwargs)

    profile_rules: dict[str, list[Rule]] = {}
    for profile_name, profile_overrides in PROFILES.items():
        merged = _apply_profile_overrides(base_kwargs, profile_overrides)
        profile_rules[profile_name] = build_default_rules(**merged)

    return Checker(
        rules=base_rules,
        source_lang=source_lang,
        target_lang=target_lang,
        profile_rules=profile_rules,
    )


# -------------------------------------------------------------------
# Sanitizer chain from CheckerConfig
# -------------------------------------------------------------------


def sanitizer_from_config(config: CheckerConfig) -> SanitizerChain:
    """Build a :class:`SanitizerChain` from the toggles in *config*."""
    s = config.sanitize
    chain: list = []
    if s.backticks:
        chain.append(BackticksStrip())
    if s.trailing_annotation:
        chain.append(TrailingAnnotationStrip())
    if s.trailing_colon_to_punct:
        chain.append(ColonToPunctuation())
    if s.quote_strip:
        chain.append(QuoteStrip())
    if s.leading_punct_strip:
        chain.append(LeadingPunctStrip())
    return SanitizerChain(chain)
