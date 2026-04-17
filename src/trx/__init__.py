"""trx — TranslatorX unified API.

Convenience facade that re-exports commonly used types from lower-level
packages and provides factory functions to reduce boilerplate.

Quick start::

    import trx

    engine = trx.create_engine(model="Qwen/Qwen3-32B", base_url="http://localhost:26592/v1")
    result = await trx.translate_srt(srt_content, engine, src="en", tgt="zh")
    for seg in result:
        print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")

The lower-level packages (``lang_ops``, ``subtitle``, ``llm_ops``,
``checker``, ``pipeline``) remain fully accessible for advanced use.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports — common types users need
# ---------------------------------------------------------------------------

# Model types
from model import Segment, SentenceRecord, Word

# Subtitle
from subtitle import Subtitle, SubtitleStream
from subtitle.io import parse_srt, read_srt, parse_whisperx, read_whisperx

# LangOps
from lang_ops import LangOps, ChunkPipeline

# LLM
from llm_ops import (
    ContextWindow,
    EngineConfig,
    LLMEngine,
    OneShotTerms,
    OpenAICompatEngine,
    PreloadableTerms,
    StaticTerms,
    TermsAgent,
    TermsProvider,
    TranslateResult,
    TranslationContext,
    translate_with_verify,
)

# Checker
from checker import (
    CheckReport,
    Checker,
    Severity,
    default_checker,
)

# Pipeline
from pipeline import (
    EN_ZH_PREFIX_RULES,
    FeedResult,
    Pipeline,
    PrefixRule,
    ProgressCallback,
    StreamAdapter,
    TranslateNodeConfig,
)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_engine(
    model: str,
    base_url: str,
    *,
    api_key: str = "EMPTY",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 150.0,
    extra_body: dict | None = None,
) -> OpenAICompatEngine:
    """Create an LLM engine with sensible defaults.

    Example::

        engine = trx.create_engine(
            model="Qwen/Qwen3-32B",
            base_url="http://localhost:26592/v1",
        )
    """
    config = EngineConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body or {},
    )
    return OpenAICompatEngine(config)


def create_context(
    src: str,
    tgt: str,
    *,
    terms: dict[str, str] | None = None,
    terms_provider: TermsProvider | None = None,
    frozen_pairs: tuple[tuple[str, str], ...] = (),
    window_size: int = 4,
    max_retries: int = 3,
    system_prompt_template: str = "",
) -> TranslationContext:
    """Create a translation context.

    Supply terminology in one of two ways:

    * ``terms={...}`` — convenience form; wraps the dict in
      :class:`StaticTerms` for you.
    * ``terms_provider=...`` — pass a pre-built provider
      (e.g. :class:`PreloadableTerms`, :class:`OneShotTerms`).

    ``frozen_pairs`` are extra non-term reference pairs concatenated
    *after* the provider terms in the few-shot context.

    Example::

        ctx = trx.create_context("en", "zh", terms={"AI": "人工智能"})

        # Or with a dynamic provider:
        provider = trx.PreloadableTerms(engine, "en", "zh")
        await provider.preload(all_source_texts)
        ctx = trx.create_context("en", "zh", terms_provider=provider)
    """
    if terms is not None and terms_provider is not None:
        raise ValueError("Pass either 'terms' or 'terms_provider', not both.")

    provider: TermsProvider
    if terms_provider is not None:
        provider = terms_provider
    elif terms:
        provider = StaticTerms(terms)
    else:
        provider = StaticTerms()

    return TranslationContext(
        source_lang=src,
        target_lang=tgt,
        terms_provider=provider,
        frozen_pairs=frozen_pairs,
        window_size=window_size,
        max_retries=max_retries,
        system_prompt_template=system_prompt_template,
    )


async def translate_srt(
    srt_content: str,
    engine: LLMEngine,
    src: str = "en",
    tgt: str = "zh",
    *,
    terms: dict[str, str] | None = None,
    config: TranslateNodeConfig | None = None,
    progress: ProgressCallback | None = None,
) -> list[SentenceRecord]:
    """Translate SRT content end-to-end in one call.

    Returns a list of :class:`SentenceRecord` with translations populated.

    Example::

        records = await trx.translate_srt(
            srt_content, engine, src="en", tgt="zh",
            terms={"machine learning": "机器学习"},
        )
        for r in records:
            print(f"{r.src_text} → {r.translations['zh']}")
    """
    segments = parse_srt(srt_content)
    sub = Subtitle(segments, language=src)
    records = sub.records()

    ctx = create_context(src, tgt, terms=terms)
    chk = default_checker(src, tgt)

    p = Pipeline(records)
    translated = await p.translate(engine, ctx, chk, config=config, progress=progress)
    return translated.build()


__all__ = [
    # Factory functions
    "create_engine",
    "create_context",
    "translate_srt",
    # Model types
    "Word",
    "Segment",
    "SentenceRecord",
    # Subtitle
    "Subtitle",
    "SubtitleStream",
    "parse_srt",
    "read_srt",
    "parse_whisperx",
    "read_whisperx",
    # LangOps
    "LangOps",
    "ChunkPipeline",
    # LLM
    "LLMEngine",
    "OpenAICompatEngine",
    "EngineConfig",
    "TranslationContext",
    "StaticTerms",
    "PreloadableTerms",
    "OneShotTerms",
    "TermsProvider",
    "TermsAgent",
    "ContextWindow",
    "TranslateResult",
    "translate_with_verify",
    # Checker
    "Checker",
    "CheckReport",
    "Severity",
    "default_checker",
    # Pipeline
    "Pipeline",
    "TranslateNodeConfig",
    "PrefixRule",
    "ProgressCallback",
    "EN_ZH_PREFIX_RULES",
    "StreamAdapter",
    "FeedResult",
]
