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
``checker``, ``runtime``) remain fully accessible for advanced use.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

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

# Runtime (new orchestration layer)
from runtime import (
    EN_ZH_PREFIX_RULES,
    JsonFileStore,
    PrefixRule,
    PushQueueSource,
    SrtSource,
    TranslateNodeConfig,
    TranslateProcessor,
    VideoKey,
    VideoOrchestrator,
    VideoResult,
    WhisperXSource,
    Workspace,
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
    """Create an LLM engine with sensible defaults."""
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
    """Create a translation context."""
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
    workspace_root: Path | str | None = None,
    course: str = "default",
    video: str = "srt",
) -> list[SentenceRecord]:
    """Translate SRT content end-to-end in one call.

    Writes to an ephemeral :class:`Workspace` unless ``workspace_root`` is
    supplied. Returns the translated records in source order.
    """
    with TemporaryDirectory() as tmp:
        srt_path = Path(tmp) / "in.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        ws_root = Path(workspace_root) if workspace_root else Path(tmp) / "ws"
        ws = Workspace(root=ws_root, course=course)
        store = JsonFileStore(ws)

        ctx = create_context(src, tgt, terms=terms)
        checker = default_checker(src, tgt)
        processor = TranslateProcessor(engine, checker, config=config)

        orch = VideoOrchestrator(
            source=SrtSource(srt_path, language=src),
            processors=[processor],
            ctx=ctx,
            store=store,
            video_key=VideoKey(course=course, video=video),
        )
        result = await orch.run()
        return result.records


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
    # Runtime
    "TranslateProcessor",
    "TranslateNodeConfig",
    "PrefixRule",
    "EN_ZH_PREFIX_RULES",
    "VideoOrchestrator",
    "VideoResult",
    "VideoKey",
    "SrtSource",
    "WhisperXSource",
    "PushQueueSource",
    "Workspace",
    "JsonFileStore",
]
