"""trx — TranslatorX unified API (slim facade).

Three factory helpers for common cases, plus a handful of core type
re-exports for ergonomics. For everything else import directly from
``domain``, ``ports``, ``adapters``, ``application``, or ``api.app``.

Quick start::

    from api import trx

    engine = trx.create_engine(model="Qwen/Qwen3-32B", base_url="http://localhost:26592/v1")
    result = await trx.translate_srt(srt_content, engine, src="en", tgt="zh")
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine
from application.processors.translate import TranslateProcessor
from application.processors.prefix import TranslateNodeConfig
from adapters.sources.srt import SrtSource
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.checker import default_checker
from application.orchestrator.video import VideoOrchestrator
from application.translate.context import StaticTerms, TermsProvider, TranslationContext
from domain.model import SentenceRecord, Segment, Word
from domain.subtitle import Subtitle
from ports.engine import LLMEngine
from ports.source import VideoKey

from api.app import App


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
    """Translate SRT content end-to-end in one call."""
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
    "create_engine",
    "create_context",
    "translate_srt",
    # Core types for convenience
    "App",
    "Word",
    "Segment",
    "SentenceRecord",
    "Subtitle",
    "LLMEngine",
]
