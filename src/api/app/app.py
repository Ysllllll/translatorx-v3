"""App facade — top-level entry point (extracted from runtime.app).

An :class:`App` owns the parsed :class:`AppConfig` and builds concrete
engines / stores / checkers / contexts on demand, caching them by name
or language-pair.  Builder factories (:meth:`video`, :meth:`course`,
:meth:`stream`) are the main public surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from application.checker import Checker, default_checker
from application.translate import EngineConfig, OpenAICompatEngine, StaticTerms, TranslationContext

from application.config import AppConfig, EngineEntry
from adapters.storage.store import JsonFileStore, Store
from adapters.storage.workspace import Workspace

ApplyFn = Callable[[list[str]], list[list[str]]]


class App:
    """Top-level facade: config + resolver cache + Builder factories."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._engines: dict[str, OpenAICompatEngine] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> App:
        """Load YAML config and construct an :class:`App`."""
        return cls(AppConfig.load(path))

    @classmethod
    def from_yaml(cls, text: str) -> App:
        """Construct from a YAML string (useful for inline demos/tests)."""
        return cls(AppConfig.from_yaml(text))

    @classmethod
    def from_dict(cls, data: dict) -> App:
        """Construct from a plain dict (no YAML needed)."""
        return cls(AppConfig.from_dict(data))

    # -- config access ---------------------------------------------------

    @property
    def config(self) -> AppConfig:
        return self._config

    # -- resolvers -------------------------------------------------------

    def engine(self, name: str = "default") -> OpenAICompatEngine:
        """Return (cached) engine by name."""
        if name not in self._engines:
            entry = self._config.engines.get(name)
            if entry is None:
                raise KeyError(f"no engine configured with name {name!r}")
            self._engines[name] = _build_engine(entry)
        return self._engines[name]

    def context(self, src: str, tgt: str) -> TranslationContext:
        """Return a fresh :class:`TranslationContext` for the given pair."""
        key = f"{src}_{tgt}"
        entry = self._config.contexts.get(key)
        if entry is None:
            return TranslationContext(
                source_lang=src,
                target_lang=tgt,
                terms_provider=StaticTerms({}),
            )
        return TranslationContext(
            source_lang=entry.src,
            target_lang=entry.tgt,
            terms_provider=StaticTerms(dict(entry.terms)),
            window_size=entry.window_size,
            max_retries=entry.max_retries,
            system_prompt_template=entry.system_prompt_template,
        )

    def checker(self, src: str, tgt: str) -> Checker:
        """Return a default :class:`Checker` for the pair."""
        return default_checker(src, tgt)

    def workspace(self, course: str) -> Workspace:
        """Materialize a :class:`Workspace` under the configured store root."""
        root = Path(self._config.store.root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        (root / course).mkdir(parents=True, exist_ok=True)
        return Workspace(root=root, course=course)

    def store(self, course: str) -> Store:
        """Return a :class:`JsonFileStore` bound to *course*."""
        return JsonFileStore(self.workspace(course))

    # -- preprocess factories --------------------------------------------

    def punc_restorer(self) -> ApplyFn | None:
        """Build a punctuation restorer from ``config.preprocess``."""
        cfg = self._config.preprocess
        if cfg.punc_mode == "none":
            return None
        if cfg.punc_mode == "ner":
            from adapters.preprocess import NerPuncRestorer

            return NerPuncRestorer.get_instance()
        if cfg.punc_mode == "llm":
            from adapters.preprocess import LlmPuncRestorer

            engine = self.engine(cfg.punc_engine)
            return LlmPuncRestorer(
                engine,
                threshold=cfg.punc_threshold,
                max_concurrent=cfg.max_concurrent,
                max_retries=cfg.punc_max_retries,
                on_failure=cfg.punc_on_failure,
            )
        if cfg.punc_mode == "remote":
            from adapters.preprocess import RemotePuncRestorer

            if cfg.punc_endpoint is None:
                raise ValueError("preprocess.punc_endpoint required for punc_mode='remote'")
            return RemotePuncRestorer(cfg.punc_endpoint, threshold=cfg.punc_threshold)
        raise ValueError(f"unknown punc_mode: {cfg.punc_mode!r}")

    def chunker(self) -> ApplyFn | None:
        """Build a chunker from ``config.preprocess``.

        - ``"spacy"`` → :class:`SpacySplitter` (NLP-based sentence splitting).
        - ``"llm"`` → :class:`LlmChunker` (recursive binary LLM splitting).
        - ``"spacy_llm"`` → spaCy coarse split, then LLM fine split for
          chunks exceeding ``chunk_len``.
        """
        cfg = self._config.preprocess
        if cfg.chunk_mode == "none":
            return None
        if cfg.chunk_mode == "spacy":
            from adapters.preprocess import SpacySplitter

            return SpacySplitter.get_instance(cfg.spacy_model)
        if cfg.chunk_mode == "llm":
            from adapters.preprocess import LlmChunker

            engine = self.engine(cfg.chunk_engine)
            return LlmChunker(
                engine,
                chunk_len=cfg.chunk_len,
                max_depth=cfg.chunk_max_depth,
                max_retries=cfg.chunk_max_retries,
                on_failure=cfg.chunk_on_failure,
                split_parts=cfg.chunk_split_parts,
                max_concurrent=cfg.max_concurrent,
            )
        if cfg.chunk_mode == "spacy_llm":
            from adapters.preprocess import LlmChunker, SpacySplitter
            from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker

            splitter = SpacySplitter.get_instance(cfg.spacy_model)
            engine = self.engine(cfg.chunk_engine)
            llm = LlmChunker(
                engine,
                chunk_len=cfg.chunk_len,
                max_depth=cfg.chunk_max_depth,
                max_retries=cfg.chunk_max_retries,
                on_failure=cfg.chunk_on_failure,
                split_parts=cfg.chunk_split_parts,
                max_concurrent=cfg.max_concurrent,
            )
            return SpacyLlmChunker(splitter, llm, chunk_len=cfg.chunk_len)
        raise ValueError(f"unknown chunk_mode: {cfg.chunk_mode!r}")

    # -- builders --------------------------------------------------------

    def video(self, *, course: str, video: str) -> "VideoBuilder":
        from api.app.video import VideoBuilder

        return VideoBuilder(app=self, course=course, video=video)

    def course(self, *, course: str) -> "CourseBuilder":
        from api.app.course import CourseBuilder

        return CourseBuilder(app=self, course=course)

    def stream(self, *, course: str, video: str, language: str) -> "StreamBuilder":
        """Builder for live-streaming translation (browser-plugin scenario)."""
        from api.app.stream import StreamBuilder

        return StreamBuilder(app=self, course=course, video=video, language=language)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_engine(entry: EngineEntry) -> OpenAICompatEngine:
    if entry.kind != "openai_compat":
        raise ValueError(f"unsupported engine kind: {entry.kind!r}")
    cfg = EngineConfig(
        model=entry.model,
        base_url=entry.base_url,
        api_key=entry.resolve_api_key(),
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        timeout=entry.timeout,
        extra_body=dict(entry.extra_body),
    )
    return OpenAICompatEngine(cfg)


__all__ = ["App"]
