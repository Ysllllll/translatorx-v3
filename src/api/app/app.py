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

    def punc_restorer(self, language: str) -> ApplyFn | None:
        """Build a punctuation restorer :data:`ApplyFn` for *language*.

        Wires ``config.preprocess`` into a :class:`PuncRestorer` with a
        single-language backends map (``{language: {...}}``) and returns
        the language-bound :data:`ApplyFn`. Returns ``None`` when
        ``punc_mode == "none"``.
        """
        cfg = self._config.preprocess
        if cfg.punc_mode == "none":
            return None

        from adapters.preprocess import PuncRestorer

        if cfg.punc_mode == "ner":
            spec: dict[str, object] = {"library": "deepmultilingualpunctuation"}
        elif cfg.punc_mode == "llm":
            spec = {
                "library": "llm",
                "engine": self.engine(cfg.punc_engine),
                "max_retries": cfg.punc_max_retries,
                "max_concurrent": cfg.max_concurrent,
            }
        elif cfg.punc_mode == "remote":
            if cfg.punc_endpoint is None:
                raise ValueError("preprocess.punc_endpoint required for punc_mode='remote'")
            spec = {"library": "remote", "endpoint": cfg.punc_endpoint, "language": language}
        else:
            raise ValueError(f"unknown punc_mode: {cfg.punc_mode!r}")

        restorer = PuncRestorer(
            backends={language: spec},
            threshold=cfg.punc_threshold,
            on_failure=cfg.punc_on_failure,
        )
        return restorer.for_language(language)

    def chunker(self, language: str) -> ApplyFn | None:
        """Build a chunker :data:`ApplyFn` for *language* from config.

        - ``"spacy"`` → :class:`SpacySplitter` (NLP-based sentence splitting,
          language-specific model resolved via
          :meth:`SpacySplitter.for_language`).
        - ``"llm"`` → :class:`LlmChunker` (recursive binary LLM splitting,
          uses :meth:`LangOps.length` / :meth:`LangOps.split_by_length`).
        - ``"spacy_llm"`` → spaCy coarse split, then LLM fine split for
          chunks exceeding ``chunk_len``.
        """
        from domain.lang import LangOps, normalize_language

        cfg = self._config.preprocess
        if cfg.chunk_mode == "none":
            return None

        lang = normalize_language(language)
        ops = LangOps.for_language(lang)

        if cfg.chunk_mode == "spacy":
            from adapters.preprocess import SpacySplitter

            return SpacySplitter.for_language(lang, model=cfg.spacy_model or None)
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
                ops=ops,
            )
        if cfg.chunk_mode == "spacy_llm":
            from adapters.preprocess import LlmChunker, SpacySplitter
            from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker

            splitter = SpacySplitter.for_language(lang, model=cfg.spacy_model or None)
            engine = self.engine(cfg.chunk_engine)
            llm = LlmChunker(
                engine,
                chunk_len=cfg.chunk_len,
                max_depth=cfg.chunk_max_depth,
                max_retries=cfg.chunk_max_retries,
                on_failure=cfg.chunk_on_failure,
                split_parts=cfg.chunk_split_parts,
                max_concurrent=cfg.max_concurrent,
                ops=ops,
            )
            return SpacyLlmChunker(splitter, llm, chunk_len=cfg.chunk_len, ops=ops)
        raise ValueError(f"unknown chunk_mode: {cfg.chunk_mode!r}")

    # -- stage factories (Stage 6/8 — transcribe / tts) -----------------

    def transcriber(self):
        """Build a :class:`Transcriber` from ``config.transcriber``.

        Returns ``None`` when ``library`` is unset.
        """
        cfg = self._config.transcriber
        if not cfg.library:
            return None
        from adapters.transcribers import create as create_transcriber

        spec: dict[str, object] = {"library": cfg.library}
        if cfg.model:
            spec["model"] = cfg.model
        if cfg.base_url:
            spec["base_url"] = cfg.base_url
        if cfg.api_key:
            spec["api_key"] = cfg.api_key
        if cfg.language:
            spec["language"] = cfg.language
        for k, v in cfg.extra.items():
            spec.setdefault(k, v)
        passthrough = cfg.model_dump(exclude={"library", "model", "base_url", "api_key", "language", "word_timestamps", "extra"})
        for k, v in passthrough.items():
            spec.setdefault(k, v)
        return create_transcriber(spec)

    def tts_backend(self):
        """Build a :class:`TTS` backend from ``config.tts``.

        Returns ``None`` when ``library`` is unset.
        """
        cfg = self._config.tts
        if not cfg.library:
            return None
        from adapters.tts import create as create_tts

        spec: dict[str, object] = {"library": cfg.library}
        if cfg.api_key:
            spec["api_key"] = cfg.api_key
        if cfg.base_url:
            spec["base_url"] = cfg.base_url
        for k, v in cfg.extra.items():
            spec.setdefault(k, v)
        passthrough = cfg.model_dump(
            exclude={
                "library",
                "default_voice",
                "format",
                "rate",
                "api_key",
                "base_url",
                "speaker_map",
                "gender_map",
                "extra",
            }
        )
        for k, v in passthrough.items():
            spec.setdefault(k, v)
        return create_tts(spec)

    def voice_picker(self, language: str):
        """Build a :class:`VoicePicker` for *language* from ``config.tts``."""
        from ports.tts import VoicePicker

        cfg = self._config.tts
        return VoicePicker(
            language=language,
            default_voice=cfg.default_voice or None,
            speaker_map=dict(cfg.speaker_map),
            gender_map=dict(cfg.gender_map),
        )

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
