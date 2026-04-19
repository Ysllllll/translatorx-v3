"""VideoBuilder — immutable per-video builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model import SentenceRecord, Segment

from runtime.orchestrator import VideoOrchestrator, VideoResult
from runtime.processors.summary import SummaryProcessor
from runtime.processors.translate import TranslateProcessor
from runtime.protocol import Source, VideoKey
from runtime.sources.srt import SrtSource
from runtime.sources.whisperx import WhisperXSource

if TYPE_CHECKING:
    from app._app import App
    from runtime.errors import ErrorReporter


@dataclass(frozen=True)
class _TranslateStage:
    src: str
    tgt: tuple[str, ...]
    engine_name: str = "default"


@dataclass(frozen=True)
class _SummaryStage:
    engine_name: str = "default"
    window_words: int = 4500


def _detect_source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".json":
        return "whisperx"
    raise ValueError(
        f"cannot auto-detect source kind for {path!r} (suffix={suffix!r}); "
        "pass kind= explicitly"
    )


def _detect_language_from_file(path: Path, kind: str) -> str:
    """Auto-detect source language from file content."""
    from lang_ops import detect_language

    if kind == "srt":
        from subtitle.io import read_srt
        segments = read_srt(path)
        sample = " ".join(s.text for s in segments[:20])
    elif kind == "whisperx":
        from subtitle.io import read_whisperx
        words = read_whisperx(path)
        sample = " ".join(w.word for w in words[:100])
    else:
        raise ValueError(f"cannot auto-detect language for kind={kind!r}")
    return detect_language(sample)


@dataclass(frozen=True)
class VideoBuilder:
    """Immutable per-video builder. Each method returns a new instance."""

    app: App
    course: str
    video: str
    _source: Source | None = None
    _source_language: str | None = None
    _translate: _TranslateStage | None = None
    _summary: _SummaryStage | None = None
    _error_reporter: ErrorReporter | None = None

    def source(
        self, path: str | Path, *, language: str | None = None, kind: str | None = None
    ) -> VideoBuilder:
        """Attach a file-based :class:`Source`.

        ``kind`` auto-detects from the file extension: ``.srt`` → srt,
        ``.json`` → whisperx. Pass explicitly to override.

        ``language`` is auto-detected from file content when omitted.
        """
        p = Path(path)
        resolved = kind or _detect_source_kind(p)
        detected_lang = language or _detect_language_from_file(p, resolved)
        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        cfg = self.app.config.preprocess
        restore_punc = self.app.punc_restorer()
        chunk_fn = self.app.chunker()
        preprocess_kw = dict(
            restore_punc=restore_punc,
            punc_position=cfg.punc_position,
            chunk_llm=chunk_fn,
            merge_under=cfg.merge_under,
            max_len=cfg.max_len,
        )
        if resolved == "srt":
            src: Source = SrtSource(
                p, language=detected_lang, store=store, video_key=video_key,
                **preprocess_kw,
            )
        elif resolved == "whisperx":
            src = WhisperXSource(
                p, language=detected_lang, store=store, video_key=video_key,
                **preprocess_kw,
            )
        else:
            raise ValueError(f"unknown source kind: {resolved!r}")
        return replace(self, _source=src, _source_language=detected_lang)

    def translate(
        self,
        *,
        tgt: str | tuple[str, ...],
        src: str | None = None,
        engine: str = "default",
    ) -> VideoBuilder:
        """Set the translation target(s).

        ``src`` is inferred from the attached source's language when omitted.
        ``tgt`` accepts a single language or a tuple of languages.
        """
        resolved_tgt = (tgt,) if isinstance(tgt, str) else tuple(tgt)
        return replace(
            self, _translate=_TranslateStage(
                src=src or "",  # resolved at run() time
                tgt=resolved_tgt,
                engine_name=engine,
            )
        )

    def summary(
        self,
        *,
        engine: str = "default",
        window_words: int = 4500,
    ) -> VideoBuilder:
        """Attach an incremental :class:`SummaryProcessor` before translate."""
        return replace(
            self,
            _summary=_SummaryStage(engine_name=engine, window_words=window_words),
        )

    def with_error_reporter(self, reporter: "ErrorReporter") -> VideoBuilder:
        return replace(self, _error_reporter=reporter)

    async def run(self) -> VideoResult:
        if self._source is None:
            raise ValueError("VideoBuilder.run() requires .source(...) first")
        if self._translate is None:
            raise ValueError("VideoBuilder.run() requires .translate(...) stage")

        t = self._translate
        src_lang = t.src or self._source_language
        if not src_lang:
            raise ValueError(
                "source language unknown; pass language= to .source() or src= to .translate()"
            )

        result: VideoResult | None = None

        for tgt_lang in t.tgt:
            engine = self.app.engine(t.engine_name)
            ctx = self.app.context(src_lang, tgt_lang)
            checker = self.app.checker(src_lang, tgt_lang)
            store = self.app.store(self.course)

            processor = TranslateProcessor(
                engine,
                checker,
                flush_every=self.app.config.runtime.flush_every,
            )
            procs: list[Any] = []
            if self._summary is not None:
                sum_engine = self.app.engine(self._summary.engine_name)
                procs.append(
                    SummaryProcessor(
                        sum_engine,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        window_words=self._summary.window_words,
                    )
                )
            procs.append(processor)
            orch = VideoOrchestrator(
                source=self._source,
                processors=procs,
                ctx=ctx,
                store=store,
                video_key=VideoKey(course=self.course, video=self.video),
                error_reporter=self._error_reporter,
            )
            result = await orch.run()

        assert result is not None  # at least one tgt guaranteed
        return result


__all__ = ["VideoBuilder"]
