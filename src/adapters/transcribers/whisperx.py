"""WhisperX-based local transcriber.

Backend library: https://github.com/m-bain/whisperX.

Runs on the local machine (GPU recommended). Returns word-level
timestamps when ``word_timestamps=True``. Imports ``whisperx`` lazily so
the module is importable even when the heavy dependency is missing —
callers should guard via :func:`whisperx_is_available`.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domain.model import Segment, Word
from ports.transcriber import TranscribeOptions, TranscriptionResult


def whisperx_is_available() -> bool:
    """Return ``True`` when the ``whisperx`` package can be imported."""
    try:
        importlib.import_module("whisperx")
        return True
    except Exception:
        return False


@dataclass
class WhisperXConfig:
    """WhisperX backend configuration.

    Args:
        model: Model name (``"large-v3"``, ``"medium"``, ...).
        device: ``"cuda"`` / ``"cpu"``. GPU strongly recommended.
        compute_type: CT2 compute type (``"float16"`` / ``"int8"``).
        batch_size: Inference batch size.
        align: Whether to run the alignment stage (word-level timings).
        diarize: Whether to run speaker diarization (requires HF token).
        hf_token: Hugging Face token for pyannote diarization.
        language: Default language if ``TranscribeOptions.language`` is
            unset at call time. ``None`` lets WhisperX auto-detect.
        extra: Backend-specific passthrough — merged into load_model
            kwargs.
    """

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    align: bool = True
    diarize: bool = False
    hf_token: str | None = None
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class WhisperXTranscriber:
    """Local WhisperX-backed :class:`Transcriber`.

    The model and optional alignment / diarize pipelines are loaded on
    first use and cached on the instance. Inference is serialized with a
    per-instance lock so concurrent callers don't stomp on the CUDA
    context.
    """

    def __init__(self, config: WhisperXConfig | None = None) -> None:
        self._config = config or WhisperXConfig()
        self._lock = threading.Lock()
        self._model: Any = None
        self._align_cache: dict[str, tuple[Any, Any]] = {}
        self._diarize_pipeline: Any = None

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: str | Path,
        opts: TranscribeOptions | None = None,
    ) -> TranscriptionResult:
        opts = opts or TranscribeOptions()
        return await asyncio.to_thread(self._transcribe_sync, str(audio), opts)

    # ------------------------------------------------------------------
    # Sync worker
    # ------------------------------------------------------------------

    def _transcribe_sync(self, audio: str, opts: TranscribeOptions) -> TranscriptionResult:
        whisperx = importlib.import_module("whisperx")
        cfg = self._config

        with self._lock:
            if self._model is None:
                self._model = whisperx.load_model(
                    cfg.model,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                    language=opts.language or cfg.language,
                    **cfg.extra,
                )

            audio_data = whisperx.load_audio(audio)
            result = self._model.transcribe(
                audio_data,
                batch_size=cfg.batch_size,
                language=opts.language or cfg.language,
            )
            language = result.get("language") or opts.language or cfg.language or ""

            if cfg.align and opts.word_timestamps and language:
                align_model, metadata = self._get_align_model(whisperx, language, cfg.device)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_data,
                    cfg.device,
                    return_char_alignments=False,
                )

            diarize_segments = None
            if cfg.diarize and cfg.hf_token:
                if self._diarize_pipeline is None:
                    self._diarize_pipeline = whisperx.DiarizationPipeline(
                        use_auth_token=cfg.hf_token,
                        device=cfg.device,
                    )
                diarize_segments = self._diarize_pipeline(audio_data)
                result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = _to_domain_segments(result.get("segments") or [])
        return TranscriptionResult(
            segments=segments,
            language=language,
            duration=float(result.get("duration") or 0.0),
            extra={"raw": result},
        )

    def _get_align_model(self, whisperx: Any, language: str, device: str):
        cached = self._align_cache.get(language)
        if cached is not None:
            return cached
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        self._align_cache[language] = (align_model, metadata)
        return align_model, metadata


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _to_domain_segments(raw_segments: list[dict[str, Any]]) -> list[Segment]:
    out: list[Segment] = []
    for seg in raw_segments:
        words = _to_domain_words(seg.get("words") or [])
        out.append(
            Segment(
                start=float(seg.get("start") or 0.0),
                end=float(seg.get("end") or 0.0),
                text=(seg.get("text") or "").strip(),
                speaker=seg.get("speaker"),
                words=words,
            )
        )
    return out


def _to_domain_words(raw_words: list[dict[str, Any]]) -> list[Word]:
    out: list[Word] = []
    for w in raw_words:
        text = w.get("word") or w.get("text") or ""
        if not text:
            continue
        out.append(
            Word(
                word=text,
                start=float(w.get("start") or 0.0),
                end=float(w.get("end") or 0.0),
                speaker=w.get("speaker"),
            )
        )
    return out


__all__ = ["WhisperXConfig", "WhisperXTranscriber", "whisperx_is_available"]
