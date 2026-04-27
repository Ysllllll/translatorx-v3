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
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domain.model import Segment, Word
from ports.transcriber import TranscribeOptions, TranscriptionResult

from adapters.transcribers.registry import DEFAULT_REGISTRY as _register_backend_src

_register_backend = _register_backend_src.register


def whisperx_is_available() -> bool:
    """Return ``True`` when the ``whisperx`` package can be imported."""
    try:
        importlib.import_module("whisperx")
        return True
    except Exception:
        return False


# C30 — process-wide cache so multiple WhisperXTranscriber instances
# created with identical (model, device, compute_type, language) tuples
# share a single GPU-resident model rather than each loading their own
# copy. ``_LOAD_LOCK`` serialises load_model() calls themselves so two
# concurrent first-callers don't both trigger the heavy load.
_LOAD_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple, Any] = {}


def _shared_load_model(
    whisperx: Any,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    extra: dict[str, Any],
) -> Any:
    key = (model, device, compute_type, language, tuple(sorted(extra.items())))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        loaded = whisperx.load_model(
            model,
            device=device,
            compute_type=compute_type,
            language=language,
            **extra,
        )
        _MODEL_CACHE[key] = loaded
        return loaded


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
    align_cache_size: int = 2
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
        # R12 fix: bounded LRU keyed by language so a long-running
        # process that visits many languages does not hold every
        # alignment model on the GPU forever. Eviction calls release()
        # on the model when supported.
        self._align_cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
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

    async def aclose(self) -> None:
        """Release per-instance model references and align cache.

        The process-wide ``_MODEL_CACHE`` is intentionally preserved so
        peer transcribers and subsequent App lifecycles can re-use loaded
        weights; per-instance state (alignment LRU, diarize pipeline) is
        dropped here.
        """

        def _release() -> None:
            with self._lock:
                for _, (model, _meta) in list(self._align_cache.items()):
                    rel = getattr(model, "release", None)
                    if callable(rel):
                        try:
                            rel()
                        except Exception:
                            pass
                self._align_cache.clear()
                self._diarize_pipeline = None
                self._model = None

        await asyncio.to_thread(_release)

    # ------------------------------------------------------------------
    # Sync worker
    # ------------------------------------------------------------------

    def _transcribe_sync(self, audio: str, opts: TranscribeOptions) -> TranscriptionResult:
        whisperx = importlib.import_module("whisperx")
        cfg = self._config

        with self._lock:
            if self._model is None:
                self._model = _shared_load_model(
                    whisperx,
                    model=cfg.model,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                    language=opts.language or cfg.language,
                    extra=cfg.extra,
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
            self._align_cache.move_to_end(language)
            return cached
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        self._align_cache[language] = (align_model, metadata)
        # Evict LRU entries to keep VRAM bounded.
        max_size = max(1, self._config.align_cache_size)
        while len(self._align_cache) > max_size:
            _, (evicted_model, _) = self._align_cache.popitem(last=False)
            try:
                if hasattr(evicted_model, "to"):
                    evicted_model.to("cpu")
            except Exception:
                pass
            del evicted_model
        return align_model, metadata

    def release(self) -> None:
        """Drop cached models (alignment + ASR + diarize) to free VRAM."""
        with self._lock:
            self._align_cache.clear()
            self._model = None
            self._diarize_pipeline = None


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _to_domain_segments(raw_segments: list[dict[str, Any]]) -> list[Segment]:
    out: list[Segment] = []
    for seg in raw_segments:
        words = _to_domain_words(seg.get("words") or [])
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        # C9 — drop pathological segments produced by alignment failures
        # (NaN / negative duration / collapsed). They corrupt downstream
        # word-timing alignment and Subtitle.from_words.
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        out.append(
            Segment(
                start=start,
                end=end,
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
        start = float(w.get("start") or 0.0)
        end = float(w.get("end") or 0.0)
        # C9 — same guard as segments. WhisperX occasionally emits
        # zero-duration or NaN word frames when the alignment model
        # cannot lock onto a span.
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        out.append(
            Word(
                word=text,
                start=start,
                end=end,
                speaker=w.get("speaker"),
            )
        )
    return out


__all__ = [
    "WhisperXConfig",
    "WhisperXTranscriber",
    "whisperx_backend",
    "whisperx_is_available",
]


@_register_backend("whisperx")
def whisperx_backend(**params: Any) -> "WhisperXTranscriber":
    """Factory for the ``whisperx`` transcriber backend.

    Keyword arguments matching :class:`WhisperXConfig` fields are
    consumed directly; anything else is merged into ``extra``.
    """
    cfg_fields = set(WhisperXConfig.__dataclass_fields__.keys())
    cfg_kw = {k: v for k, v in params.items() if k in cfg_fields}
    leftover = {k: v for k, v in params.items() if k not in cfg_fields}
    if leftover:
        cfg_kw.setdefault("extra", {}).update(leftover)
    return WhisperXTranscriber(WhisperXConfig(**cfg_kw))
