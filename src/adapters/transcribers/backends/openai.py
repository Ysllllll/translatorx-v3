"""OpenAI-compatible Whisper transcription API adapter.

Works with ``api.openai.com/v1/audio/transcriptions`` and any self-hosted
OpenAI-compatible server (e.g. groq, faster-whisper-server). Uses
``httpx`` for async I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from domain.model import Segment, Word
from ports.transcriber import TranscribeOptions, TranscriptionResult

from adapters.transcribers.registry import DEFAULT_REGISTRY as _registry

_register_backend = _registry.register


@dataclass
class OpenAiTranscriberConfig:
    """OpenAI Whisper API configuration.

    Args:
        base_url: Base URL of the OpenAI-compatible server (without the
            trailing ``/audio/transcriptions`` path).
        api_key: API key (sent as ``Authorization: Bearer`` header).
        model: Model id (``"whisper-1"`` for OpenAI; backend-specific
            otherwise).
        response_format: ``"verbose_json"`` is required for
            ``word``/``segment`` timestamp granularity.
        timeout: Per-request timeout in seconds.
        extra: Backend-specific form fields merged into every request.
    """

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "whisper-1"
    response_format: str = "verbose_json"
    timeout: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)


class OpenAiTranscriber:
    """OpenAI-compatible Whisper API :class:`Transcriber`."""

    def __init__(self, config: OpenAiTranscriberConfig | None = None) -> None:
        self._config = config or OpenAiTranscriberConfig()

    async def transcribe(
        self,
        audio: str | Path,
        opts: TranscribeOptions | None = None,
    ) -> TranscriptionResult:
        opts = opts or TranscribeOptions()
        cfg = self._config
        path = Path(audio)

        granularities = []
        if opts.word_timestamps:
            granularities.append("word")
        granularities.append("segment")

        data: dict[str, Any] = {
            "model": cfg.model,
            "response_format": cfg.response_format,
            "temperature": str(opts.temperature),
        }
        for g in granularities:
            # httpx serializes list values as repeated fields.
            data.setdefault("timestamp_granularities[]", []).append(g)
        if opts.language:
            data["language"] = opts.language
        if opts.prompt:
            data["prompt"] = opts.prompt
        data.update(cfg.extra)

        url = cfg.base_url.rstrip("/") + "/audio/transcriptions"
        headers: dict[str, str] = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            with path.open("rb") as fh:
                files = {"file": (path.name, fh, "application/octet-stream")}
                response = await client.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            payload = response.json()

        return _parse_openai_response(payload, fallback_language=opts.language or "")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_openai_response(payload: dict[str, Any], *, fallback_language: str) -> TranscriptionResult:
    language = payload.get("language") or fallback_language
    duration = float(payload.get("duration") or 0.0)

    raw_words = payload.get("words") or []
    word_objs = _words_to_domain(raw_words)

    raw_segments = payload.get("segments") or []
    segments: list[Segment] = []
    for seg in raw_segments:
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or 0.0)
        seg_words = [w for w in word_objs if w.start >= seg_start and w.end <= seg_end + 1e-6]
        segments.append(
            Segment(
                start=seg_start,
                end=seg_end,
                text=(seg.get("text") or "").strip(),
                words=seg_words,
            )
        )

    if not segments and payload.get("text"):
        segments = [Segment(start=0.0, end=duration, text=str(payload["text"]).strip(), words=word_objs)]

    return TranscriptionResult(
        segments=segments,
        language=language,
        duration=duration,
        extra={"raw": payload},
    )


def _words_to_domain(raw_words: list[dict[str, Any]]) -> list[Word]:
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
            )
        )
    return out


__all__ = [
    "OpenAiTranscriberConfig",
    "OpenAiTranscriber",
    "openai_backend",
]


@_register_backend("openai")
def openai_backend(**params: Any) -> "OpenAiTranscriber":
    """Factory for the ``openai`` Whisper-compatible transcriber."""
    cfg_fields = set(OpenAiTranscriberConfig.__dataclass_fields__.keys())
    cfg_kw = {k: v for k, v in params.items() if k in cfg_fields}
    return OpenAiTranscriber(OpenAiTranscriberConfig(**cfg_kw))
