"""HTTP remote transcriber — talks to a self-hosted WhisperX-style service.

Assumed service contract (POST multipart/form-data):

    POST <base_url>/transcribe
    Fields:
      file: <audio bytes>
      language: <code or "">
      word_timestamps: "true" / "false"
      prompt: <optional>
    Response (application/json):
      {
        "segments": [{"start": float, "end": float, "text": str,
                      "speaker": str | null,
                      "words": [{"word": str, "start": float, "end": float, "speaker": str | null}, ...]}, ...],
        "language": "<code>",
        "duration": float
      }

Any server that matches this shape works. Override ``endpoint`` for
alternative routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from domain.model import Segment, Word
from ports.transcriber import TranscribeOptions, TranscriptionResult


@dataclass
class HttpRemoteConfig:
    """Self-hosted HTTP transcription server configuration.

    Args:
        base_url: Base URL (``"http://localhost:9000"``).
        endpoint: Request path — default ``"/transcribe"``.
        api_key: Optional bearer token.
        timeout: Per-request timeout in seconds.
        extra_headers: Additional headers merged into every request.
        extra_fields: Additional multipart form fields.
    """

    base_url: str = ""
    endpoint: str = "/transcribe"
    api_key: str = ""
    timeout: float = 600.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class HttpRemoteTranscriber:
    """HTTP-based remote :class:`Transcriber`."""

    def __init__(self, config: HttpRemoteConfig) -> None:
        if not config.base_url:
            raise ValueError("HttpRemoteTranscriber requires a non-empty base_url.")
        self._config = config

    async def transcribe(
        self,
        audio: str | Path,
        opts: TranscribeOptions | None = None,
    ) -> TranscriptionResult:
        opts = opts or TranscribeOptions()
        cfg = self._config
        path = Path(audio)

        data: dict[str, Any] = {
            "language": opts.language or "",
            "word_timestamps": "true" if opts.word_timestamps else "false",
            "temperature": str(opts.temperature),
        }
        if opts.prompt:
            data["prompt"] = opts.prompt
        data.update(cfg.extra_fields)

        headers = dict(cfg.extra_headers)
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        url = cfg.base_url.rstrip("/") + cfg.endpoint
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            with path.open("rb") as fh:
                files = {"file": (path.name, fh, "application/octet-stream")}
                response = await client.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            payload = response.json()

        return _parse_http_response(payload, fallback_language=opts.language or "")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_http_response(payload: dict[str, Any], *, fallback_language: str) -> TranscriptionResult:
    raw_segments = payload.get("segments") or []
    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(
            Segment(
                start=float(seg.get("start") or 0.0),
                end=float(seg.get("end") or 0.0),
                text=(seg.get("text") or "").strip(),
                speaker=seg.get("speaker"),
                words=[
                    Word(
                        word=w.get("word") or w.get("text") or "",
                        start=float(w.get("start") or 0.0),
                        end=float(w.get("end") or 0.0),
                        speaker=w.get("speaker"),
                    )
                    for w in (seg.get("words") or [])
                    if (w.get("word") or w.get("text"))
                ],
            )
        )

    return TranscriptionResult(
        segments=segments,
        language=payload.get("language") or fallback_language,
        duration=float(payload.get("duration") or 0.0),
        extra={"raw": payload},
    )


__all__ = ["HttpRemoteConfig", "HttpRemoteTranscriber"]
