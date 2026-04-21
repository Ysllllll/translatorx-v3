"""Transcriber Protocol + value objects.

Stage 6 port. A :class:`Transcriber` converts an audio file into a list of
timed :class:`Segment` records (optionally with word-level timings).

Design goals
------------
* **Protocol-first** — multiple backends (local whisperx, OpenAI API,
  self-hosted HTTP) conform to a single contract.
* **Async-first** — transcription is typically I/O or GPU bound; callers
  drive it from async orchestrators. A sync ``transcribe`` wrapper is
  provided by adapters for REPL / test convenience.
* **Segment-level output** — adapters return :class:`Segment` objects that
  downstream components (WhisperXSource-like adapters, AlignProcessor,
  TTSProcessor) consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from domain.model import Segment


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    """Per-call transcription options.

    Args:
        language: ISO language code (``"en"``, ``"zh"``, ...). ``None``
            requests auto-detection when supported by the backend.
        prompt: Optional initial prompt / biasing text (e.g. glossary
            hints for OpenAI Whisper).
        temperature: Sampling temperature. ``0.0`` is greedy.
        word_timestamps: Whether to request per-word timings.
        extra: Backend-specific passthrough options.
    """

    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    word_timestamps: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Transcription output shared across backends.

    Args:
        segments: Ordered list of :class:`Segment`. Each segment carries
            ``text`` + ``start/end`` seconds; ``words`` populated when
            available.
        language: Detected / confirmed language code.
        duration: Input audio duration in seconds (``0.0`` if unknown).
        extra: Backend-specific fields (raw response, model id, ...).
    """

    segments: list[Segment]
    language: str = ""
    duration: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Transcriber(Protocol):
    """Audio → timed segments contract.

    Implementations:
        * :class:`adapters.transcribers.whisperx.WhisperXTranscriber`
        * :class:`adapters.transcribers.openai_api.OpenAiTranscriber`
        * :class:`adapters.transcribers.http_remote.HttpRemoteTranscriber`
    """

    async def transcribe(
        self,
        audio: str | Path,
        opts: TranscribeOptions | None = None,
    ) -> TranscriptionResult:
        """Transcribe ``audio`` into a :class:`TranscriptionResult`."""
        ...


__all__ = [
    "TranscribeOptions",
    "TranscriptionResult",
    "Transcriber",
]
