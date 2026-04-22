"""Pydantic request/response schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


class CreateVideoRequest(BaseModel):
    """Body for ``POST /api/courses/{course}/videos``."""

    model_config = ConfigDict(extra="forbid")

    video: str = Field(..., description="Video identifier (filename stem).")
    src: str | None = Field(None, description="Source language (auto-detected when omitted).")
    tgt: list[str] = Field(..., min_length=1, description="Target languages.")
    source_kind: Literal["srt", "whisperx", "text"] | None = None
    source_path: str | None = Field(
        None,
        description="Server-side path relative to workspace root. Exactly one of source_path / source_content must be provided.",
    )
    source_content: str | None = Field(None, description="Inline SRT content.")
    stages: list[Literal["translate", "align", "tts", "summary"]] = Field(
        default_factory=lambda: ["translate"],
        description="Pipeline stages to run, in order.",
    )
    engine: str = "default"


class VideoState(BaseModel):
    """Public representation of a task's state."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    course: str
    video: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    stages: list[str]
    src: str | None = None
    tgt: list[str] = []
    done: int = 0
    total: int | None = None
    error: str | None = None
    elapsed_s: float | None = None


class VideoList(BaseModel):
    items: list[VideoState]


# ---------------------------------------------------------------------------
# Streams (live)
# ---------------------------------------------------------------------------


class CreateStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course: str
    video: str
    src: str
    tgt: str
    engine: str = "default"


class StreamInfo(BaseModel):
    stream_id: str
    course: str
    video: str
    src: str
    tgt: str
    status: Literal["open", "closing", "closed"]


class StreamSegmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str | None = None


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] | None = None


__all__ = [
    "CreateVideoRequest",
    "VideoState",
    "VideoList",
    "CreateStreamRequest",
    "StreamInfo",
    "StreamSegmentIn",
    "Health",
    "ErrorResponse",
]
