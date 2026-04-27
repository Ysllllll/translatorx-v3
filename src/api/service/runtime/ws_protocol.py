"""WebSocket protocol frames for live streaming (Phase 4 / K).

Defines the **bidirectional** wire format spoken between a browser /
mobile / CLI client and the FastAPI service. Pydantic v2 discriminated
unions give us:

* Compile-time field validation on every frame in/out.
* A single ``type`` field clients can dispatch on without inspecting
  shape.
* Round-trip JSON safety — every frame round-trips through
  ``model_dump_json()`` / ``model_validate_json()``.

Two unions:

* :data:`ClientFrame` — frames the **client** sends to the server.
* :data:`ServerFrame` — frames the **server** sends back.

This module is import-side-effect free and has zero runtime
dependencies beyond Pydantic — the WS endpoint and session module
import these dataclasses. Keeping the protocol here means a
non-Python client (TypeScript, Go, …) can be generated from the same
JSON Schema (``ClientFrame.model_json_schema()`` /
``ServerFrame.model_json_schema()``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Client → Server
# ---------------------------------------------------------------------------


class WsStart(BaseModel):
    """Open a new live stream. Must be the first frame on the
    connection. Server responds with :class:`WsStarted`.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["start"] = "start"
    pipeline: str
    course: str
    video: str
    src: str
    tgt: str
    vars: dict[str, Any] = Field(default_factory=dict)


class WsSegment(BaseModel):
    """Push a transcribed text segment into the live stream. ``seq``
    is monotonic per connection — server uses it to detect gaps.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["segment"] = "segment"
    seq: int
    start: float
    end: float
    text: str
    speaker: str | None = None


class WsAudioChunk(BaseModel):
    """Push a raw audio chunk (base64 encoded). The current K-stage
    does **not** decode audio in the protocol layer — this is a
    transport-only envelope. Stages that need transcription consume
    these chunks themselves; stages that don't will reject the frame
    with :class:`WsError` (``category="unsupported_frame"``).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["audio_chunk"] = "audio_chunk"
    seq: int
    data: str  # base64-encoded bytes
    sample_rate: int | None = None


class WsConfigUpdate(BaseModel):
    """Mid-stream config patch — e.g. swap target language, change
    glossary, tweak overflow policy. Server may reject (``WsError``)
    or apply (no ack frame; the next ``WsFinal`` reflects the change).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["config_update"] = "config_update"
    params: dict[str, Any]


class WsAbort(BaseModel):
    """Cancel the stream. Server flushes outstanding work and replies
    with :class:`WsClosed`.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["abort"] = "abort"


class WsPing(BaseModel):
    """Keep-alive. Server responds with :class:`WsPong`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ping"] = "ping"


ClientFrame = Annotated[
    Union[WsStart, WsSegment, WsAudioChunk, WsConfigUpdate, WsAbort, WsPing],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Server → Client
# ---------------------------------------------------------------------------


class WsStarted(BaseModel):
    """Acknowledges :class:`WsStart`. ``stream_id`` identifies the
    LiveStreamHandle on the server side and can be used in logs /
    later abort calls over a different channel.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["started"] = "started"
    stream_id: str


class WsPartial(BaseModel):
    """Intermediate per-stage output — e.g. punctuation-restored
    source text before translation, or a translation candidate before
    the checker has approved it.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["partial"] = "partial"
    stage: str
    text: str


class WsFinal(BaseModel):
    """A finalised :class:`SentenceRecord` from the live pipeline.
    Frame fields are a flat subset of ``SentenceRecord`` — clients
    that need the full structure should hit the REST endpoint with
    ``record_id``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["final"] = "final"
    record_id: str
    src: str
    tgt: str
    start: float
    end: float


class WsProgress(BaseModel):
    """Back-pressure / progress snapshot. ``channel_fill`` is the
    most-saturated channel in the pipeline (``filled / capacity``);
    clients can use it to slow down the producer side.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["progress"] = "progress"
    stage: str
    channel_fill: float | None = None


class WsError(BaseModel):
    """Recoverable or fatal error. ``retry_after`` (seconds) is set
    when the client may retry the offending frame after a back-off.
    ``category`` mirrors :class:`ports.errors.ErrorCategory` plus
    protocol-only categories like ``unsupported_frame`` /
    ``invalid_frame`` / ``unauthorised``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    category: str
    message: str
    retry_after: float | None = None


class WsClosed(BaseModel):
    """Server has shut down the stream. ``reason`` is one of
    ``"client_abort"`` / ``"server_shutdown"`` / ``"completed"`` /
    ``"error"``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["closed"] = "closed"
    reason: str


class WsPong(BaseModel):
    """Reply to :class:`WsPing`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pong"] = "pong"


ServerFrame = Annotated[
    Union[WsStarted, WsPartial, WsFinal, WsProgress, WsError, WsClosed, WsPong],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Helpers — round-trip safe parse/serialise.
# ---------------------------------------------------------------------------


from pydantic import TypeAdapter

_CLIENT_ADAPTER = TypeAdapter(ClientFrame)
_SERVER_ADAPTER = TypeAdapter(ServerFrame)


def parse_client_frame(raw: str | bytes | dict) -> ClientFrame:
    """Parse a raw payload into a discriminated :data:`ClientFrame`.

    Accepts JSON ``str`` / ``bytes`` (validated via
    ``model_validate_json``) or already-parsed ``dict``. Raises
    ``pydantic.ValidationError`` on shape mismatch — the WS endpoint
    catches this and replies with :class:`WsError`
    (``category="invalid_frame"``).
    """

    if isinstance(raw, (str, bytes)):
        return _CLIENT_ADAPTER.validate_json(raw)
    return _CLIENT_ADAPTER.validate_python(raw)


def parse_server_frame(raw: str | bytes | dict) -> ServerFrame:
    """Parse a raw payload into a discriminated :data:`ServerFrame`.

    Mainly useful for tests / client SDKs that consume the same wire
    format.
    """

    if isinstance(raw, (str, bytes)):
        return _SERVER_ADAPTER.validate_json(raw)
    return _SERVER_ADAPTER.validate_python(raw)


def dump_frame(frame: BaseModel) -> str:
    """Serialise any frame to a JSON ``str`` ready for
    :meth:`fastapi.WebSocket.send_text`.
    """

    return frame.model_dump_json()


__all__ = [
    "ClientFrame",
    "ServerFrame",
    "WsStart",
    "WsSegment",
    "WsAudioChunk",
    "WsConfigUpdate",
    "WsAbort",
    "WsPing",
    "WsStarted",
    "WsPartial",
    "WsFinal",
    "WsProgress",
    "WsError",
    "WsClosed",
    "WsPong",
    "parse_client_frame",
    "parse_server_frame",
    "dump_frame",
]
