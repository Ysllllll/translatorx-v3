"""Phase 4 (K1) — WebSocket protocol frame round-trip tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.service.runtime.ws_protocol import WsAbort, WsAudioChunk, WsClosed, WsConfigUpdate, WsError, WsFinal, WsPartial, WsPing, WsPong, WsProgress, WsSegment, WsStart, WsStarted, dump_frame, parse_client_frame, parse_server_frame


class TestClientFrameRoundTrip:
    def test_start_minimal(self):
        f = parse_client_frame('{"type":"start","pipeline":"p","course":"c","video":"v","src":"en","tgt":"zh"}')
        assert isinstance(f, WsStart)
        assert f.pipeline == "p"
        assert f.vars == {}

    def test_start_with_vars(self):
        raw = WsStart(pipeline="p", course="c", video="v", src="en", tgt="zh", vars={"tenant": "acme"})
        rt = parse_client_frame(dump_frame(raw))
        assert isinstance(rt, WsStart)
        assert rt.vars == {"tenant": "acme"}

    def test_segment(self):
        raw = WsSegment(seq=3, start=1.5, end=2.5, text="hello", speaker="S1")
        rt = parse_client_frame(dump_frame(raw))
        assert isinstance(rt, WsSegment)
        assert rt.seq == 3
        assert rt.speaker == "S1"

    def test_audio_chunk(self):
        raw = WsAudioChunk(seq=1, data="Zm9v", sample_rate=16000)
        rt = parse_client_frame(dump_frame(raw))
        assert isinstance(rt, WsAudioChunk)
        assert rt.data == "Zm9v"

    def test_config_update(self):
        rt = parse_client_frame('{"type":"config_update","params":{"tgt":"ja"}}')
        assert isinstance(rt, WsConfigUpdate)
        assert rt.params == {"tgt": "ja"}

    def test_abort(self):
        rt = parse_client_frame('{"type":"abort"}')
        assert isinstance(rt, WsAbort)

    def test_ping(self):
        rt = parse_client_frame('{"type":"ping"}')
        assert isinstance(rt, WsPing)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            parse_client_frame('{"type":"nope"}')

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            parse_client_frame('{"type":"abort","extra":1}')

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            parse_client_frame('{"type":"start","pipeline":"p"}')

    def test_dict_input_accepted(self):
        rt = parse_client_frame({"type": "ping"})
        assert isinstance(rt, WsPing)


class TestServerFrameRoundTrip:
    def test_started(self):
        rt = parse_server_frame(dump_frame(WsStarted(stream_id="abc")))
        assert isinstance(rt, WsStarted)
        assert rt.stream_id == "abc"

    def test_partial(self):
        rt = parse_server_frame(dump_frame(WsPartial(stage="translate", text="你好")))
        assert isinstance(rt, WsPartial)

    def test_final(self):
        rt = parse_server_frame(dump_frame(WsFinal(record_id="r1", src="hello", tgt="你好", start=0.0, end=1.0)))
        assert isinstance(rt, WsFinal)
        assert rt.tgt == "你好"

    def test_progress_optional_fill(self):
        rt = parse_server_frame(dump_frame(WsProgress(stage="translate")))
        assert isinstance(rt, WsProgress)
        assert rt.channel_fill is None

    def test_error_with_retry(self):
        rt = parse_server_frame(dump_frame(WsError(category="rate_limited", message="slow down", retry_after=2.5)))
        assert isinstance(rt, WsError)
        assert rt.retry_after == 2.5

    def test_closed(self):
        rt = parse_server_frame(dump_frame(WsClosed(reason="completed")))
        assert isinstance(rt, WsClosed)

    def test_pong(self):
        rt = parse_server_frame(dump_frame(WsPong()))
        assert isinstance(rt, WsPong)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            parse_server_frame('{"type":"pong","extra":1}')


class TestProtocolJsonSchema:
    def test_client_frame_schema_lists_all_types(self):
        from pydantic import TypeAdapter

        from api.service.runtime.ws_protocol import ClientFrame

        schema = TypeAdapter(ClientFrame).json_schema()
        # Discriminated union exposes the discriminator mapping
        assert "discriminator" in schema or "oneOf" in schema or "anyOf" in schema

    def test_server_frame_schema_lists_all_types(self):
        from pydantic import TypeAdapter

        from api.service.runtime.ws_protocol import ServerFrame

        schema = TypeAdapter(ServerFrame).json_schema()
        assert "discriminator" in schema or "oneOf" in schema or "anyOf" in schema
