"""Tests for the ``remote`` chunk backend."""

from __future__ import annotations

import json

import httpx
import pytest

from adapters.preprocess.chunk.backends.remote import remote_backend
from adapters.preprocess.chunk.registry import ChunkBackendRegistry


def _handler_factory(responses: list[httpx.Response]):
    """Return a MockTransport handler that pops one response per call."""
    idx = {"i": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert "text" in payload
        assert "language" in payload
        assert "split_parts" in payload
        resp = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return resp

    return _handler


def _patch_transport(monkeypatch, transport):
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs.pop("transport", None)
        original(self, *args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)


class TestRemoteChunkBackend:
    def test_registered(self):
        assert ChunkBackendRegistry.is_registered("remote")

    def test_batch_success(self, monkeypatch):
        source = "Hello world, this is a test."
        responses = [httpx.Response(200, json={"parts": ["Hello world,", "this is a test."]})]
        _patch_transport(monkeypatch, httpx.MockTransport(_handler_factory(responses)))

        backend = remote_backend(endpoint="https://api.test/chunk", language="en")
        out = backend([source])
        assert out == [["Hello world,", "this is a test."]]

    def test_retries_on_http_error(self, monkeypatch):
        source = "Hello world, this is a test."
        responses = [httpx.Response(500, json={"error": "oops"}), httpx.Response(200, json={"parts": ["Hello world,", "this is a test."]})]
        _patch_transport(monkeypatch, httpx.MockTransport(_handler_factory(responses)))

        backend = remote_backend(endpoint="https://api.test/chunk", language="en", max_retries=2)
        out = backend([source])
        assert out == [["Hello world,", "this is a test."]]

    def test_raises_after_exhausting_retries(self, monkeypatch):
        source = "Hello world, this is a test."
        responses = [httpx.Response(500, json={"error": "oops"})]
        _patch_transport(monkeypatch, httpx.MockTransport(_handler_factory(responses)))

        backend = remote_backend(endpoint="https://api.test/chunk", language="en", max_retries=1)
        with pytest.raises(RuntimeError, match="Remote chunk failed"):
            backend([source])

    def test_recover_pair_salvages_partial_response(self, monkeypatch):
        """When split_parts==2 and only first half returned, recovery derives the rest."""
        source = "Hello world, this is a test."
        # Backend returns only one line — should trigger 2-piece recovery.
        responses = [httpx.Response(200, json={"parts": ["Hello world,"]})]
        _patch_transport(monkeypatch, httpx.MockTransport(_handler_factory(responses)))

        backend = remote_backend(endpoint="https://api.test/chunk", language="en")
        out = backend([source])
        assert len(out) == 1 and len(out[0]) == 2
        # Reconstruction must hold.
        from adapters.preprocess.chunk.reconstruct import chunks_match_source

        assert chunks_match_source(out[0], source, language="en")

    def test_reconstruction_mismatch_retries(self, monkeypatch):
        source = "Hello world, this is a test."
        responses = [httpx.Response(200, json={"parts": ["Totally", "unrelated"]}), httpx.Response(200, json={"parts": ["Hello world,", "this is a test."]})]
        _patch_transport(monkeypatch, httpx.MockTransport(_handler_factory(responses)))

        backend = remote_backend(endpoint="https://api.test/chunk", language="en", max_retries=2)
        out = backend([source])
        assert out == [["Hello world,", "this is a test."]]

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError, match="split_parts"):
            remote_backend(endpoint="x", language="en", split_parts=1)
        with pytest.raises(ValueError, match="max_retries"):
            remote_backend(endpoint="x", language="en", max_retries=-1)
        with pytest.raises(ValueError, match="transport_retries"):
            remote_backend(endpoint="x", language="en", transport_retries=-1)
