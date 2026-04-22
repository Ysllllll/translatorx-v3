"""Tests for the ``remote`` punc backend."""

from __future__ import annotations

import json

import httpx
import pytest

from adapters.preprocess.punc.backends.remote import factory as remote_factory
from adapters.preprocess.punc.registry import PuncBackendRegistry


def _handler_factory(responses: list[httpx.Response]):
    """Return a MockTransport handler that pops one response per call."""
    idx = {"i": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        # Assert shape of payload.
        assert "text" in payload
        resp = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return resp

    return _handler


class TestRemoteBackend:
    def test_registered(self):
        assert PuncBackendRegistry.is_registered("remote")

    def test_batch_success(self, monkeypatch):
        responses = [httpx.Response(200, json={"result": "hello world."}), httpx.Response(200, json={"result": "goodbye moon."})]
        transport = httpx.MockTransport(_handler_factory(responses))

        original = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("transport", None)
            original(self, *args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched_init)

        backend = remote_factory(endpoint="https://api.test/punc", language="en")
        out = backend(["hello world", "goodbye moon"])
        assert out == ["hello world.", "goodbye moon."]

    def test_retries_on_http_error(self, monkeypatch):
        responses = [httpx.Response(500, json={"error": "oops"}), httpx.Response(200, json={"result": "hello world."})]
        transport = httpx.MockTransport(_handler_factory(responses))
        original = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("transport", None)
            original(self, *args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched_init)

        backend = remote_factory(endpoint="https://api.test/punc", max_retries=2)
        out = backend(["hello world"])
        assert out == ["hello world."]

    def test_raises_after_exhaustion(self, monkeypatch):
        responses = [httpx.Response(500)] * 5
        transport = httpx.MockTransport(_handler_factory(responses))
        original = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("transport", None)
            original(self, *args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched_init)

        backend = remote_factory(endpoint="https://api.test/punc", max_retries=1)
        with pytest.raises(RuntimeError, match="Remote punc failed"):
            backend(["hello world"])

    def test_retries_on_content_mismatch(self, monkeypatch):
        # First response mutates words (rejected), second returns clean output.
        responses = [httpx.Response(200, json={"result": "hello world extra."}), httpx.Response(200, json={"result": "hello world."})]
        transport = httpx.MockTransport(_handler_factory(responses))
        original = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("transport", None)
            original(self, *args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched_init)

        backend = remote_factory(endpoint="https://api.test/punc", max_retries=2)
        out = backend(["hello world"])
        assert out == ["hello world."]

    def test_raises_after_content_mismatch_exhausted(self, monkeypatch):
        responses = [httpx.Response(200, json={"result": "bye bye."})] * 5
        transport = httpx.MockTransport(_handler_factory(responses))
        original = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("transport", None)
            original(self, *args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched_init)

        backend = remote_factory(endpoint="https://api.test/punc", max_retries=1)
        with pytest.raises(RuntimeError, match="Remote punc failed"):
            backend(["hello world"])

    def test_rejects_invalid_max_retries(self):
        with pytest.raises(ValueError):
            remote_factory(endpoint="https://api.test/punc", max_retries=-1)
