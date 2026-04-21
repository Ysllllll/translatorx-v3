"""Tests for RemotePuncRestorer."""

from __future__ import annotations

import json

import httpx
import pytest

from adapters.preprocess import RemotePuncRestorer


class TestRemotePuncRestorer:
    def test_basic_call(self, httpx_mock) -> None:
        httpx_mock.add_response(
            json={"results": [["Hello world."]]},
        )
        restorer = RemotePuncRestorer("http://localhost:8080/restore")
        result = restorer(["hello world"])
        assert result == [["Hello world."]]

    def test_batch_call(self, httpx_mock) -> None:
        httpx_mock.add_response(
            json={"results": [["First."], ["Second."]]},
        )
        restorer = RemotePuncRestorer("http://localhost:8080/restore")
        result = restorer(["first", "second"])
        assert len(result) == 2

    def test_threshold_skips_short(self, httpx_mock) -> None:
        httpx_mock.add_response(
            json={"results": [["Long enough text."]]},
        )
        restorer = RemotePuncRestorer(
            "http://localhost:8080/restore",
            threshold=10,
        )
        result = restorer(["hi", "long enough text"])
        assert result[0] == ["hi"]  # Not sent to remote
        assert result[1] == ["Long enough text."]

    def test_all_short_no_http(self) -> None:
        # No httpx_mock needed — no HTTP call should happen
        restorer = RemotePuncRestorer(
            "http://localhost:8080/restore",
            threshold=100,
        )
        result = restorer(["short", "also short"])
        assert result == [["short"], ["also short"]]

    def test_http_error_raises(self, httpx_mock) -> None:
        httpx_mock.add_response(status_code=500)
        restorer = RemotePuncRestorer("http://localhost:8080/restore")
        with pytest.raises(httpx.HTTPStatusError):
            restorer(["test"])


@pytest.fixture
def httpx_mock(monkeypatch):
    """Minimal httpx mock for testing."""
    return _HttpxMock(monkeypatch)


class _HttpxMock:
    def __init__(self, monkeypatch) -> None:
        self._monkeypatch = monkeypatch
        self._responses: list[dict] = []

    def add_response(self, *, json: dict | None = None, status_code: int = 200) -> None:
        self._responses.append({"json": json, "status_code": status_code})
        self._install()

    def _install(self) -> None:
        responses = list(self._responses)

        class _FakeResponse:
            def __init__(self, resp_def):
                self.status_code = resp_def["status_code"]
                self._json = resp_def.get("json")

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {self.status_code}",
                        request=httpx.Request("POST", "http://test"),
                        response=httpx.Response(self.status_code),
                    )

            def json(self):
                return self._json

        class _FakeClient:
            def __init__(self, **_):
                self._call_idx = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def post(self, url, **_):
                idx = min(self._call_idx, len(responses) - 1)
                self._call_idx += 1
                return _FakeResponse(responses[idx])

        self._monkeypatch.setattr(httpx, "Client", _FakeClient)
