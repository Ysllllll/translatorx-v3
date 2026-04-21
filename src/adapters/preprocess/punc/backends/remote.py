"""Remote HTTP punc backend.

POSTs single texts to a configurable endpoint. Network-level retries
are handled by :class:`httpx.HTTPTransport`; application-level retries
by a simple loop. Content validation and ``on_failure`` policy live
upstream in :class:`PuncRestorer`.

Endpoint contract::

    POST <endpoint>
    Content-Type: application/json
    {"text": "<raw>", "language": "<lang>"}  # language omitted if None

    → 200 OK
    {"result": "<punctuated>"}
"""

from __future__ import annotations

import logging

import httpx

from adapters.preprocess.punc.registry import Backend, PuncBackendRegistry

logger = logging.getLogger(__name__)


LIBRARY_NAME = "remote"


@PuncBackendRegistry.register(LIBRARY_NAME)
def factory(
    *,
    endpoint: str,
    language: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    transport_retries: int = 2,
) -> Backend:
    """Build an HTTP-backed punc backend.

    Parameters
    ----------
    endpoint:
        Full URL receiving the POST request.
    language:
        Optional language hint sent in the payload.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Application-level retries; total attempts = ``max_retries + 1``.
    transport_retries:
        Network-level retries performed transparently by httpx.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if transport_retries < 0:
        raise ValueError("transport_retries must be >= 0")

    transport = httpx.HTTPTransport(retries=transport_retries)

    def _call_one(client: httpx.Client, text: str) -> str:
        payload: dict[str, object] = {"text": text}
        if language is not None:
            payload["language"] = language

        attempts = max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                result = data["result"]
                if not isinstance(result, str):
                    raise ValueError(f"remote response 'result' must be str, got {type(result).__name__}")
                return result
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_exc = exc
                logger.warning("Remote punc attempt %d/%d failed: %r", attempt + 1, attempts, exc)

        raise RuntimeError(f"Remote punc failed after {attempts} attempts: {last_exc!r}")

    def _call(texts: list[str]) -> list[str]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return [_call_one(client, t) for t in texts]

    return _call


__all__ = ["LIBRARY_NAME", "factory"]
