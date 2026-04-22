"""Remote HTTP punc backend — registered as ``"remote"``.

POSTs single texts to a configurable HTTP endpoint. Two retry layers:

* **Transport** (``transport_retries``) — handled transparently by
  :class:`httpx.HTTPTransport` for connection-level failures.
* **Application** (``max_retries``) — re-sends the request on both HTTP
  errors and content mismatches (:func:`punc_content_matches`).

Raises :class:`RuntimeError` per text when every attempt fails; the
orchestrator :class:`~adapters.preprocess.punc.restorer.PuncRestorer`
catches that and applies the configured ``on_failure`` policy.

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

from domain.lang import punc_content_matches

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
        Application-level retries on HTTP errors or content mismatch;
        total attempts = ``max_retries + 1``.
    transport_retries:
        Network-level retries performed transparently by httpx.

    Raises
    ------
    ValueError
        If ``max_retries < 0`` or ``transport_retries < 0``.
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
        last_reason: str = "no attempts made"
        last_result: str | None = None
        for attempt in range(attempts):
            try:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                result = data["result"]
                if not isinstance(result, str):
                    raise ValueError(f"remote response 'result' must be str, got {type(result).__name__}")
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_reason = repr(exc)
                logger.warning("Remote punc attempt %d/%d failed: %r", attempt + 1, attempts, exc)
                continue

            if not punc_content_matches(text, result):
                last_reason = f"content mismatch: {text[:60]!r} → {result[:60]!r}"
                last_result = result
                logger.warning("Remote punc attempt %d/%d rejected: %s", attempt + 1, attempts, last_reason)
                continue

            return result

        raise RuntimeError(f"Remote punc failed after {attempts} attempts: {last_reason} (last_result={last_result!r})")

    def _call(texts: list[str]) -> list[str]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return [_call_one(client, t) for t in texts]

    return _call


__all__ = ["LIBRARY_NAME", "factory"]
