"""Remote HTTP punc backend — registered as ``"remote"``.

POSTs single texts to a configurable HTTP endpoint. Two retry layers:

* **Transport** (``transport_retries``) — handled transparently by
  :class:`httpx.HTTPTransport` for connection-level failures.
* **Application** (``max_retries``) — routed through
  :func:`~ports.retries.retry_until_valid`, retrying on HTTP errors,
  malformed responses, or content mismatch
  (:func:`punc_content_matches`). Matches the LLM backend's retry shape
  so logging/diagnostics stay consistent.

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
from ports.retries import retry_until_valid

from adapters.preprocess._common import run_async_in_sync
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

    async def _restore_one(client: httpx.Client, text: str) -> str:
        payload: dict[str, object] = {"text": text}
        if language is not None:
            payload["language"] = language

        async def _attempt(_n: int) -> str:
            # Sync httpx call; each retry is sequential so blocking the
            # single-shot event loop is acceptable.
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            result = data["result"]
            if not isinstance(result, str):
                raise ValueError(f"remote response 'result' must be str, got {type(result).__name__}")
            return result

        def _validate(value: str):
            if not punc_content_matches(text, value):
                return False, value, f"content mismatch: {text[:60]!r} → {value[:60]!r}"
            return True, value, ""

        outcome = await retry_until_valid(
            _attempt,
            validate=_validate,
            max_retries=max_retries,
            on_reject=lambda attempt, reason: logger.warning(
                "Remote punc attempt %d/%d rejected: %s",
                attempt + 1,
                max_retries + 1,
                reason,
            ),
            on_exception=lambda attempt, exc: logger.warning(
                "Remote punc attempt %d/%d failed: %r",
                attempt + 1,
                max_retries + 1,
                exc,
            ),
        )
        if not outcome.accepted:
            raise RuntimeError(f"Remote punc failed after {outcome.attempts} attempts: {outcome.last_reason}")
        return outcome.value  # type: ignore[return-value]

    async def _restore_batch(texts: list[str]) -> list[str]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return [await _restore_one(client, t) for t in texts]

    def _call(texts: list[str]) -> list[str]:
        return run_async_in_sync(lambda: _restore_batch(texts))

    return _call


__all__ = ["LIBRARY_NAME", "factory"]
