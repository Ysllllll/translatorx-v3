"""Remote punctuation restoration via HTTP endpoint.

For service-oriented deployments where the NER model runs as a separate
service (avoids loading large models into every process).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class RemotePuncRestorer:
    """Punctuation restoration via HTTP POST to a remote service.

    Expected endpoint contract::

        POST /restore
        Content-Type: application/json
        {"texts": ["hello world", "another sentence"]}

        → 200 OK
        {"results": [["Hello world."], ["Another sentence."]]}

    Usage::

        restorer = RemotePuncRestorer("http://localhost:8080/restore")
        results = restorer(["hello world"])
    """

    def __init__(self, endpoint: str, *, timeout: float = 30.0, threshold: int = 0) -> None:
        self._endpoint = endpoint
        self._timeout = timeout
        self._threshold = threshold

    def __call__(self, texts: list[str]) -> list[list[str]]:
        short: dict[int, list[str]] = {}
        to_send: list[str] = []
        indices: list[int] = []
        for i, t in enumerate(texts):
            if len(t) < self._threshold:
                short[i] = [t]
            else:
                to_send.append(t)
                indices.append(i)
        if not to_send:
            return [short.get(i, [texts[i]]) for i in range(len(texts))]
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._endpoint, json={"texts": to_send})
            resp.raise_for_status()
            data = resp.json()
        remote_results = data["results"]
        merged: list[list[str]] = []
        ri = 0
        for i in range(len(texts)):
            if i in short:
                merged.append(short[i])
            else:
                merged.append(remote_results[ri])
                ri += 1
        return merged


__all__ = ["RemotePuncRestorer"]
