"""Structured request logging middleware."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


log = logging.getLogger("api.request")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Logs one structured line per request with ``request_id`` + latency."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable]):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            dur_ms = (time.perf_counter() - started) * 1000.0
            log.info(
                "rid=%s method=%s path=%s status=%d dur_ms=%.2f",
                rid,
                request.method,
                request.url.path,
                status_code,
                dur_ms,
            )


__all__ = ["RequestLogMiddleware"]
