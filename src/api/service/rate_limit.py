"""Per-user request rate limiting (token bucket, in-memory)."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.service.auth import API_KEY_HEADER


class _Bucket:
    __slots__ = ("tokens", "last")

    def __init__(self, tokens: float, last: float) -> None:
        self.tokens = tokens
        self.last = last


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket limiter keyed on authenticated user id.

    Unauthenticated callers (dev mode) share a single ``"anonymous"``
    bucket. Endpoints exempt from the limiter: ``/health``, ``/ready``,
    ``/metrics``, and the CORS preflight (``OPTIONS``).
    """

    EXEMPT_PATHS = ("/health", "/ready", "/metrics")

    def __init__(self, app, *, rps: float, burst: int | None = None) -> None:
        super().__init__(app)
        self._rps = max(0.0, float(rps))
        self._capacity = float(burst) if burst and burst > 0 else max(1.0, self._rps)
        self._buckets: dict[str, _Bucket] = defaultdict(lambda: _Bucket(self._capacity, time.monotonic()))
        self._lock = asyncio.Lock()

    def _key(self, request: Request) -> str:
        # Try to reuse the auth resolution shortcut — the header/cookie/query
        # order mirrors require_principal. We don't fully authenticate here;
        # we only partition the buckets so that callers with different
        # principals cannot starve each other.
        auth_map = getattr(request.app.state, "auth_map", {}) or {}
        if not auth_map:
            return "anonymous"
        key = request.headers.get(API_KEY_HEADER) or request.cookies.get("trx_api_key") or request.query_params.get("access_token")
        if key and key in auth_map:
            return auth_map[key].user_id
        # Invalid / missing key — group together; require_principal will
        # reject them anyway, so we just need a bucket to soak noise.
        return "__unauth__"

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable]):
        if self._rps <= 0 or request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in self.EXEMPT_PATHS):
            return await call_next(request)

        key = self._key(request)
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets[key]
            elapsed = max(0.0, now - bucket.last)
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rps)
            bucket.last = now
            if bucket.tokens < 1.0:
                retry_after = max(0.01, (1.0 - bucket.tokens) / self._rps)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "category": "client",
                            "code": "rate_limited",
                            "message": f"rate limit exceeded ({self._rps:g} rps)",
                            "retryable": True,
                            "retry_after": retry_after,
                        }
                    },
                    headers={"Retry-After": f"{retry_after:.2f}"},
                )
            bucket.tokens -= 1.0
        return await call_next(request)


__all__ = ["RateLimitMiddleware"]
