"""Unified JSON error responses for the FastAPI service.

Maps internal exception hierarchies (``EngineError``, ``ValueError``,
etc.) to structured JSON envelopes with a stable shape so the frontend
and CLI consumers can render consistent error UI::

    {
      "error": {
        "category": "transient" | "permanent" | "fatal" | "client",
        "code": "engine.timeout" | "validation" | ...,
        "message": "human-readable summary",
        "retryable": true|false,
        "retry_after": 1.5 | null
      }
    }
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ports.errors import EngineError, PermanentEngineError, TransientEngineError


log = logging.getLogger(__name__)


def _envelope(category: str, code: str, message: str, *, retryable: bool = False, retry_after: float | None = None) -> dict:
    return {
        "error": {
            "category": category,
            "code": code,
            "message": message,
            "retryable": retryable,
            "retry_after": retry_after,
        }
    }


def install_error_handlers(api: FastAPI) -> None:
    """Attach JSON error handlers to the given FastAPI app."""

    @api.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        category = "client" if 400 <= exc.status_code < 500 else "fatal"
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(category, f"http.{exc.status_code}", str(exc.detail)),
            headers=exc.headers or None,
        )

    @api.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "client",
                "validation",
                "request validation failed",
            )
            | {"details": exc.errors()},
        )

    @api.exception_handler(TransientEngineError)
    async def _transient(request: Request, exc: TransientEngineError):
        return JSONResponse(
            status_code=503,
            content=_envelope(
                "transient",
                f"engine.{exc.code}",
                exc.message,
                retryable=True,
                retry_after=exc.retry_after,
            ),
        )

    @api.exception_handler(PermanentEngineError)
    async def _permanent(request: Request, exc: PermanentEngineError):
        return JSONResponse(
            status_code=502,
            content=_envelope("permanent", f"engine.{exc.code}", exc.message),
        )

    @api.exception_handler(EngineError)
    async def _engine(request: Request, exc: EngineError):
        return JSONResponse(
            status_code=502,
            content=_envelope("permanent", f"engine.{exc.code}", exc.message),
        )

    @api.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_envelope("fatal", "internal", f"{type(exc).__name__}: {exc}"),
        )


__all__ = ["install_error_handlers"]
