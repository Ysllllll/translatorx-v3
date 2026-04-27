"""WebSocket router — bidirectional ``/api/ws/streams`` endpoint.

Parallel to the existing SSE-based ``/api/streams`` router; both paths
share the same :class:`StreamBuilder` / :class:`LiveStreamHandle`
implementation. Clients pick whichever transport their environment
supports.

Auth reuses :func:`api.service.auth.require_principal` — the standard
``X-API-Key`` header (or ``trx_api_key`` cookie / ``access_token``
query param) is honoured for the ``WebSocket`` upgrade. Connections
without a valid principal are rejected with policy code ``1008``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, status

from api.service.auth import API_KEY_HEADER, Principal
from api.service.runtime.ws_session import WsSession
from application.resources import DEFAULT_TIERS

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["streams-ws"])


def _ws_resolve_principal(ws: WebSocket) -> Principal | None:
    """WebSocket-flavoured copy of :func:`require_principal`.

    Returns ``None`` to signal a 1008 close. WebSocket can't raise
    HTTPException meaningfully — the upgrade is either accepted or
    rejected before any frames flow.
    """

    auth_map: dict[str, Principal] = getattr(ws.app.state, "auth_map", {}) or {}
    if not auth_map:
        return Principal(user_id="anonymous", tier=DEFAULT_TIERS["free"])
    key = ws.headers.get(API_KEY_HEADER.lower()) or ws.cookies.get("trx_api_key") or ws.query_params.get("access_token")
    if not key:
        return None
    return auth_map.get(key)


@router.websocket("/api/ws/streams")
async def ws_streams(ws: WebSocket) -> None:
    """Bidirectional WebSocket entry point.

    Accepts the upgrade, authenticates the principal, then delegates
    everything to :class:`WsSession`. Returns when the session
    finishes (client abort / disconnect / server shutdown).
    """

    principal = _ws_resolve_principal(ws)
    if principal is None:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()

    app = getattr(ws.app.state, "app", None)
    if app is None:
        _logger.error("ws_streams: app.state.app is not configured")
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    session = WsSession(ws, app=app, tenant_id=principal.tenant)
    try:
        await session.run()
    except Exception:
        _logger.exception("ws_streams session crashed")
        try:
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    _ = principal  # auth required, not currently used downstream


__all__ = ["router"]
