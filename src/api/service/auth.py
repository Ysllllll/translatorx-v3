"""Authentication — X-API-Key header → (user_id, UserTier).

The mapping is configured via :class:`AuthConfig` in :mod:`application.config`.
Each key resolves to a principal (user id + tier name). When the service is
configured with no keys the endpoints require no auth (dev mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from application.resources import DEFAULT_TIERS, UserTier


API_KEY_HEADER = "X-API-Key"


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated caller."""

    user_id: str
    tier: UserTier


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


async def require_principal(request: Request) -> Principal:
    """FastAPI dependency — resolve the authenticated principal.

    Reads ``app.state.auth`` — a dict ``{api_key: Principal}`` — set up
    by :func:`create_app`. When the dict is empty, a default anonymous
    principal is returned (dev mode).
    """
    auth_map: dict[str, Principal] = getattr(request.app.state, "auth_map", {}) or {}
    if not auth_map:
        # Dev mode — no auth required.
        return Principal(user_id="anonymous", tier=DEFAULT_TIERS["free"])

    key = request.headers.get(API_KEY_HEADER)
    if not key:
        raise _unauthorized(f"Missing {API_KEY_HEADER} header")

    principal = auth_map.get(key)
    if principal is None:
        raise _unauthorized("Invalid API key")
    return principal


RequirePrincipal = Depends(require_principal)


__all__ = ["API_KEY_HEADER", "Principal", "require_principal", "RequirePrincipal"]
