"""API-key dependencies for chat and admin routes (Phase 6).

Keys travel in the `Authorization: Bearer <key>` header. An empty configured
key disables the check for that tier (local development); an unset admin key
falls back to the main API key so enabling `API_KEY` alone locks everything.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _verify(provided: str | None, expected: str, realm: str) -> None:
    if not expected:
        return  # auth disabled for this tier
    if provided is None:
        raise HTTPException(status_code=401, detail=f"missing {realm} key")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail=f"invalid {realm} key")


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    _verify(_bearer(authorization), settings.api_key, "API")


async def require_admin_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = settings.admin_key or settings.api_key
    _verify(_bearer(authorization), expected, "admin")
