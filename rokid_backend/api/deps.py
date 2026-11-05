"""Authentication dependencies: device bearer tokens, admin tokens and admin sessions."""
from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from rokid_backend.core.security import SCOPE_ADMIN, SCOPE_DEVICE, hash_token
from rokid_backend.models import ApiToken

SESSION_COOKIE = "rokid_admin_session"
# Declared so the OpenAPI schema (Swagger "Authorize" button) knows about the bearer token.
bearer_scheme = HTTPBearer(auto_error=False, description="Token de dispozitiv (scope device) sau token admin (scope admin).")


@dataclass
class Principal:
    scope: str
    name: str


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def _lookup_token(request: Request, token: str) -> Principal | None:
    app = request.app.state
    if app.settings.admin_api_token and token == app.settings.admin_api_token:
        return Principal(SCOPE_ADMIN, "env-admin-token")
    digest = hash_token(token)
    with app.db.session() as session:
        row = session.execute(select(ApiToken).where(ApiToken.token_hash == digest, ApiToken.revoked.is_(False))).scalars().first()
        if row is None:
            return None
        row.last_used_at = time.time()
        return Principal(row.scope, row.name or row.id)


def current_principal(request: Request, _credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> Principal:
    token = _bearer(request)
    if token:
        principal = _lookup_token(request, token)
        if principal is None:
            raise HTTPException(401, "Token invalid sau revocat.")
        return principal
    session = request.app.state.signer.verify(request.cookies.get(SESSION_COOKIE))
    if session and session.get("scope") == SCOPE_ADMIN:
        return Principal(SCOPE_ADMIN, session.get("user", "admin"))
    raise HTTPException(401, "Autentificare necesara.", headers={"WWW-Authenticate": "Bearer"})


def require_device(principal: Principal = Depends(current_principal)) -> Principal:
    if principal.scope not in (SCOPE_DEVICE, SCOPE_ADMIN):
        raise HTTPException(403, "Domeniu de autorizare insuficient.")
    return principal


def require_admin(request: Request, principal: Principal = Depends(current_principal)) -> Principal:
    if principal.scope != SCOPE_ADMIN:
        raise HTTPException(403, "Operatiune rezervata administratorului.")
    if request.method not in ("GET", "HEAD", "OPTIONS") and not _bearer(request):
        # Cookie-authenticated mutations require the CSRF header set by the admin UI.
        if request.headers.get("X-Requested-With") != "rokid-admin":
            raise HTTPException(403, "Lipseste antetul X-Requested-With.")
    return principal
