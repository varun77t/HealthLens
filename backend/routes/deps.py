"""Shared dependencies: the model registry, the database session, and the signed-in user.

The registry is created once in the app lifespan and stashed on ``app.state``; routes take
it through :func:`get_registry` so tests can substitute a registry loaded without explainers
(which is 5 s faster) without touching the routes.

Authentication is **mandatory for every route that touches health information**. The
enforcement point is :func:`require_user`, attached to whole routers rather than individual
endpoints, so a route added later is protected by default rather than by being remembered.
:func:`current_principal` is the permissive variant, used only by ``GET /auth/me``.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session as DbSession

from backend.db import get_db
from backend.services import auth_service
from backend.services.registry import ModelBundle, Registry
from backend.settings import settings
from backend.tables import User, UserSession


def get_registry(request: Request) -> Registry:
    registry: Registry | None = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(status_code=503, detail="Model registry is not loaded.")
    return registry


def get_bundle(registry: Registry, disease: str) -> ModelBundle:
    """Resolve a disease from a path parameter, with a 404 that names the alternatives."""
    try:
        return registry.get(disease)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{exc}. Available modules: {', '.join(registry.diseases) or 'none'}.",
        ) from exc


def client_ip(request: Request) -> str | None:
    """The address to rate-limit against.

    ``X-Forwarded-For`` is only believed when ``TRUST_PROXY_HEADERS`` is set, because a
    client can put anything in it. Reading it unconditionally would turn every per-IP limit
    into a formality: one host could present a fresh address on every request.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()[:45] or None
    return request.client.host if request.client else None


@dataclass(frozen=True)
class Principal:
    """Who is making this request, and which browser session they are using."""

    user: User
    session: UserSession


def current_principal(
    request: Request, db: DbSession = Depends(get_db)
) -> Principal | None:
    """The signed-in user, or ``None``. Never raises — used where anonymous is a valid answer."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    resolved = auth_service.resolve_session(db, token)
    if resolved is None:
        return None
    user, session = resolved
    return Principal(user=user, session=session)


def require_principal(principal: Principal | None = Depends(current_principal)) -> Principal:
    """401 unless a valid session cookie was presented.

    ``WWW-Authenticate: Cookie`` is not a standard challenge scheme, so it is deliberately
    omitted: a browser must not be prompted with a native credential dialog for what is a
    single-page-application redirect to ``/signin``.
    """
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to use this. This service does not accept anonymous requests.",
        )
    return principal


def require_user(principal: Principal = Depends(require_principal)) -> User:
    return principal.user
