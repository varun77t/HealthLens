"""Account routes — the only endpoints reachable without a session.

Everything else in this API requires authentication, so this file is the entire public
attack surface. The properties it is written to hold:

* **Sign-in failures are uniform.** Unknown address and wrong password produce the same
  status, the same message and the same latency. The only route that discloses whether an
  address is registered is ``/auth/signup``, and that is a considered trade — see the note
  there.
* **The raw token leaves exactly once**, in the ``Set-Cookie`` header. It is never in a
  response body, so it cannot end up in a log, a browser history entry or a screenshot of
  the network tab.
* **Every password change ends the other sessions.** Changing a password is what someone
  does when they believe a session is compromised.
* **Rate limits are enforced before the expensive work.** The argon2 verification is
  deliberately slow; checking the limit first is what stops that becoming the denial of
  service.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from backend.db import get_db
from backend.mail import get_mailer, reset_email
from backend.routes.deps import Principal, client_ip, current_principal, require_principal
from backend.schemas.auth import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    ResetPasswordRequest,
    SessionResponse,
    SessionSummary,
    SignupRequest,
    UserResponse,
)
from backend.security import clear_session_cookie, set_session_cookie, verify_password
from backend.services import auth_service
from backend.services.auth_service import EmailAlreadyRegistered, RateLimited
from backend.settings import settings
from backend.tables import UserSession, as_utc

router = APIRouter(prefix="/auth", tags=["auth"])

# One string for both sign-in failure modes. Two different messages would be a free
# account-existence oracle for anyone with a list of email addresses.
_BAD_CREDENTIALS = "Incorrect email or password."


def _guard(db: DbSession, kind: str, *, email: str | None, ip: str | None) -> None:
    try:
        auth_service.check_rate_limit(db, kind, email=email, ip=ip)
    except RateLimited as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Try again later.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


def _sign_in(db: DbSession, request: Request, response: Response, user) -> SessionResponse:
    raw, session = auth_service.create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip=client_ip(request),
    )
    set_session_cookie(response, raw)
    return SessionResponse(user=UserResponse.of(user), expires_at=as_utc(session.expires_at))


@router.post(
    "/signup",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account and sign in",
)
def signup(
    body: SignupRequest,
    request: Request,
    response: Response,
    db: DbSession = Depends(get_db),
):
    email = auth_service.normalize_email(str(body.email))
    ip = client_ip(request)
    _guard(db, "signup", email=email, ip=ip)
    try:
        user = auth_service.create_user(
            db, email=email, password=body.password, display_name=body.display_name
        )
    except EmailAlreadyRegistered as exc:
        # This does disclose that an address is registered. The alternative — accepting the
        # signup and sending a "you already have an account" email — needs reliable email
        # delivery, which this deployment does not assume, and produces a form that appears
        # to succeed while creating nothing. Sign-in and password reset remain
        # non-disclosing, which is where an attacker with an address list would actually go.
        auth_service.record_attempt(db, "signup", email=email, ip=ip, succeeded=False)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account already exists for that email address.",
        ) from exc
    auth_service.record_attempt(db, "signup", email=email, ip=ip, succeeded=True)
    return _sign_in(db, request, response, user)


@router.post("/login", response_model=SessionResponse, summary="Sign in")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession = Depends(get_db),
):
    email = auth_service.normalize_email(str(body.email))
    ip = client_ip(request)
    _guard(db, "login", email=email, ip=ip)

    user = auth_service.authenticate(db, email=email, password=body.password)
    auth_service.record_attempt(db, "login", email=email, ip=ip, succeeded=user is not None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_BAD_CREDENTIALS)

    auth_service.clear_failed_logins(db, email)
    return _sign_in(db, request, response, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Sign out this browser")
def logout(
    principal: Principal | None = Depends(current_principal),
    db: DbSession = Depends(get_db),
):
    # The cookie is cleared whether or not the session resolved, so a stale cookie cannot
    # leave the browser stuck in a state where it believes it is signed in.
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    if principal is not None:
        auth_service.revoke_session(db, principal.session)
    clear_session_cookie(response)
    return response


@router.get(
    "/me",
    response_model=UserResponse,
    responses={204: {"description": "Not signed in."}},
    summary="Who is signed in",
)
def me(principal: Principal | None = Depends(current_principal)):
    """204 rather than 401 when anonymous.

    This is the call the frontend makes on every page load to decide what to render. A 401
    here would be an error in the console on the landing page of a perfectly healthy
    application, and would train everyone to ignore it.
    """
    if principal is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return UserResponse.of(principal.user)


@router.get("/sessions", response_model=list[SessionSummary], summary="Signed-in browsers")
def sessions(
    principal: Principal = Depends(require_principal),
    db: DbSession = Depends(get_db),
):
    rows = db.execute(
        select(UserSession)
        .where(UserSession.user_id == principal.user.id)
        .order_by(UserSession.last_seen_at.desc())
    ).scalars().all()
    return [
        SessionSummary(
            id=s.id,
            created_at=as_utc(s.created_at),
            last_seen_at=as_utc(s.last_seen_at),
            user_agent=s.user_agent,
            current=(s.id == principal.session.id),
        )
        for s in rows
    ]


@router.post("/logout-all", response_model=MessageResponse, summary="Sign out other browsers")
def logout_all(
    principal: Principal = Depends(require_principal),
    db: DbSession = Depends(get_db),
):
    n = auth_service.revoke_other_sessions(db, principal.user, principal.session.id)
    return MessageResponse(
        message=f"Signed out of {n} other browser{'s' if n != 1 else ''}. This one stays signed in."
    )


@router.post("/password", response_model=MessageResponse, summary="Change password")
def change_password(
    body: ChangePasswordRequest,
    principal: Principal = Depends(require_principal),
    db: DbSession = Depends(get_db),
):
    ok, _ = verify_password(principal.user.password_hash, body.current_password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect."
        )
    auth_service.set_password(
        db, principal.user, body.new_password, keep_session_id=principal.session.id
    )
    return MessageResponse(
        message="Password changed. Any other browser signed in to this account has been signed out."
    )


@router.post(
    "/forgot",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request a password-reset link",
)
def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: DbSession = Depends(get_db),
):
    email = auth_service.normalize_email(str(body.email))
    ip = client_ip(request)
    _guard(db, "reset", email=email, ip=ip)
    auth_service.record_attempt(db, "reset", email=email, ip=ip, succeeded=True)

    user = auth_service.get_user_by_email(db, email)
    if user is not None:
        raw = auth_service.create_password_reset(db, user)
        url = f"{settings.app_base_url}/reset/{raw}"
        subject, text = reset_email(url, settings.reset_ttl_minutes)
        get_mailer().send(user.email, subject, text)

    # Identical response either way: this endpoint must not report whether an address is
    # registered. In development the link is printed to the server log.
    return MessageResponse(
        message="If an account exists for that address, a reset link is on its way."
    )


@router.post("/reset", response_model=MessageResponse, summary="Set a new password from a reset link")
def reset_password(body: ResetPasswordRequest, db: DbSession = Depends(get_db)):
    user = auth_service.consume_password_reset(db, body.token, body.new_password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link has expired or has already been used. Request a new one.",
        )
    # Deliberately not signed in here: possession of a link from an inbox is weaker evidence
    # than knowing the password, and the new password should be proved once before use.
    return MessageResponse(message="Password updated. You can sign in with it now.")


@router.delete(
    "/account",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete this account and everything saved to it",
)
def delete_account(
    body: DeleteAccountRequest,
    principal: Principal = Depends(require_principal),
    db: DbSession = Depends(get_db),
):
    ok, _ = verify_password(principal.user.password_hash, body.password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Password is incorrect."
        )
    auth_service.delete_user(db, principal.user)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(response)
    return response
