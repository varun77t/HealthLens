"""Request and response bodies for the account routes.

``UserResponse`` is built field by field from the ORM object rather than with
``from_attributes``. Attribute-mapping a ``User`` is one forgotten field away from serving
``password_hash``, and this is the one place in the application where that mistake would be
unrecoverable.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.security import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH
from backend.tables import User, as_utc

# A deliberately short list. Real credential-stuffing defence is a breach-corpus check
# (HIBP's k-anonymity range API), which needs a network call this application does not
# make. This catches the handful that a demonstration will otherwise be typed with, and
# the length floor does the rest.
_OBVIOUS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwertyuiop", "letmein123", "iloveyou1", "administrator", "multidisease",
}


def _check_password(value: str, *, email: str | None = None) -> str:
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(value) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    if value.strip() != value.strip(" ") or not value.strip():
        raise ValueError("Password cannot be blank.")
    if value.lower() in _OBVIOUS:
        raise ValueError("That password is too common. Choose something less predictable.")
    if email:
        local = email.split("@", 1)[0].lower()
        if value.lower() in {email.lower(), local}:
            raise ValueError("Password cannot be your email address.")
    return value


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = Field(
        default=None,
        max_length=60,
        description="Optional. Used for the greeting, so the interface need not show an email address.",
    )

    @field_validator("password")
    @classmethod
    def _password(cls, v: str, info) -> str:
        return _check_password(v, email=(info.data or {}).get("email"))


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _password(cls, v: str) -> str:
        return _check_password(v)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _password(cls, v: str) -> str:
        return _check_password(v)


class DeleteAccountRequest(BaseModel):
    """Deletion is irreversible, so it is re-authenticated rather than taken on the session alone."""

    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    confirm: str = Field(description="Must be the literal string 'DELETE'.")

    @field_validator("confirm")
    @classmethod
    def _confirm(cls, v: str) -> str:
        if v.strip() != "DELETE":
            raise ValueError("Type DELETE to confirm.")
        return v


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    greeting_name: str
    created_at: datetime

    @classmethod
    def of(cls, user: User) -> "UserResponse":
        return cls(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            greeting_name=user.greeting_name,
            created_at=as_utc(user.created_at),
        )


class SessionResponse(BaseModel):
    user: UserResponse
    expires_at: datetime


class MessageResponse(BaseModel):
    message: str


class SessionSummary(BaseModel):
    id: str
    created_at: datetime
    last_seen_at: datetime
    user_agent: str | None
    current: bool
