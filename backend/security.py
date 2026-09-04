"""Password hashing, opaque token minting, and the session cookie.

Passwords use **argon2id** with parameters pinned here rather than inherited from the
library default, so an upgrade of ``argon2-cffi`` cannot silently change the cost of every
new hash. Existing hashes carry their own parameters, and :func:`verify_password` reports
when one should be re-hashed at the current settings.

Session and reset tokens are **opaque**: 256 bits of randomness, handed to the browser once
and stored only as a SHA-256 digest. Two consequences worth stating, because they are the
reason this was chosen over a signed token:

* a database dump does not let the reader mint or replay a session, and
* deleting the row ends the session *now* — there is no window during which a signed token
  remains valid because it has not yet expired.

SHA-256 is the right hash for these and the wrong one for passwords. The distinction is
entropy: a 256-bit random token cannot be brute-forced whatever the hash costs, so speed is
free; a human-chosen password can be, so the hash must be deliberately slow.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type
from fastapi import Response

from backend.settings import settings

# RFC 9106's low-memory profile: 64 MiB, 3 passes, 4 lanes. Comfortable on a laptop and
# still expensive enough that an offline attack on a stolen hash is not casual work.
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)

# Passwords are bounded before hashing. Argon2 has no bcrypt-style truncation, so the cap
# is not about correctness — it stops a multi-megabyte "password" turning one unauthenticated
# request into seconds of CPU.
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128

TOKEN_BYTES = 32  # 256 bits

# Verified against when no account matches the submitted email, so that "no such user" and
# "wrong password" take the same time. Without it, response latency is an account-existence
# oracle and the generic error message achieves nothing.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(32))


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str | None, password: str) -> tuple[bool, str | None]:
    """``(ok, refreshed_hash)``.

    ``refreshed_hash`` is set only when the password was correct *and* the stored hash uses
    older parameters than the ones configured above, so the caller can upgrade it in place.
    Passing ``None`` for ``stored_hash`` still performs a full verification against the
    dummy hash, keeping the timing of an unknown account indistinguishable.
    """
    if stored_hash is None:
        try:
            _hasher.verify(_DUMMY_HASH, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            pass
        return False, None

    try:
        _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False, None

    try:
        if _hasher.check_needs_rehash(stored_hash):
            return True, _hasher.hash(password)
    except InvalidHashError:  # pragma: no cover - verify would already have raised
        pass
    return True, None


def mint_token() -> str:
    """A fresh opaque token. Returned to the browser once and never stored in this form."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_digest(raw: str) -> str:
    """What actually goes in the database."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.session_ttl_seconds,
        httponly=True,           # unreadable from JavaScript, so an XSS cannot lift it
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )
