import hashlib
import secrets
from datetime import timedelta

import jwt
from pwdlib import PasswordHash

from .models import AuthSession, now

passwords = PasswordHash.recommended()
DUMMY_HASH = passwords.hash("not-a-real-user-password")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def secret_token() -> str:
    return secrets.token_urlsafe(32)


def issue_tokens(db, user, settings, *, family_id=None, expires_at=None):
    from uuid import uuid4

    raw = secret_token()
    session = AuthSession(
        user_id=user.id,
        family_id=family_id or uuid4(),
        refresh_hash=digest(raw),
        expires_at=expires_at or now() + timedelta(days=settings.refresh_days),
    )
    db.add(session)
    db.flush()
    issued = now()
    access = jwt.encode(
        {
            "sub": str(user.id),
            "sid": str(session.id),
            "iat": issued,
            "exp": issued + timedelta(minutes=settings.access_minutes),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "type": "access",
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    return {
        "access_token": access,
        "refresh_token": raw,
        "token_type": "bearer",
        "expires_in": settings.access_minutes * 60,
    }
