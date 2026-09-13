from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db import get_db
from .dependencies import Identity, audit, human, unauthorized
from .models import AuthSession, Organization, PasswordReset, User, now
from .schemas import Login, NewPassword, Refresh, UserOut
from .security import DUMMY_HASH, digest, issue_tokens, passwords

router = APIRouter(prefix="/api/v1", tags=["authentication"])


def usable_user(db, user):
    if user is None or not user.active:
        raise unauthorized()
    if user.organization_id:
        org = db.get(Organization, user.organization_id)
        if org is None or not org.active:
            raise unauthorized()


@router.post("/auth/login")
def login(payload: Login, request: Request, db: Session = Depends(get_db)):
    request.app.state.limiter.check(
        "login-email", payload.email.lower(), request.app.state.settings.login_rate_per_minute
    )
    user = db.scalar(select(User).where(User.email == payload.email.lower()).with_for_update())
    valid = passwords.verify(payload.password, user.password_hash if user else DUMMY_HASH)
    if not valid:
        raise unauthorized()
    usable_user(db, user)
    tokens = issue_tokens(db, user, request.app.state.settings)
    audit(db, Identity(user.organization_id, user.role, user=user), "auth.login", user.id)
    db.commit()
    return tokens


@router.post("/auth/refresh")
def refresh(payload: Refresh, request: Request, db: Session = Depends(get_db)):
    old = db.scalar(
        select(AuthSession).where(AuthSession.refresh_hash == digest(payload.refresh_token))
    )
    if old is None:
        raise unauthorized()
    user = db.scalar(select(User).where(User.id == old.user_id).with_for_update())
    db.refresh(old)  # another worker may have rotated while we waited for the user lock
    if old.revoked:
        db.execute(
            update(AuthSession).where(AuthSession.family_id == old.family_id).values(revoked=True)
        )
        db.commit()
        raise unauthorized()
    if old.expires_at <= now():
        raise unauthorized()
    usable_user(db, user)
    old.revoked = True
    tokens = issue_tokens(
        db, user, request.app.state.settings, family_id=old.family_id, expires_at=old.expires_at
    )
    db.commit()
    return tokens


@router.post("/auth/logout", status_code=204)
def logout(auth: Identity = Depends(human), db: Session = Depends(get_db)):
    db.scalar(select(User).where(User.id == auth.user.id).with_for_update())
    db.execute(
        update(AuthSession)
        .where(AuthSession.family_id == auth.session.family_id)
        .values(revoked=True)
    )
    db.commit()


@router.post("/auth/reset-password", status_code=204)
def reset_password(payload: NewPassword, db: Session = Depends(get_db)):
    token = db.scalar(
        select(PasswordReset).where(PasswordReset.token_hash == digest(payload.token))
    )
    if token is None:
        raise unauthorized()
    user = db.scalar(select(User).where(User.id == token.user_id).with_for_update())
    db.refresh(token)
    if token.used or token.expires_at <= now():
        raise unauthorized()
    usable_user(db, user)
    user.password_hash = passwords.hash(payload.password)
    db.execute(update(PasswordReset).where(PasswordReset.user_id == user.id).values(used=True))
    db.execute(update(AuthSession).where(AuthSession.user_id == user.id).values(revoked=True))
    audit(db, Identity(user.organization_id, user.role, user=user), "auth.password_reset", user.id)
    db.commit()


@router.get("/me", response_model=UserOut)
def me(auth: Identity = Depends(human)):
    return auth.user
