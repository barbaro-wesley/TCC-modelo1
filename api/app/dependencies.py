from dataclasses import dataclass
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import ApiKey, AuthSession, Organization, Subscription, User, now
from .security import digest

bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class Identity:
    organization_id: UUID | None
    role: str
    user: User | None = None
    key: ApiKey | None = None
    session: AuthSession | None = None


def unauthorized():
    return HTTPException(
        401, detail={"code": "invalid_credentials"}, headers={"WWW-Authenticate": "Bearer"}
    )


def subscription(db, organization_id, *, required=True):
    sub = db.scalar(
        select(Subscription)
        .where(Subscription.organization_id == organization_id)
        .execution_options(populate_existing=True)
    )
    if required and (
        sub is None
        or sub.status not in {"active", "trialing"}
        or not sub.starts_at <= now() < sub.ends_at
    ):
        raise HTTPException(403, detail={"code": "subscription_inactive"})
    return sub


def identity(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    raw_key: str | None = Depends(api_key_header),
):
    if bool(credentials) == bool(raw_key):
        raise unauthorized()
    if credentials:
        try:
            settings = request.app.state.settings
            claims = jwt.decode(
                credentials.credentials,
                settings.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
                options={"require": ["exp", "iat", "sub", "sid", "iss", "aud", "type"]},
            )
            if claims["type"] != "access":
                raise ValueError("invalid token type")
            session = db.get(AuthSession, UUID(claims["sid"]))
            user = db.get(User, UUID(claims["sub"]))
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise unauthorized() from exc
        if (
            session is None
            or user is None
            or session.user_id != user.id
            or session.revoked
            or session.expires_at <= now()
            or not user.active
        ):
            raise unauthorized()
        result = Identity(user.organization_id, user.role, user=user, session=session)
    else:
        if not raw_key.startswith("s10_") or len(raw_key) > 100:
            raise unauthorized()
        key = db.scalar(
            select(ApiKey).where(ApiKey.key_hash == digest(raw_key), ApiKey.active.is_(True))
        )
        if key is None:
            raise unauthorized()
        result = Identity(key.organization_id, "integration", key=key)
    if result.organization_id:
        org = db.get(Organization, result.organization_id)
        if org is None or not org.active:
            raise unauthorized()
        sub = subscription(db, org.id, required=False)
        limit = sub.rate_per_minute if sub else 30
        headers = request.app.state.limiter.check("organization", str(org.id), limit)
        response.headers.update(headers)
        if result.key and (not subscription(db, org.id).api_access):
            raise HTTPException(403, detail={"code": "api_access_not_in_plan"})
    return result


def human(auth: Identity = Depends(identity)):
    if auth.user is None:
        raise HTTPException(403, detail={"code": "user_login_required"})
    return auth


def admin(auth: Identity = Depends(human)):
    if auth.role != "platform_admin":
        raise HTTPException(403, detail={"code": "admin_required"})
    return auth


def owner(auth: Identity = Depends(human)):
    if auth.role != "owner":
        raise HTTPException(403, detail={"code": "owner_required"})
    return auth


def tenant(auth: Identity = Depends(identity)):
    if auth.organization_id is None:
        raise HTTPException(403, detail={"code": "organization_required"})
    return auth


def audit(db, auth, action, target_id, organization_id=None):
    from .models import AuditLog

    db.add(
        AuditLog(
            actor_id=auth.user.id if auth.user else None,
            organization_id=organization_id or auth.organization_id,
            action=action,
            target_id=str(target_id),
        )
    )
