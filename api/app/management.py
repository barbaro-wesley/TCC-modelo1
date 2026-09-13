from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .db import get_db
from .dependencies import Identity, admin, audit, human, owner, subscription, tenant
from .models import (
    ApiKey,
    AuditLog,
    AuthSession,
    MonthlyUsage,
    Organization,
    PasswordReset,
    Plan,
    Subscription,
    UsageEvent,
    User,
    now,
)
from .pagination import Page, paginate, pagination
from .schemas import (
    AuditOut,
    KeyCreate,
    KeyOut,
    OrganizationCreate,
    OrganizationOut,
    OrganizationUpdate,
    PlanCreate,
    PlanOut,
    SubscriptionOut,
    SubscriptionSet,
    UsageOut,
    UserCreate,
    UserOut,
    UserUpdate,
)
from .security import digest, passwords, secret_token

router = APIRouter(prefix="/api/v1", tags=["management"])


def find(db, model, object_id):
    obj = db.get(model, object_id)
    if obj is None:
        raise HTTPException(404, detail={"code": "not_found"})
    return obj


@router.get("/admin/users")
def admin_users(
    organization_id: UUID | None = None,
    auth: Identity = Depends(admin),
    db: Session = Depends(get_db),
    page: Page = Depends(pagination),
):
    query = select(User)
    if organization_id:
        query = query.where(User.organization_id == organization_id)
    return paginate(db, query, User, page, UserOut)


@router.get("/admin/organizations/{organization_id}", response_model=OrganizationOut)
def get_organization(
    organization_id: UUID, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    return find(db, Organization, organization_id)


@router.get("/admin/organizations/{organization_id}/subscription", response_model=SubscriptionOut)
def get_subscription(
    organization_id: UUID, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    find(db, Organization, organization_id)
    sub = subscription(db, organization_id, required=False)
    if sub is None:
        raise HTTPException(404, detail={"code": "subscription_not_found"})
    return sub


def lock_org(db, organization_id):
    org = db.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    if org is None:
        raise HTTPException(404, detail={"code": "not_found"})
    if not org.active:
        raise HTTPException(403, detail={"code": "organization_inactive"})
    return org


@router.post("/admin/organizations", response_model=OrganizationOut, status_code=201)
def create_organization(
    payload: OrganizationCreate, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    org = Organization(name=payload.name, cnpj=payload.cnpj)
    db.add(org)
    db.flush()
    db.add(
        User(
            organization_id=org.id,
            name=payload.owner.name,
            email=payload.owner.email.lower(),
            password_hash=passwords.hash(payload.owner.password),
            role="owner",
        )
    )
    audit(db, auth, "organization.create", org.id, org.id)
    db.commit()
    return org


@router.get("/admin/organizations")
def organizations(
    auth: Identity = Depends(admin), db: Session = Depends(get_db), page: Page = Depends(pagination)
):
    return paginate(db, select(Organization), Organization, page, OrganizationOut)


@router.patch("/admin/organizations/{organization_id}", response_model=OrganizationOut)
def update_organization(
    organization_id: UUID,
    payload: OrganizationUpdate,
    auth: Identity = Depends(admin),
    db: Session = Depends(get_db),
):
    org = db.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    if org is None:
        raise HTTPException(404, detail={"code": "not_found"})
    for name, value in payload.model_dump(exclude_none=True).items():
        setattr(org, name, value)
    if payload.active is False:
        db.execute(update(ApiKey).where(ApiKey.organization_id == org.id).values(active=False))
        db.execute(
            update(AuthSession)
            .where(AuthSession.user_id.in_(select(User.id).where(User.organization_id == org.id)))
            .values(revoked=True)
        )
    audit(db, auth, "organization.update", org.id, org.id)
    db.commit()
    return org


@router.post("/admin/plans", response_model=PlanOut, status_code=201)
def create_plan(
    payload: PlanCreate, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    plan = Plan(**payload.model_dump())
    db.add(plan)
    db.flush()
    audit(db, auth, "plan.create", plan.id)
    db.commit()
    return plan


@router.delete("/admin/plans/{plan_id}", status_code=204)
def retire_plan(plan_id: UUID, auth: Identity = Depends(admin), db: Session = Depends(get_db)):
    plan = find(db, Plan, plan_id)
    plan.active = False
    audit(db, auth, "plan.retire", plan.id)
    db.commit()


@router.get("/plans")
def plans(
    auth: Identity = Depends(human), db: Session = Depends(get_db), page: Page = Depends(pagination)
):
    query = select(Plan)
    if auth.role != "platform_admin":
        query = query.where(Plan.active.is_(True))
    return paginate(db, query, Plan, page, PlanOut)


@router.put("/admin/organizations/{organization_id}/subscription", response_model=SubscriptionOut)
def set_subscription(
    organization_id: UUID,
    payload: SubscriptionSet,
    auth: Identity = Depends(admin),
    db: Session = Depends(get_db),
):
    lock_org(db, organization_id)
    plan = find(db, Plan, payload.plan_id)
    if not plan.active:
        raise HTTPException(409, detail={"code": "plan_retired"})
    seats = db.scalar(
        select(func.count())
        .select_from(User)
        .where(User.organization_id == organization_id, User.active.is_(True))
    )
    if seats > plan.max_users:
        raise HTTPException(409, detail={"code": "plan_has_insufficient_seats"})
    sub = subscription(db, organization_id, required=False)
    if sub is None:
        sub = Subscription(organization_id=organization_id)
        db.add(sub)
    for name, value in payload.model_dump().items():
        setattr(sub, name, value)
    for name in ("price_cents", "monthly_quota", "rate_per_minute", "max_users", "api_access"):
        setattr(sub, name, getattr(plan, name))
    db.flush()
    audit(db, auth, "subscription.set", sub.id, organization_id)
    db.commit()
    return sub


@router.get("/subscription", response_model=SubscriptionOut)
def my_subscription(auth: Identity = Depends(tenant), db: Session = Depends(get_db)):
    sub = subscription(db, auth.organization_id, required=False)
    if sub is None:
        raise HTTPException(404, detail={"code": "subscription_not_found"})
    return sub


@router.get("/organization/users")
def users(
    auth: Identity = Depends(owner), db: Session = Depends(get_db), page: Page = Depends(pagination)
):
    return paginate(
        db, select(User).where(User.organization_id == auth.organization_id), User, page, UserOut
    )


@router.post("/organization/users", response_model=UserOut, status_code=201)
def create_user(
    payload: UserCreate, auth: Identity = Depends(owner), db: Session = Depends(get_db)
):
    lock_org(db, auth.organization_id)
    sub = subscription(db, auth.organization_id)
    check_seats(db, auth.organization_id, sub.max_users)
    user = User(
        name=payload.name,
        email=payload.email.lower(),
        role=payload.role,
        organization_id=auth.organization_id,
        password_hash=passwords.hash(payload.password),
    )
    db.add(user)
    db.flush()
    audit(db, auth, "user.create", user.id)
    db.commit()
    return user


def check_seats(db, organization_id, max_users):
    count = db.scalar(
        select(func.count())
        .select_from(User)
        .where(User.organization_id == organization_id, User.active.is_(True))
    )
    if count >= max_users:
        raise HTTPException(409, detail={"code": "user_limit_exceeded"})


@router.patch("/organization/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: UUID,
    payload: UserUpdate,
    auth: Identity = Depends(owner),
    db: Session = Depends(get_db),
):
    lock_org(db, auth.organization_id)
    user = db.scalar(
        select(User)
        .where(User.id == user_id, User.organization_id == auth.organization_id)
        .with_for_update()
    )
    if user is None:
        raise HTTPException(404, detail={"code": "not_found"})
    if (
        user.role == "owner"
        and user.active
        and (payload.active is False or payload.role in {"member", "viewer"})
    ):
        count = db.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.organization_id == auth.organization_id,
                User.role == "owner",
                User.active.is_(True),
            )
        )
        if count <= 1:
            raise HTTPException(409, detail={"code": "last_owner"})
    if payload.active is True and not user.active:
        check_seats(db, auth.organization_id, subscription(db, auth.organization_id).max_users)
    for name, value in payload.model_dump(exclude_none=True).items():
        setattr(user, name, value)
    if payload.active is False or payload.role is not None:
        db.execute(update(AuthSession).where(AuthSession.user_id == user.id).values(revoked=True))
    audit(db, auth, "user.update", user.id)
    db.commit()
    return user


@router.post("/admin/users/{user_id}/password-reset")
def password_reset_token(
    user_id: UUID, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    user = db.scalar(select(User).where(User.id == user_id).with_for_update())
    if user is None:
        raise HTTPException(404, detail={"code": "not_found"})
    db.execute(update(PasswordReset).where(PasswordReset.user_id == user.id).values(used=True))
    raw = secret_token()
    expires = now() + timedelta(minutes=30)
    db.add(PasswordReset(user_id=user.id, token_hash=digest(raw), expires_at=expires))
    audit(db, auth, "user.password_reset_issued", user.id, user.organization_id)
    db.commit()
    return {"token": raw, "expires_at": expires, "delivery": "manual_secure_channel"}


@router.get("/organization/api-keys")
def keys(
    auth: Identity = Depends(owner), db: Session = Depends(get_db), page: Page = Depends(pagination)
):
    return paginate(
        db,
        select(ApiKey).where(ApiKey.organization_id == auth.organization_id),
        ApiKey,
        page,
        KeyOut,
    )


@router.post("/organization/api-keys", status_code=201)
def create_key(payload: KeyCreate, auth: Identity = Depends(owner), db: Session = Depends(get_db)):
    lock_org(db, auth.organization_id)
    if not subscription(db, auth.organization_id).api_access:
        raise HTTPException(403, detail={"code": "api_access_not_in_plan"})
    count = db.scalar(
        select(func.count())
        .select_from(ApiKey)
        .where(ApiKey.organization_id == auth.organization_id, ApiKey.active.is_(True))
    )
    if count >= 10:
        raise HTTPException(409, detail={"code": "api_key_limit"})
    raw = "s10_" + secret_token()
    key = ApiKey(
        organization_id=auth.organization_id,
        label=payload.label,
        key_hash=digest(raw),
        prefix=raw[:12],
    )
    db.add(key)
    db.flush()
    audit(db, auth, "api_key.create", key.id)
    db.commit()
    return {**KeyOut.model_validate(key).model_dump(), "api_key": raw}


@router.delete("/organization/api-keys/{key_id}", status_code=204)
def revoke_key(key_id: UUID, auth: Identity = Depends(owner), db: Session = Depends(get_db)):
    key = db.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.organization_id == auth.organization_id)
    )
    if key is None:
        raise HTTPException(404, detail={"code": "not_found"})
    key.active = False
    audit(db, auth, "api_key.revoke", key.id)
    db.commit()


def usage_summary(db, organization_id):
    period = now().strftime("%Y-%m")
    sub = subscription(db, organization_id, required=False)
    usage = db.get(MonthlyUsage, (organization_id, period))
    units = usage.units if usage else 0
    quota = sub.monthly_quota if sub else 0
    return {
        "period": period,
        "timezone": "UTC",
        "units": units,
        "quota": quota,
        "remaining": max(0, quota - units),
    }


@router.get("/usage")
def usage(auth: Identity = Depends(tenant), db: Session = Depends(get_db)):
    return usage_summary(db, auth.organization_id)


@router.get("/usage/events")
def usage_events(
    auth: Identity = Depends(tenant),
    db: Session = Depends(get_db),
    page: Page = Depends(pagination),
):
    return paginate(
        db,
        select(UsageEvent).where(UsageEvent.organization_id == auth.organization_id),
        UsageEvent,
        page,
        UsageOut,
    )


@router.get("/admin/organizations/{organization_id}/usage")
def admin_usage(
    organization_id: UUID, auth: Identity = Depends(admin), db: Session = Depends(get_db)
):
    find(db, Organization, organization_id)
    return usage_summary(db, organization_id)


@router.get("/admin/audit-logs")
def audits(
    auth: Identity = Depends(admin), db: Session = Depends(get_db), page: Page = Depends(pagination)
):
    return paginate(db, select(AuditLog), AuditLog, page, AuditOut)
