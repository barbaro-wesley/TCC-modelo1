"""Estado comercial separado dos artefatos imutaveis do modelo."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Entity:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Organization(Entity, Base):
    __tablename__ = "organizations"
    name: Mapped[str] = mapped_column(String(200))
    cnpj: Mapped[str] = mapped_column(String(14), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class User(Entity, Base):
    __tablename__ = "users"
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    __table_args__ = (
        CheckConstraint(
            "(role = 'platform_admin' AND organization_id IS NULL) OR "
            "(role IN ('owner', 'member', 'viewer') AND organization_id IS NOT NULL)",
            name="ck_user_role_scope",
        ),
    )


class Plan(Entity, Base):
    __tablename__ = "plans"
    name: Mapped[str] = mapped_column(String(100))
    price_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="BRL")
    monthly_quota: Mapped[int] = mapped_column(Integer)
    rate_per_minute: Mapped[int] = mapped_column(Integer)
    max_users: Mapped[int] = mapped_column(Integer)
    api_access: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    __table_args__ = (
        CheckConstraint(
            "price_cents >= 0 AND monthly_quota >= 0 AND rate_per_minute > 0 AND max_users > 0",
            name="ck_plan_limits",
        ),
    )


class Subscription(Entity, Base):
    __tablename__ = "subscriptions"
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), unique=True)
    plan_id: Mapped[UUID] = mapped_column(ForeignKey("plans.id"))
    status: Mapped[str] = mapped_column(String(20))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Snapshot: editar/desativar o catalogo nao altera um contrato existente.
    price_cents: Mapped[int] = mapped_column(Integer)
    monthly_quota: Mapped[int] = mapped_column(Integer)
    rate_per_minute: Mapped[int] = mapped_column(Integer)
    max_users: Mapped[int] = mapped_column(Integer)
    api_access: Mapped[bool] = mapped_column(Boolean)
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="ck_subscription_dates"),
        CheckConstraint(
            "status IN ('trialing','active','past_due','canceled','suspended')",
            name="ck_subscription_status",
        ),
        CheckConstraint(
            "price_cents >= 0 AND monthly_quota >= 0 AND rate_per_minute > 0 AND max_users > 0",
            name="ck_subscription_limits",
        ),
    )


class AuthSession(Entity, Base):
    __tablename__ = "auth_sessions"
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    family_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    refresh_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class PasswordReset(Entity, Base):
    __tablename__ = "password_resets"
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used: Mapped[bool] = mapped_column(Boolean, default=False)


class ApiKey(Entity, Base):
    __tablename__ = "api_keys"
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    label: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    prefix: Mapped[str] = mapped_column(String(12))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class MonthlyUsage(Base):
    __tablename__ = "monthly_usage"
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), primary_key=True)
    period: Mapped[str] = mapped_column(String(7), primary_key=True)
    units: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (CheckConstraint("units >= 0", name="ck_usage_positive"),)


class UsageEvent(Entity, Base):
    __tablename__ = "usage_events"
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    api_key_id: Mapped[UUID | None] = mapped_column(ForeignKey("api_keys.id"))
    request_id: Mapped[str] = mapped_column(String(36))
    resource: Mapped[str] = mapped_column(String(150))
    units: Mapped[int] = mapped_column(Integer, default=1)
    period: Mapped[str] = mapped_column(String(7))
    __table_args__ = (
        Index("ix_usage_org_created", "organization_id", "created_at", "id"),
        UniqueConstraint("organization_id", "request_id", name="uq_usage_request"),
    )


class AuditLog(Entity, Base):
    __tablename__ = "audit_logs"
    actor_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("organizations.id"))
    action: Mapped[str] = mapped_column(String(100))
    target_id: Mapped[str] = mapped_column(String(36))
    __table_args__ = (Index("ix_audit_created", "created_at", "id"),)
