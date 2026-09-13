"""Initial PostgreSQL platform schema (frozen, independent of runtime models)."""

import sqlalchemy as sa
from alembic import op

revision = "0001_platform"
down_revision = None
branch_labels = None
depends_on = None


def identity():
    return [
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ]


def fk(name, table, nullable=False):
    return sa.Column(name, sa.Uuid(), sa.ForeignKey(f"{table}.id"), nullable=nullable)


def upgrade():
    op.create_table(
        "organizations",
        *identity(),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("cnpj", sa.String(14), nullable=False, unique=True),
        sa.Column("active", sa.Boolean(), nullable=False),
    )
    op.create_table(
        "users",
        *identity(),
        fk("organization_id", "organizations", True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("email", sa.String(254), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(30), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "(role = 'platform_admin' AND organization_id IS NULL) OR "
            "(role IN ('owner', 'member', 'viewer') AND organization_id IS NOT NULL)",
            name="ck_user_role_scope",
        ),
    )
    op.create_index("ix_users_organization_id", "users", ["organization_id"])
    op.create_table(
        "plans",
        *identity(),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("price_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("monthly_quota", sa.Integer(), nullable=False),
        sa.Column("rate_per_minute", sa.Integer(), nullable=False),
        sa.Column("max_users", sa.Integer(), nullable=False),
        sa.Column("api_access", sa.Boolean(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "price_cents >= 0 AND monthly_quota >= 0 AND rate_per_minute > 0 AND max_users > 0",
            name="ck_plan_limits",
        ),
    )
    op.create_table(
        "subscriptions",
        *identity(),
        fk("organization_id", "organizations"),
        fk("plan_id", "plans"),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price_cents", sa.Integer(), nullable=False),
        sa.Column("monthly_quota", sa.Integer(), nullable=False),
        sa.Column("rate_per_minute", sa.Integer(), nullable=False),
        sa.Column("max_users", sa.Integer(), nullable=False),
        sa.Column("api_access", sa.Boolean(), nullable=False),
        sa.UniqueConstraint("organization_id"),
        sa.CheckConstraint("ends_at > starts_at", name="ck_subscription_dates"),
        sa.CheckConstraint(
            "status IN ('trialing','active','past_due','canceled','suspended')",
            name="ck_subscription_status",
        ),
        sa.CheckConstraint(
            "price_cents >= 0 AND monthly_quota >= 0 AND rate_per_minute > 0 AND max_users > 0",
            name="ck_subscription_limits",
        ),
    )
    op.create_table(
        "auth_sessions",
        *identity(),
        fk("user_id", "users"),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("refresh_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_family_id", "auth_sessions", ["family_id"])
    op.create_table(
        "password_resets",
        *identity(),
        fk("user_id", "users"),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_password_resets_user_id", "password_resets", ["user_id"])
    op.create_table(
        "api_keys",
        *identity(),
        fk("organization_id", "organizations"),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("prefix", sa.String(12), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_api_keys_organization_id", "api_keys", ["organization_id"])
    op.create_table(
        "monthly_usage",
        fk("organization_id", "organizations"),
        sa.Column("period", sa.String(7), nullable=False),
        sa.Column("units", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("organization_id", "period"),
        sa.CheckConstraint("units >= 0", name="ck_usage_positive"),
    )
    op.create_table(
        "usage_events",
        *identity(),
        fk("organization_id", "organizations"),
        fk("user_id", "users", True),
        fk("api_key_id", "api_keys", True),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("resource", sa.String(150), nullable=False),
        sa.Column("units", sa.Integer(), nullable=False),
        sa.Column("period", sa.String(7), nullable=False),
        sa.UniqueConstraint("organization_id", "request_id", name="uq_usage_request"),
    )
    op.create_index("ix_usage_org_created", "usage_events", ["organization_id", "created_at", "id"])
    op.create_table(
        "audit_logs",
        *identity(),
        fk("actor_id", "users", True),
        fk("organization_id", "organizations", True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_id", sa.String(36), nullable=False),
    )
    op.create_index("ix_audit_created", "audit_logs", ["created_at", "id"])


def downgrade():
    for table in (
        "audit_logs",
        "usage_events",
        "monthly_usage",
        "api_keys",
        "password_resets",
        "auth_sessions",
        "subscriptions",
        "plans",
        "users",
        "organizations",
    ):
        op.drop_table(table)
