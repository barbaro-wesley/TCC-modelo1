"""Runtime schema; changes require a separate, frozen Alembic migration."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData()

runs = sa.Table(
    "model_runs",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column(
        "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("completed_at", sa.DateTime(timezone=True)),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("code_version", sa.String(100), nullable=False),
    sa.Column("protocol", sa.String(100), nullable=False),
    sa.Column("config", JSONB, nullable=False),
    sa.Column("sources", JSONB, nullable=False),
    sa.Column("error_code", sa.String(100)),
    sa.CheckConstraint(
        "status IN ('running','published','failed','no_change','data_only','superseded')",
        name="ck_model_run_status",
    ),
)
forecasts = sa.Table(
    "model_forecasts",
    metadata,
    sa.Column("run_id", sa.BigInteger, sa.ForeignKey("model_runs.id"), primary_key=True),
    sa.Column("origin_date", sa.Date, nullable=False),
    sa.Column("target_date", sa.Date, nullable=False),
    sa.Column("model", sa.String(100), nullable=False),
    sa.Column("point", sa.Float, nullable=False),
    sa.Column("observed_price", sa.Float, nullable=False),
    sa.Column("p10", sa.Float),
    sa.Column("p90", sa.Float),
    sa.Column("details", JSONB, nullable=False),
    sa.CheckConstraint("target_date = origin_date + 7", name="ck_model_forecast_h1"),
    sa.CheckConstraint(
        "point > 0 AND point < 'Infinity'::double precision AND observed_price > 0 AND observed_price < 'Infinity'::double precision",
        name="ck_model_forecast_prices",
    ),
    sa.CheckConstraint(
        "(p10 IS NULL AND p90 IS NULL) OR (p10 IS NOT NULL AND p90 IS NOT NULL AND p10 > 0 AND p90 >= p10 AND p90 < 'Infinity'::double precision)",
        name="ck_model_forecast_interval",
    ),
)
observations = sa.Table(
    "model_observations",
    metadata,
    sa.Column("run_id", sa.BigInteger, sa.ForeignKey("model_runs.id"), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("price", sa.Float, nullable=False),
    sa.CheckConstraint(
        "price > 0 AND price < 'Infinity'::double precision", name="ck_model_observation_price"
    ),
)
metrics = sa.Table(
    "model_metrics",
    metadata,
    sa.Column("run_id", sa.BigInteger, sa.ForeignKey("model_runs.id"), primary_key=True),
    sa.Column("model", sa.String(100), primary_key=True),
    sa.Column("horizon", sa.Integer, primary_key=True),
    sa.Column("values", JSONB, nullable=False),
    sa.CheckConstraint("horizon > 0", name="ck_model_metric_horizon"),
)
publication = sa.Table(
    "model_publication",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("run_id", sa.BigInteger, sa.ForeignKey("model_forecasts.run_id")),
    sa.CheckConstraint("id = 1", name="ck_model_publication_singleton"),
)
