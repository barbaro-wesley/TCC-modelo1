"""Atomic publication and bounded reads of the national weekly forecast."""

import math
from datetime import date, timedelta

import sqlalchemy as sa

from .schema import forecasts, metrics, observations, publication, runs


class PublicationError(ValueError):
    pass


def clean(value):
    """PostgreSQL JSONB metadata cannot contain NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def positive(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise PublicationError("Expected a finite positive price")
    return float(value)


def prepare(forecast, series, metric_rows):
    origin = date.fromisoformat(forecast["ultima_semana_observada"])
    target = date.fromisoformat(forecast["semana_prevista"])
    if target != origin + timedelta(days=7):
        raise PublicationError("Only the national h=1 weekly product can be published")
    lo, hi = forecast.get("p10"), forecast.get("p90")
    if (lo is None) != (hi is None):
        raise PublicationError("Both interval bounds are required together")
    if lo is not None and positive(lo) > positive(hi):
        raise PublicationError("Reversed interval")
    model = forecast["modelo"]
    if not isinstance(model, str) or not model or len(model) > 100:
        raise PublicationError("Invalid model name")
    row = dict(
        origin_date=origin,
        target_date=target,
        model=model,
        point=positive(forecast["previsao_pontual"]),
        observed_price=positive(forecast["preco_observado_ultima_semana"]),
        p10=lo,
        p90=hi,
        details=clean(forecast),
    )
    observed = [
        {"date": date.fromisoformat(r["data"]), "price": positive(r["revenda"])} for r in series
    ]
    if not observed or len({r["date"] for r in observed}) != len(observed):
        raise PublicationError("Observations must be nonempty and unique by date")
    latest = max(observed, key=lambda r: r["date"])
    if latest["date"] != origin or not math.isclose(
        latest["price"], row["observed_price"], abs_tol=1e-8
    ):
        raise PublicationError("Forecast origin and latest observation disagree")
    prepared_metrics = []
    for m in metric_rows:
        if not isinstance(m["horizon"], int) or m["horizon"] < 1:
            raise PublicationError("Invalid metric horizon")
        prepared_metrics.append(dict(model=m["model"], horizon=m["horizon"], values=clean(m)))
    if not any(m["model"] == model and m["horizon"] == 1 for m in prepared_metrics):
        raise PublicationError("Selected model must have h=1 evaluation metrics")
    return row, observed, prepared_metrics


class Publisher:
    """Owns short transactions only; model fitting never holds a DB connection."""

    def __init__(self, engine):
        self.engine = engine

    def start(self, *, code_version, protocol, config):
        with self.engine.begin() as conn:
            return conn.scalar(
                runs.insert()
                .values(
                    status="running",
                    code_version=code_version,
                    protocol=protocol,
                    config=clean(config),
                    sources={},
                )
                .returning(runs.c.id)
            )

    def finish(self, run_id, status, *, sources=None, error_code=None):
        if status not in {"failed", "no_change", "data_only"}:
            raise PublicationError("Invalid terminal status")
        with self.engine.begin() as conn:
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id, runs.c.status == "running")
                .values(
                    status=status,
                    completed_at=sa.func.now(),
                    sources=clean(sources or {}),
                    error_code=error_code,
                )
            )

    def state(self):
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    sa.select(forecasts, runs.c.protocol)
                    .join(runs, runs.c.id == forecasts.c.run_id)
                    .where(forecasts.c.run_id == sa.select(publication.c.run_id).scalar_subquery())
                )
                .mappings()
                .first()
            )
            if not row:
                return {}
            count = conn.scalar(
                sa.select(sa.func.count())
                .select_from(observations)
                .where(observations.c.run_id == row["run_id"])
            )
            return {
                "ultima_semana_processada": row["origin_date"].isoformat(),
                "preco_ultima_semana": row["observed_price"],
                "n_linhas_semanal": count,
                "temporal_protocol": row["protocol"],
            }

    def publish(self, run_id, forecast, series, metric_rows, *, sources=None):
        row, observed, evaluations = prepare(forecast, series, metric_rows)
        with self.engine.begin() as conn:
            # Singleton seeded by the migration. All writers lock the same row;
            # readers see either the old complete publication or the new one.
            active = conn.execute(
                sa.select(publication.c.run_id).where(publication.c.id == 1).with_for_update()
            ).first()
            if active is None:
                raise PublicationError("Missing publication row: apply migrations")
            run = (
                conn.execute(sa.select(runs).where(runs.c.id == run_id).with_for_update())
                .mappings()
                .one()
            )
            if run["status"] == "published":
                return "published"  # retry after a committed response was lost
            if run["status"] != "running":
                raise PublicationError("Execution is already terminal")
            if forecast.get("temporal_protocol") != run["protocol"]:
                raise PublicationError("Forecast protocol does not match the execution")
            if active.run_id is not None:
                previous = conn.execute(
                    sa.select(forecasts.c.origin_date).where(forecasts.c.run_id == active.run_id)
                ).scalar_one()
                if previous > row["origin_date"] or active.run_id > run_id:
                    conn.execute(
                        runs.update()
                        .where(runs.c.id == run_id)
                        .values(status="superseded", completed_at=sa.func.now())
                    )
                    return "superseded"
            conn.execute(forecasts.insert().values(run_id=run_id, **row))
            conn.execute(observations.insert(), [dict(run_id=run_id, **r) for r in observed])
            conn.execute(metrics.insert(), [dict(run_id=run_id, **r) for r in evaluations])
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    status="published", completed_at=sa.func.now(), sources=clean(sources or {})
                )
            )
            conn.execute(publication.update().where(publication.c.id == 1).values(run_id=run_id))
        return "published"


def current_forecast(db):
    row = (
        db.execute(
            sa.select(forecasts).where(
                forecasts.c.run_id == sa.select(publication.c.run_id).scalar_subquery()
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    return {
        **row["details"],
        "run_id": row["run_id"],
        "modelo": row["model"],
        "ultima_semana_observada": row["origin_date"].isoformat(),
        "semana_prevista": row["target_date"].isoformat(),
        "previsao_pontual": row["point"],
        "preco_observado_ultima_semana": row["observed_price"],
        "p10": row["p10"],
        "p90": row["p90"],
    }


def history_page(db, series, start, end, offset, limit):
    active = db.scalar(sa.select(publication.c.run_id).where(publication.c.id == 1))
    if active is None:
        return None
    if series == "observed":
        query = sa.select(observations.c.date, observations.c.price).where(
            observations.c.run_id == active
        )
        field = observations.c.date
    else:
        # Keep every immutable run for audit, expose the most recent successful
        # prediction per target week through the existing history contract.
        ranked = (
            sa.select(
                forecasts,
                runs.c.completed_at,
                sa.func.row_number()
                .over(partition_by=forecasts.c.target_date, order_by=runs.c.id.desc())
                .label("position"),
            )
            .join(runs, runs.c.id == forecasts.c.run_id)
            .where(runs.c.status == "published", runs.c.id <= active)
            .subquery()
        )
        query = (
            sa.select(ranked, observations.c.price.label("actual"))
            .outerjoin(
                observations,
                sa.and_(
                    observations.c.run_id == active, observations.c.date == ranked.c.target_date
                ),
            )
            .where(ranked.c.position == 1)
        )
        field = ranked.c.target_date
    if start:
        query = query.where(field >= start)
    if end:
        query = query.where(field <= end)
    total = db.scalar(sa.select(sa.func.count()).select_from(query.subquery()))
    rows = db.execute(query.order_by(field.desc()).offset(offset).limit(limit)).mappings()
    if series == "observed":
        return [{"data": r["date"].isoformat(), "revenda": r["price"]} for r in rows], total
    return [
        {
            "run_id": r["run_id"],
            "semana_prevista": r["target_date"].isoformat(),
            "semana_referencia": r["origin_date"].isoformat(),
            "modelo": r["model"],
            "previsao": r["point"],
            "p10": r["p10"],
            "p90": r["p90"],
            "preco_referencia": r["observed_price"],
            "gerado_em": r["completed_at"].isoformat(),
            "preco_realizado": r["actual"],
            "erro": None if r["actual"] is None else round(r["actual"] - r["point"], 6),
        }
        for r in rows
    ], total


def pipeline_status(db):
    latest = db.execute(sa.select(runs.c.status).order_by(runs.c.id.desc()).limit(1)).first()
    active = db.execute(
        sa.select(forecasts.c.origin_date, forecasts.c.model, runs.c.completed_at)
        .join(runs, runs.c.id == forecasts.c.run_id)
        .where(forecasts.c.run_id == sa.select(publication.c.run_id).scalar_subquery())
    ).first()
    names = {
        "published": "atualizado",
        "running": "executando",
        "failed": "erro",
        "no_change": "sem_novidade",
        "data_only": "dados_atualizados",
        "superseded": "substituido",
    }
    return {
        "status": names.get(latest.status, "sem_previsao") if latest else "sem_previsao",
        "ultima_semana_processada": active.origin_date.isoformat() if active else None,
        "ultima_execucao_ok": active.completed_at.isoformat() if active else None,
        "modelo_producao": active.model if active else None,
    }
