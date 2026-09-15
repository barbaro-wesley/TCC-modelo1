import math
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from forecast_store.repository import current_forecast, history_page, pipeline_status

from .db import get_db
from .dependencies import Identity, subscription, tenant
from .management import lock_org
from .models import MonthlyUsage, UsageEvent, now
from .pagination import Page, envelope, pagination
from .schemas import CostScenario

router = APIRouter(prefix="/api/v1", tags=["forecast"])


class ForecastReader:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db

    def forecast(self):
        result = current_forecast(self.db)
        if result is None:
            raise HTTPException(503, detail={"code": "forecast_unavailable_or_stale"})
        try:
            origin = date.fromisoformat(result["ultima_semana_observada"])
            target = date.fromisoformat(result["semana_prevista"])
            today = now().date()
            if (
                not 0 <= (today - origin).days <= self.settings.forecast_max_age_days
                or target < today
            ):
                raise ValueError("stale forecast")
            if target <= origin:
                raise ValueError("invalid forecast dates")
            for field in ("previsao_pontual", "preco_observado_ultima_semana", "p10", "p90"):
                value = result.get(field)
                if field in {"p10", "p90"} and value is None:
                    continue
                if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid forecast")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(503, detail={"code": "forecast_unavailable_or_stale"}) from exc
        return result


def consume(db, auth, request, response):
    # Serializa por empresa, inclusive assinatura e criacao/reativacao de usuarios.
    # Cota e evento sao confirmados na mesma transacao; Redis nunca e fonte de cobranca.
    lock_org(db, auth.organization_id)
    sub = subscription(db, auth.organization_id)
    period = now().strftime("%Y-%m")
    db.execute(
        insert(MonthlyUsage)
        .values(organization_id=auth.organization_id, period=period, units=0)
        .on_conflict_do_nothing(index_elements=["organization_id", "period"])
    )
    usage = db.scalar(
        select(MonthlyUsage)
        .where(MonthlyUsage.organization_id == auth.organization_id, MonthlyUsage.period == period)
        .with_for_update()
    )
    if usage.units >= sub.monthly_quota:
        raise HTTPException(429, detail={"code": "monthly_quota_exceeded", "period": period})
    usage.units += 1
    db.add(
        UsageEvent(
            organization_id=auth.organization_id,
            user_id=auth.user.id if auth.user else None,
            api_key_id=auth.key.id if auth.key else None,
            resource=request.url.path,
            request_id=request.state.request_id,
            period=period,
        )
    )
    db.commit()
    response.headers["X-Quota-Limit"] = str(sub.monthly_quota)
    response.headers["X-Quota-Remaining"] = str(sub.monthly_quota - usage.units)
    response.headers["X-Quota-Period"] = period


@router.get("/forecast")
def forecast(
    request: Request,
    response: Response,
    auth: Identity = Depends(tenant),
    db: Session = Depends(get_db),
):
    subscription(db, auth.organization_id)
    data = ForecastReader(request.app.state.settings, db).forecast()
    consume(db, auth, request, response)
    return data


@router.get("/history")
def history(
    request: Request,
    response: Response,
    series: Literal["observed", "forecasts"] = "observed",
    start: date | None = Query(None),
    end: date | None = Query(None),
    auth: Identity = Depends(tenant),
    db: Session = Depends(get_db),
    page: Page = Depends(pagination),
):
    subscription(db, auth.organization_id)
    if start and end and start > end:
        raise HTTPException(422, detail={"code": "invalid_date_range"})
    result_page = history_page(db, series, start, end, page.offset, page.size)
    if result_page is None:
        raise HTTPException(503, detail={"code": "history_unavailable"})
    records, total = result_page
    result = envelope(records, total, page)
    consume(db, auth, request, response)
    return result


@router.post("/scenarios/cost")
def cost(
    payload: CostScenario,
    request: Request,
    response: Response,
    auth: Identity = Depends(tenant),
    db: Session = Depends(get_db),
):
    if auth.role == "viewer":
        raise HTTPException(403, detail={"code": "read_only_role"})
    subscription(db, auth.organization_id)
    data = ForecastReader(request.app.state.settings, db).forecast()
    volume = payload.volume_liters
    result = {
        "volume_liters": volume,
        "currency": "BRL",
        "forecast_date": data["semana_prevista"],
        "forecast_cost": round(volume * data["previsao_pontual"], 2),
        "observed_cost": round(volume * data["preco_observado_ultima_semana"], 2),
        "lower_cost": round(volume * data["p10"], 2) if data.get("p10") is not None else None,
        "upper_cost": round(volume * data["p90"], 2) if data.get("p90") is not None else None,
    }
    consume(db, auth, request, response)
    return result


@router.get("/status")
def product_status(auth: Identity = Depends(tenant), db: Session = Depends(get_db)):
    return pipeline_status(db)
