import json
import math
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .db import get_db
from .dependencies import Identity, subscription, tenant
from .management import lock_org
from .models import MonthlyUsage, UsageEvent, now
from .pagination import Page, envelope, pagination
from .schemas import CostScenario

router = APIRouter(prefix="/api/v1", tags=["forecast"])


class Artifacts:
    def __init__(self, settings):
        self.settings = settings

    def read(self, name):
        try:
            # Nomes fornecidos exclusivamente pelo servidor, nunca pelo cliente.
            with (self.settings.data_dir / name).open("rb") as stream:
                raw = stream.read(self.settings.max_artifact_bytes + 1)
            if len(raw) > self.settings.max_artifact_bytes:
                raise ValueError("artifact too large")
            payload = json.loads(raw, parse_constant=lambda value: None)
            if not isinstance(payload, dict):
                raise ValueError("invalid artifact")
            return payload
        except (OSError, ValueError) as exc:
            raise HTTPException(503, detail={"code": "artifact_unavailable"}) from exc

    def forecast(self):
        result = self.read("previsao.json")
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
    data = request.app.state.artifacts.forecast()
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
    artifact = request.app.state.artifacts.read("historico.json")
    records = artifact.get("serie" if series == "observed" else "previsoes", [])
    field = "data" if series == "observed" else "semana_prevista"
    try:
        records = sorted(records, key=lambda row: date.fromisoformat(row[field]), reverse=True)
        records = [
            row
            for row in records
            if (not start or row[field] >= start.isoformat())
            and (not end or row[field] <= end.isoformat())
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(503, detail={"code": "invalid_history_artifact"}) from exc
    result = envelope(records[page.offset : page.offset + page.size], len(records), page)
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
    data = request.app.state.artifacts.forecast()
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
def product_status(request: Request, auth: Identity = Depends(tenant)):
    result = request.app.state.artifacts.read("status.json")
    # Nao expor caminhos, mensagens internas ou credenciais contidas em erros do pipeline.
    return {
        key: result.get(key)
        for key in ("status", "ultima_semana_processada", "ultima_execucao_ok", "modelo_producao")
    }
