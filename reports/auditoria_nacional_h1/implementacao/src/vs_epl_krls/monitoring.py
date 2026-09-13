"""Leitura dos ledgers prospectivos: o que exige atencao humana, e quando.

Por que este modulo existe
--------------------------
Os ledgers acumulam corretamente — cada previsao registrada, cada semana
liquidada contra o valor oficial, tudo encadeado por SHA-256.  E ninguem e
avisado de nada.  Uma evidencia prospectiva que so existe se alguem lembrar de
olhar nao e governanca, e disciplina; e disciplina semanal e a primeira coisa que
falha.

Os quatro sinais
----------------
1. **Cobertura fora da faixa.**  O intervalo declara 80%; se a cobertura
   observada sair da tolerancia depois de amostra suficiente, o cenario de custo
   que o produto serve esta errado.
2. **Pior que a persistencia.**  Se o modelo erra mais que "o preco de hoje" ao
   longo de varias semanas, ele nao esta pagando o proprio custo.
3. **Contagem atingida.**  Ao chegar ao minimo prospectivo, a decisao de promocao
   passa a ser possivel — e precisa de gente, nunca de automatismo.
4. **Liquidacao atrasada.**  Uma previsao pendente por mais semanas do que o
   ciclo de publicacao da ANP significa que o processo semanal parou de rodar.
   E o alerta mais chato e o mais util: sem ele, os outros tres nunca disparam.

Nada aqui promove, rebaixa ou altera modelo.  O modulo le e classifica.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .audit import FORECAST_EVENTS, verify_audit_ledger

__all__ = [
    "DEFAULT_COVERAGE_TOLERANCE",
    "DEFAULT_PROSPECTIVE_TARGET",
    "LedgerAlert",
    "LedgerStatus",
    "review_ledger",
    "review_ledgers",
]

#: Tolerancia da cobertura antes de alertar, em pontos de proporcao.  Dez pontos
#: e larga de proposito: com poucas dezenas de semanas, a cobertura observada tem
#: erro amostral grande, e alertar cedo demais treina todo mundo a ignorar.
DEFAULT_COVERAGE_TOLERANCE = 0.10

#: Semanas prospectivas exigidas antes de qualquer decisao de promocao.
DEFAULT_PROSPECTIVE_TARGET = 26

#: Amostra minima antes de julgar cobertura ou erro.
DEFAULT_MIN_SAMPLES = 20

#: Semanas de tolerancia para a liquidacao.  A ANP publica com alguns dias de
#: atraso; duas semanas sem liquidar significa processo parado, nao atraso.
DEFAULT_SETTLEMENT_GRACE_WEEKS = 2


@dataclass(frozen=True)
class LedgerAlert:
    """Um sinal, o numero que o disparou e o limite que ele cruzou."""

    level: str
    code: str
    message: str
    observed: float
    threshold: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LedgerStatus:
    """Estado prospectivo de um modelo, do jeito que um operador precisa ler."""

    ledger: str
    model: str
    forecasts: int
    settled: int
    target: int
    pending_target_date: str | None
    weeks_pending: int | None
    coverage: float | None
    mae: float | None
    persistence_mae: float | None
    alerts: tuple[LedgerAlert, ...] = field(default_factory=tuple)

    @property
    def worst_level(self) -> str:
        for level in ("critical", "warning", "info"):
            if any(alert.level == level for alert in self.alerts):
                return level
        return "ok"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["alerts"] = [alert.as_dict() for alert in self.alerts]
        payload["worst_level"] = self.worst_level
        return payload


def _as_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def review_ledger(
    path: str | Path,
    *,
    model: str | None = None,
    nominal_coverage: float = 0.80,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    target: int = DEFAULT_PROSPECTIVE_TARGET,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    grace_weeks: int = DEFAULT_SETTLEMENT_GRACE_WEEKS,
    today: date | None = None,
) -> LedgerStatus:
    """Le um ledger verificado e devolve estado e alertas.

    O ledger e verificado antes de qualquer leitura: um elo adulterado levanta
    excecao em vez de produzir um relatorio bonito sobre dado corrompido.
    """

    source = Path(path)
    records = verify_audit_ledger(source)
    issued = [record for record in records if record["event"] in FORECAST_EVENTS]
    settled = [record for record in records if record["event"] == "realized"]
    reference = today or datetime.now(timezone.utc).date()

    name = model or (
        str(settled[-1]["payload"].get("model")) if settled else source.stem
    )

    errors = np.array(
        [float(record["payload"]["absolute_error"]) for record in settled], dtype=float
    )
    persistence = np.array(
        [
            float(record["payload"]["persistence_absolute_error"])
            for record in settled
            if "persistence_absolute_error" in record["payload"]
        ],
        dtype=float,
    )
    covered = [
        bool(record["payload"]["interval_covered"])
        for record in settled
        if "interval_covered" in record["payload"]
    ]

    pending_date: str | None = None
    weeks_pending: int | None = None
    if issued:
        last_target = str(issued[-1]["payload"]["forecast"]["target_date"])
        already = {str(record["payload"]["target_date"]) for record in settled}
        if last_target not in already:
            pending_date = last_target
            parsed = _as_date(last_target)
            if parsed is not None:
                weeks_pending = max(0, (reference - parsed).days // 7)

    alerts: list[LedgerAlert] = []

    if weeks_pending is not None and weeks_pending > grace_weeks:
        alerts.append(
            LedgerAlert(
                level="critical",
                code="liquidacao_atrasada",
                message=(
                    f"a previsao para {pending_date} esta pendente ha {weeks_pending} "
                    "semanas: o processo semanal parou de rodar"
                ),
                observed=float(weeks_pending),
                threshold=float(grace_weeks),
            )
        )

    coverage = float(np.mean(covered)) if covered else None
    if coverage is not None and len(covered) >= min_samples:
        deviation = coverage - nominal_coverage
        if abs(deviation) > coverage_tolerance:
            alerts.append(
                LedgerAlert(
                    level="warning",
                    code="cobertura_fora_da_faixa",
                    message=(
                        f"cobertura observada de {coverage:.1%} contra nominal de "
                        f"{nominal_coverage:.0%} em {len(covered)} semanas"
                    ),
                    observed=coverage,
                    threshold=nominal_coverage,
                )
            )

    mae = float(np.mean(errors)) if errors.size else None
    persistence_mae = float(np.mean(persistence)) if persistence.size else None
    if (
        mae is not None
        and persistence_mae is not None
        and errors.size >= min_samples
        and mae > persistence_mae
    ):
        alerts.append(
            LedgerAlert(
                level="critical",
                code="pior_que_persistencia",
                message=(
                    f"MAE prospectivo de {mae:.4f} contra {persistence_mae:.4f} da "
                    f"persistencia em {errors.size} semanas: o modelo nao paga o proprio custo"
                ),
                observed=mae,
                threshold=persistence_mae,
            )
        )

    if len(settled) >= target:
        alerts.append(
            LedgerAlert(
                level="info",
                code="contagem_atingida",
                message=(
                    f"{len(settled)} semanas liquidadas: a decisao de promocao passa a ser "
                    "possivel e exige revisao humana"
                ),
                observed=float(len(settled)),
                threshold=float(target),
            )
        )

    return LedgerStatus(
        ledger=str(source),
        model=name,
        forecasts=len(issued),
        settled=len(settled),
        target=target,
        pending_target_date=pending_date,
        weeks_pending=weeks_pending,
        coverage=coverage,
        mae=mae,
        persistence_mae=persistence_mae,
        alerts=tuple(alerts),
    )


def review_ledgers(
    paths: Mapping[str, str | Path], **kwargs: object
) -> dict[str, LedgerStatus]:
    """Revisa varios ledgers, ignorando os que ainda nao existem."""

    reviewed: dict[str, LedgerStatus] = {}
    for name, path in paths.items():
        if not Path(path).is_file():
            continue
        reviewed[name] = review_ledger(path, model=name, **kwargs)  # type: ignore[arg-type]
    return reviewed
