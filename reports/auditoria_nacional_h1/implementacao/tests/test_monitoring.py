"""Alertas dos ledgers: a evidencia prospectiva tem de reclamar sozinha."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from vs_epl_krls.audit import record_forecast, settle_pending_forecast
from vs_epl_krls.monitoring import review_ledger, review_ledgers


def _weeks(n: int, start: str = "2026-01-04") -> list[str]:
    return [str(day.date()) for day in pd.date_range(start, periods=n, freq="7D")]


def _forecast(target: str, *, origin: float, point: float, band: float, sha: str) -> dict:
    return {
        "artifact_sha256": sha,
        "forecast": {
            "target_date": target,
            "origin_price": origin,
            "point": point,
            "lower": point - band,
            "upper": point + band,
        },
        "decision": {"recommend_prebuy": False},
    }


def _fill(
    ledger,
    *,
    n: int = 30,
    band: float = 0.05,
    model_error: float = 0.02,
    miss_every: int = 5,
    miss_error: float = 0.20,
    persistence_error: float = 0.10,
) -> list[str]:
    """Emite e liquida ``n`` semanas com erros controlados.

    Cada semana anda ``persistence_error`` a partir da anterior, que e portanto o
    erro que a persistencia cometeria. ``miss_every`` controla em que fracao das
    semanas o realizado cai fora da banda, permitindo construir uma cobertura
    alvo exata.
    """

    weeks = _weeks(n)
    price = 6.50
    for index, target in enumerate(weeks):
        origin = price
        observed = origin + persistence_error
        error = miss_error if index % miss_every == 0 else model_error
        record_forecast(
            ledger,
            _forecast(
                target,
                origin=origin,
                point=observed - error,
                band=band,
                sha=f"{index:064d}",
            ),
            target_date=target,
        )
        settle_pending_forecast(ledger, {target: observed}, model_name="teste")
        price = observed
    return weeks


def test_a_quiet_ledger_raises_nothing(tmp_path):
    ledger = tmp_path / "quiet.jsonl"
    # 4 em 5 semanas dentro da banda: cobertura de 80%, o nominal declarado.
    # 25 semanas fica abaixo do alvo prospectivo, entao nada mais deve falar.
    weeks = _fill(ledger, n=25, miss_every=5)

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]))

    assert status.settled == 25
    assert status.coverage == pytest.approx(0.80)
    assert status.mae < status.persistence_mae
    assert status.worst_level == "ok"
    assert status.alerts == ()


def test_settlement_that_stopped_running_is_critical(tmp_path):
    """O alerta mais chato e o mais util: sem ele, os outros nunca disparam."""

    ledger = tmp_path / "stalled.jsonl"
    record_forecast(
        ledger,
        _forecast("2026-01-04", origin=6.50, point=6.51, band=0.05, sha="a" * 64),
        target_date="2026-01-04",
    )

    status = review_ledger(ledger, today=date(2026, 2, 15))

    assert status.pending_target_date == "2026-01-04"
    assert status.weeks_pending == 6
    assert status.worst_level == "critical"
    alert = next(a for a in status.alerts if a.code == "liquidacao_atrasada")
    assert "parou de rodar" in alert.message


def test_a_pending_week_inside_the_grace_period_is_not_an_alert(tmp_path):
    ledger = tmp_path / "recent.jsonl"
    record_forecast(
        ledger,
        _forecast("2026-01-04", origin=6.50, point=6.51, band=0.05, sha="b" * 64),
        target_date="2026-01-04",
    )

    status = review_ledger(ledger, today=date(2026, 1, 11))

    assert status.weeks_pending == 1
    assert status.worst_level == "ok"


def test_an_interval_that_stopped_covering_is_flagged(tmp_path):
    ledger = tmp_path / "coverage.jsonl"
    # Banda estreitissima: nada cai dentro.
    weeks = _fill(ledger, n=30, band=0.0001)

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]))

    assert status.coverage == pytest.approx(0.0)
    alert = next(a for a in status.alerts if a.code == "cobertura_fora_da_faixa")
    assert alert.level == "warning"
    assert alert.threshold == pytest.approx(0.80)


def test_an_interval_that_covers_everything_is_also_flagged(tmp_path):
    """Cobrir demais nao e seguranca gratuita: distorce o cenario de custo."""

    ledger = tmp_path / "wide.jsonl"
    weeks = _fill(ledger, n=30, band=5.0)

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]))

    assert status.coverage == pytest.approx(1.0)
    assert any(a.code == "cobertura_fora_da_faixa" for a in status.alerts)


def test_a_model_that_loses_to_persistence_is_critical(tmp_path):
    """Errar mais que o preco de hoje significa nao pagar o proprio custo."""

    ledger = tmp_path / "worse.jsonl"
    weeks = _fill(
        ledger,
        n=25,
        band=1.0,
        model_error=0.30,
        miss_every=1000,
        persistence_error=0.05,
    )

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]))

    assert status.mae > status.persistence_mae
    alert = next(a for a in status.alerts if a.code == "pior_que_persistencia")
    assert alert.level == "critical"
    assert status.worst_level == "critical"


def test_reaching_the_target_asks_for_a_human_not_a_promotion(tmp_path):
    ledger = tmp_path / "ready.jsonl"
    weeks = _fill(ledger, n=26, miss_every=5)

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]), target=26)

    alert = next(a for a in status.alerts if a.code == "contagem_atingida")
    assert alert.level == "info"
    assert "revisao humana" in alert.message
    # Informacao, nao criticidade: chegar ao alvo nao e um problema.
    assert status.worst_level == "info"


def test_small_samples_do_not_trigger_judgement(tmp_path):
    """Alertar cedo demais treina todo mundo a ignorar o alerta."""

    ledger = tmp_path / "small.jsonl"
    weeks = _fill(ledger, n=5, band=0.0001)

    status = review_ledger(ledger, today=date.fromisoformat(weeks[-1]))

    assert status.coverage == pytest.approx(0.0)
    assert status.alerts == ()


def test_a_tampered_ledger_raises_instead_of_reporting(tmp_path):
    ledger = tmp_path / "tampered.jsonl"
    _fill(ledger, n=3)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert "teste" in lines[1]
    lines[1] = lines[1].replace("teste", "outro")
    ledger.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    with pytest.raises(RuntimeError, match="audit record hash mismatch"):
        review_ledger(ledger)


def test_review_skips_ledgers_that_do_not_exist_yet(tmp_path):
    ledger = tmp_path / "exists.jsonl"
    _fill(ledger, n=3)

    reviewed = review_ledgers({"existe": ledger, "ausente": tmp_path / "nada.jsonl"})

    assert set(reviewed) == {"existe"}
    assert reviewed["existe"].model == "existe"


def test_an_empty_ledger_reports_nothing_pending(tmp_path):
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("", encoding="utf-8")

    status = review_ledger(ledger)

    assert status.forecasts == 0
    assert status.settled == 0
    assert status.pending_target_date is None
    assert status.coverage is None
    assert status.worst_level == "ok"
