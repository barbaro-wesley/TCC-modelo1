"""As tres taxas: cada uma so significa algo junto da definicao que a produziu."""

from __future__ import annotations

import numpy as np
import pytest

from vs_epl_krls.performance import PRICE_TOLERANCES_BRL, performance_report


def _report(**kwargs):
    base = {
        "model": "teste",
        "window": "sintetica",
        "actual": [6.00, 6.10, 6.00, 6.20, 6.00],
        "prediction": [6.00, 6.05, 6.02, 6.15, 6.01],
        "origin": [6.00, 6.00, 6.10, 6.00, 6.20],
    }
    base.update(kwargs)
    return performance_report(**base)


def test_a_trigger_that_went_nowhere_still_counts_as_a_trigger() -> None:
    """Inferir disparo por economia nao nula infla a taxa de acerto."""

    # Cinco disparos: tres com lucro, um com prejuizo, um em que o preco nao moveu.
    savings = np.array([100.0, 50.0, -30.0, 0.0, 80.0])

    inferido = _report(weekly_savings=savings)
    declarado = _report(weekly_savings=savings, triggers=5, winning_triggers=3)

    # Sem a contagem da politica, o disparo de resultado zero desaparece.
    assert inferido.profitability.triggers == 4
    assert declarado.profitability.triggers == 5
    assert declarado.profitability.win_rate == pytest.approx(0.60)
    assert inferido.profitability.win_rate > declarado.profitability.win_rate


def test_profit_factor_separates_gains_from_losses() -> None:
    report = _report(
        weekly_savings=np.array([100.0, 50.0, -30.0, 0.0, 80.0]),
        triggers=5,
        winning_triggers=3,
    )
    profit = report.profitability

    assert profit.gross_profit_brl == pytest.approx(230.0)
    assert profit.gross_loss_brl == pytest.approx(30.0)
    assert profit.profit_factor == pytest.approx(230.0 / 30.0)
    assert profit.net_savings_brl == pytest.approx(200.0)
    assert profit.expectancy_brl == pytest.approx(40.0)


def test_a_strategy_that_never_loses_reports_infinite_factor_not_a_crash() -> None:
    report = _report(weekly_savings=np.array([10.0, 0.0, 5.0]), triggers=2, winning_triggers=2)

    assert np.isinf(report.profitability.profit_factor)
    assert report.profitability.win_rate == pytest.approx(1.0)


def test_return_on_spend_answers_whether_it_is_worth_it() -> None:
    report = _report(
        weekly_savings=np.array([100.0, 100.0]),
        triggers=2,
        winning_triggers=2,
        baseline_spend_brl=200_000.0,
    )

    assert report.profitability.return_on_spend == pytest.approx(0.001)


def test_movement_accuracy_excludes_the_weeks_that_did_not_move() -> None:
    """Acertar que nada aconteceria nao e previsao."""

    report = performance_report(
        model="m",
        window="w",
        # Semana 1 parada; as outras tres se movem.
        actual=[6.00, 6.10, 5.90, 6.05],
        origin=[6.00, 6.00, 6.00, 6.00],
        prediction=[6.00, 6.05, 5.95, 5.95],
    )
    movement = report.movement

    assert movement.n_moved == 3
    # Acertou alta, acertou queda, errou a ultima.
    assert movement.hit_rate_moved == pytest.approx(2 / 3)


def test_event_weeks_are_scored_apart_because_they_decide_the_purchase() -> None:
    report = performance_report(
        model="m",
        window="w",
        actual=[6.005, 6.20, 5.80],   # a primeira nao passa do limiar
        origin=[6.000, 6.00, 6.00],
        prediction=[5.990, 6.10, 5.90],
        event_threshold=0.02,
    )
    movement = report.movement

    assert movement.n_events == 2
    assert movement.hit_rate_events == pytest.approx(1.0)
    # No agregado a semana minuscula, errada, derruba o numero.
    assert movement.hit_rate_moved == pytest.approx(2 / 3)


def test_the_price_ladder_only_grows_with_the_tolerance() -> None:
    report = _report()
    ladder = [report.price.within[f"{t:.2f}"] for t in PRICE_TOLERANCES_BRL]

    assert ladder == sorted(ladder)
    assert all(0.0 <= value <= 1.0 for value in ladder)
    assert report.price.within["0.10"] == pytest.approx(1.0)
    assert report.price.median_absolute_error <= report.price.mean_absolute_error


def test_interval_coverage_is_reported_only_when_a_band_is_given() -> None:
    without = _report()
    assert without.price.interval_coverage is None
    assert without.price.interval_nominal is None

    # A terceira banda comeca acima do realizado: quatro coberturas em cinco.
    with_band = _report(
        lower=[5.95, 6.00, 6.02, 6.10, 5.96],
        upper=[6.05, 6.10, 6.07, 6.20, 6.06],
    )
    assert with_band.price.interval_coverage == pytest.approx(0.8)
    assert with_band.price.interval_nominal == pytest.approx(0.80)


def test_a_model_without_a_policy_reports_no_profitability_instead_of_zero() -> None:
    report = _report()

    assert report.profitability is None
    assert report.movement is not None
    assert report.price is not None


def test_the_report_carries_the_window_it_came_from() -> None:
    """Janelas nao sao intercambiaveis, e o numero tem de dizer de qual veio."""

    report = _report(window="holdout nacional, 104 semanas", notes=("lido duas vezes",))
    payload = report.as_dict()

    assert payload["window"] == "holdout nacional, 104 semanas"
    assert payload["notes"] == ["lido duas vezes"]


def test_mismatched_series_are_refused() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        performance_report(
            model="m", window="w", actual=[1.0, 2.0], prediction=[1.0], origin=[1.0, 2.0]
        )
