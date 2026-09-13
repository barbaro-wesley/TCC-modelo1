"""Gates decidiveis: as propriedades que fazem eles concluirem onde o antigo nao concluia."""

from __future__ import annotations

import numpy as np
import pytest

from vs_epl_krls.gates import (
    EVENT_THRESHOLD_BRL,
    accuracy_report,
    bootstrap_mean_ci,
    diebold_mariano_hln,
    evaluate_challenger,
    event_mask,
    interval_report,
    moving_block_bootstrap,
    paired_block_bootstrap,
    winkler_score,
)


def test_winkler_is_the_width_when_the_interval_covers() -> None:
    score = winkler_score([10.0], [9.0], [11.0], alpha=0.2)

    assert score.tolist() == pytest.approx([2.0])


def test_winkler_charges_the_miss_at_two_over_alpha() -> None:
    # Furou por 1.0 abaixo, com alpha=0.2 -> penalidade 2/0.2 * 1.0 = 10, mais a largura.
    below = winkler_score([8.0], [9.0], [11.0], alpha=0.2)
    above = winkler_score([12.0], [9.0], [11.0], alpha=0.2)

    assert below.tolist() == pytest.approx([12.0])
    assert above.tolist() == pytest.approx([12.0])


def test_winkler_prefers_the_sharp_interval_over_the_lazy_wide_one() -> None:
    """E o ponto do modulo: cobertura sozinha premia quem abre a banda."""

    truth = np.array([10.0, 10.2, 9.8, 10.1, 9.9])
    sharp_low, sharp_high = truth - 0.3, truth + 0.3
    wide_low, wide_high = truth - 3.0, truth + 3.0

    sharp = interval_report(truth, sharp_low, sharp_high, nominal_coverage=0.8)
    wide = interval_report(truth, wide_low, wide_high, nominal_coverage=0.8)

    # As duas cobrem 100%; so o Winkler distingue.
    assert sharp.empirical_coverage == wide.empirical_coverage == 1.0
    assert sharp.mean_winkler < wide.mean_winkler


def test_interval_report_measures_calibration_error_with_sign() -> None:
    truth = np.arange(10, dtype=float)
    lower = truth - 0.5
    upper = truth + 0.5
    lower[0] = truth[0] + 0.1  # um furo deliberado

    report = interval_report(truth, lower, upper, nominal_coverage=0.80)

    assert report.n == 10
    assert report.empirical_coverage == pytest.approx(0.9)
    assert report.calibration_error == pytest.approx(0.1)
    assert report.mean_width > 0


def test_interval_report_rejects_inverted_and_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="upper nao pode ser menor"):
        winkler_score([1.0], [2.0], [1.0], alpha=0.2)
    with pytest.raises(ValueError, match="alpha precisa estar"):
        winkler_score([1.0], [0.0], [2.0], alpha=1.5)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        interval_report([1.0, 2.0], [0.0], [3.0], nominal_coverage=0.8)


def test_event_mask_separates_the_weeks_that_actually_moved() -> None:
    actual = np.array([6.00, 6.05, 6.05, 6.20])
    origin = np.array([6.00, 6.00, 6.05, 6.05])

    mask = event_mask(actual, origin, threshold=EVENT_THRESHOLD_BRL)

    assert mask.tolist() == [False, True, False, True]


def test_moving_block_bootstrap_is_deterministic_and_unbiased() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(size=200)

    first = moving_block_bootstrap(series, n_resamples=500, random_state=7)
    second = moving_block_bootstrap(series, n_resamples=500, random_state=7)

    assert first.shape == (500,)
    assert np.array_equal(first, second)
    assert float(np.mean(first)) == pytest.approx(float(np.mean(series)), abs=0.05)


def test_bootstrap_ci_widens_with_the_confidence_level() -> None:
    rng = np.random.default_rng(1)
    series = rng.normal(loc=2.0, scale=1.0, size=120)

    low90, high90 = bootstrap_mean_ci(series, level=0.90, random_state=3)
    low99, high99 = bootstrap_mean_ci(series, level=0.99, random_state=3)

    assert low90 < 2.0 < high90
    assert (high99 - low99) > (high90 - low90)


def test_paired_bootstrap_finds_no_edge_between_identical_models() -> None:
    rng = np.random.default_rng(2)
    loss = np.abs(rng.normal(size=150))

    result = paired_block_bootstrap(loss, loss, random_state=5)

    assert result["mean_difference"] == 0.0
    assert result["pvalue_not_better"] == 1.0


def test_paired_bootstrap_detects_a_real_and_consistent_edge() -> None:
    rng = np.random.default_rng(3)
    incumbent = np.abs(rng.normal(size=200)) + 0.5
    challenger = incumbent - 0.2  # melhor em toda semana

    result = paired_block_bootstrap(challenger, incumbent, random_state=5)

    assert result["mean_difference"] == pytest.approx(-0.2)
    assert result["ci90_high"] < 0.0
    assert result["pvalue_not_better"] == 0.0


def test_hln_correction_is_more_conservative_than_the_plain_statistic() -> None:
    """A correcao existe porque o DM assintotico e otimista em amostra curta."""

    rng = np.random.default_rng(4)
    incumbent = rng.normal(scale=1.0, size=60)
    challenger = incumbent * 0.9

    result = diebold_mariano_hln(challenger, incumbent)

    assert abs(result["dm_stat_hln"]) < abs(result["dm_stat"])
    assert result["n"] == 60
    assert 0.0 <= result["pvalue"] <= 1.0


def test_dm_returns_a_neutral_verdict_when_the_models_are_identical() -> None:
    errors = np.array([0.1, -0.2, 0.3, -0.4, 0.5])

    result = diebold_mariano_hln(errors, errors)

    assert result["mean_difference"] == 0.0
    assert result["pvalue"] == 1.0


def test_dm_validates_its_inputs() -> None:
    errors = np.array([0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="'absolute' ou 'squared'"):
        diebold_mariano_hln(errors, errors, loss="huber")
    with pytest.raises(ValueError, match="horizon precisa ser positivo"):
        diebold_mariano_hln(errors, errors, horizon=0)
    with pytest.raises(ValueError, match="pelo menos tres"):
        diebold_mariano_hln([0.1, 0.2], [0.3, 0.4])


def test_squared_loss_is_available_but_not_the_default() -> None:
    rng = np.random.default_rng(6)
    incumbent = rng.normal(size=80)
    challenger = incumbent * 0.8

    absolute = diebold_mariano_hln(challenger, incumbent)
    squared = diebold_mariano_hln(challenger, incumbent, loss="squared")

    assert absolute["mean_difference"] < 0
    assert squared["mean_difference"] < 0
    assert absolute != squared


def test_accuracy_report_splits_regimes_and_exposes_concentration() -> None:
    origin = np.full(10, 6.0)
    actual = origin.copy()
    actual[9] = 6.60  # uma unica semana de evento, com salto grande
    prediction = origin.copy()

    report = accuracy_report(actual, prediction, origin)

    assert report.n == 10
    assert report.n_event == 1
    assert report.n_quiet == 9
    assert report.mae_quiet == 0.0
    assert report.mae_event == pytest.approx(0.60)
    # Toda a soma de quadrados vem de um ponto: e a assinatura que motivou
    # trocar RMSE por MAE nesta serie.
    assert report.largest_error_share_of_sse == pytest.approx(1.0)


def test_accuracy_report_scores_direction_only_on_weeks_that_moved() -> None:
    origin = np.array([6.0, 6.0, 6.0, 6.0])
    actual = np.array([6.0, 6.1, 5.9, 6.2])
    prediction = np.array([6.0, 6.05, 5.95, 5.90])  # erra a direcao so na ultima

    report = accuracy_report(actual, prediction, origin)

    assert report.directional_accuracy == pytest.approx(2 / 3)


def _linear_savings(triggered: np.ndarray, gain: float) -> np.ndarray:
    savings = np.zeros(triggered.size)
    savings[triggered] = gain
    return savings


def test_verdict_promotes_a_challenger_that_wins_everywhere() -> None:
    rng = np.random.default_rng(11)
    n = 120
    origin = np.full(n, 6.0)
    actual = origin + rng.normal(scale=0.03, size=n)
    incumbent = origin.copy()
    challenger = actual - rng.normal(scale=0.005, size=n)  # bem mais preciso
    triggered = np.zeros(n, dtype=bool)
    triggered[::10] = True

    verdict = evaluate_challenger(
        challenger="paridade",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=challenger,
        prediction_incumbent=incumbent,
        savings_challenger=_linear_savings(triggered, 50.0),
        savings_incumbent=_linear_savings(triggered, 20.0),
        # Banda de 80% coerente com o desvio de 0,005 do challenger: 1,2816 sigma.
        interval_challenger=(challenger - 0.0064, challenger + 0.0064),
        interval_incumbent=(incumbent - 0.30, incumbent + 0.30),
    )

    assert verdict.promote is True
    assert verdict.failed_gates == ()
    assert verdict.mae_comparison["ci90_high"] < 0
    assert verdict.as_dict()["challenger"] == "paridade"


def test_verdict_blocks_a_challenger_whose_gain_is_one_lucky_week() -> None:
    """Economia concentrada num evento e aposta, nao politica — e o gate diz isso."""

    rng = np.random.default_rng(12)
    n = 104
    origin = np.full(n, 6.0)
    actual = origin + rng.normal(scale=0.03, size=n)
    incumbent = origin.copy()
    challenger = incumbent.copy()
    savings = np.zeros(n)
    savings[50] = 10_000.0  # tudo veio de uma semana
    savings[10] = 100.0

    verdict = evaluate_challenger(
        challenger="paridade",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=challenger,
        prediction_incumbent=incumbent,
        savings_challenger=savings,
    )

    assert verdict.promote is False
    assert "economia_nao_concentrada" in verdict.failed_gates


def test_verdict_blocks_a_challenger_that_regresses_on_event_weeks() -> None:
    n = 80
    origin = np.full(n, 6.0)
    actual = origin.copy()
    event = np.zeros(n, dtype=bool)
    event[::8] = True
    actual[event] = 6.30
    incumbent = np.where(event, 6.28, 6.0)
    challenger = np.where(event, 6.05, 6.0)  # desiste justamente nos eventos

    verdict = evaluate_challenger(
        challenger="challenger",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=challenger,
        prediction_incumbent=incumbent,
    )

    assert verdict.promote is False
    assert "sem_regressao_em_semana_de_evento" in verdict.failed_gates


def test_verdict_blocks_an_interval_that_is_conservative_instead_of_calibrated() -> None:
    rng = np.random.default_rng(13)
    n = 104
    origin = np.full(n, 6.0)
    actual = origin + rng.normal(scale=0.02, size=n)
    prediction = origin.copy()

    verdict = evaluate_challenger(
        challenger="challenger",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=prediction,
        prediction_incumbent=prediction,
        interval_challenger=(prediction - 1.0, prediction + 1.0),
        nominal_coverage=0.80,
    )

    interval = verdict.interval_challenger
    assert interval is not None
    assert interval.empirical_coverage == 1.0
    assert "intervalo_calibrado" in verdict.failed_gates
