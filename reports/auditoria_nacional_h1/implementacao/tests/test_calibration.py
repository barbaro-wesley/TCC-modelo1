"""Inferencia conformal adaptativa: a propriedade que justifica trocar o quantil fixo."""

from __future__ import annotations

import numpy as np
import pytest

from vs_epl_krls.calibration import AdaptiveConformalInterval, backtest_adaptive_interval
from vs_epl_krls.gates import interval_report


def _regime_shift(seed: int = 0, calm: int = 300, shock: int = 300):
    """Serie com um choque de volatilidade no meio, 5x a escala anterior."""

    rng = np.random.default_rng(seed)
    residuals = np.concatenate(
        [rng.normal(scale=0.02, size=calm), rng.normal(scale=0.10, size=shock)]
    )
    point = np.full(residuals.size, 6.0)
    return point + residuals, point, residuals


def test_adaptive_interval_recovers_nominal_coverage_after_a_regime_shift() -> None:
    """O caso que motivou o modulo: o quantil fixo perde a cobertura, o ACI nao."""

    actual, point, residuals = _regime_shift()

    adaptive = backtest_adaptive_interval(actual, point, nominal_coverage=0.80)

    # Quantil movel com alpha praticamente congelado: o comportamento atual.
    frozen = AdaptiveConformalInterval(nominal_coverage=0.80, gamma=1e-12)
    lower = np.empty_like(actual)
    upper = np.empty_like(actual)
    for index in range(actual.size):
        lower[index], upper[index] = frozen.interval(point[index])
        frozen.observe(actual[index], lower[index], upper[index], residual=residuals[index])

    after = slice(300, None)
    adaptive_report = interval_report(
        actual[after], adaptive["lower"][after], adaptive["upper"][after], nominal_coverage=0.80
    )
    frozen_report = interval_report(
        actual[after], lower[after], upper[after], nominal_coverage=0.80
    )

    assert abs(adaptive_report.calibration_error) < abs(frozen_report.calibration_error)
    assert adaptive_report.empirical_coverage == pytest.approx(0.80, abs=0.06)
    assert frozen_report.empirical_coverage < 0.70


def test_adaptive_interval_also_wins_on_the_proper_score() -> None:
    """Cobertura melhor sem pagar em largura: o Winkler tem que cair."""

    actual, point, residuals = _regime_shift()
    adaptive = backtest_adaptive_interval(actual, point, nominal_coverage=0.80)

    frozen = AdaptiveConformalInterval(nominal_coverage=0.80, gamma=1e-12)
    lower = np.empty_like(actual)
    upper = np.empty_like(actual)
    for index in range(actual.size):
        lower[index], upper[index] = frozen.interval(point[index])
        frozen.observe(actual[index], lower[index], upper[index], residual=residuals[index])

    adaptive_score = interval_report(
        actual, adaptive["lower"], adaptive["upper"], nominal_coverage=0.80
    ).mean_winkler
    frozen_score = interval_report(actual, lower, upper, nominal_coverage=0.80).mean_winkler

    assert adaptive_score < frozen_score


def test_alpha_moves_in_the_direction_the_paper_prescribes() -> None:
    band = AdaptiveConformalInterval(nominal_coverage=0.80, gamma=0.05)
    start = band.alpha

    band.observe(10.0, 9.0, 11.0)  # cobriu -> alpha sobe, banda encolhe
    covered = band.alpha
    band.observe(99.0, 9.0, 11.0)  # furou -> alpha cai, banda abre
    missed = band.alpha

    assert covered == pytest.approx(start + 0.05 * 0.20)
    assert missed == pytest.approx(covered + 0.05 * (0.20 - 1.0))
    assert missed < start


def test_state_reports_running_coverage_and_clipping() -> None:
    band = AdaptiveConformalInterval(nominal_coverage=0.80, gamma=0.30, alpha_bounds=(0.05, 0.30))

    for _ in range(10):
        band.observe(99.0, 9.0, 11.0)  # so furos, alpha desce ate o piso

    state = band.state()
    assert state.updates == 10
    assert state.covered == 0
    assert state.running_coverage == 0.0
    assert state.alpha_current == pytest.approx(0.05)
    assert state.clipped_steps > 0
    assert state.as_dict()["running_coverage"] == 0.0


def test_running_coverage_is_nan_before_any_update() -> None:
    band = AdaptiveConformalInterval()

    assert np.isnan(band.state().running_coverage)


def test_warmup_falls_back_to_the_gaussian_multiple() -> None:
    band = AdaptiveConformalInterval(nominal_coverage=0.80, min_residuals=20)

    assert band.warmed_up is False
    lower, upper = band.interval(6.0, scale=0.05)
    # 1,2816 sigma de cada lado enquanto nao ha residuos suficientes.
    assert (upper - lower) == pytest.approx(2 * 1.2815515655 * 0.05, rel=1e-6)

    band.seed(np.random.default_rng(0).normal(scale=0.05, size=40))
    assert band.warmed_up is True


def test_seeded_residuals_do_not_count_as_observed_coverage() -> None:
    band = AdaptiveConformalInterval()

    band.seed([0.01, -0.02, np.nan, 0.03])

    state = band.state()
    assert state.residuals_in_window == 3  # o NaN e descartado
    assert state.updates == 0
    assert state.alpha_current == pytest.approx(band.alpha_target)


def test_window_forgets_the_oldest_residual() -> None:
    band = AdaptiveConformalInterval(window=5, min_residuals=2)

    band.seed(np.arange(20, dtype=float))

    assert band.state().residuals_in_window == 5
    lower, upper = band.interval(0.0)
    assert lower >= 15.0  # so os cinco ultimos sobreviveram


def test_backtest_is_causal_and_returns_one_band_per_week() -> None:
    actual, point, _ = _regime_shift(seed=3, calm=60, shock=60)

    result = backtest_adaptive_interval(actual, point, nominal_coverage=0.80, min_residuals=10)

    assert result["lower"].shape == actual.shape
    assert result["upper"].shape == actual.shape
    assert np.all(result["upper"] >= result["lower"])
    # O primeiro alpha e o alvo: nada foi observado antes da primeira banda.
    assert result["alpha"][0] == pytest.approx(0.20)
    assert result["state"].updates == actual.size


def test_backtest_accepts_a_conditional_scale_and_seed_residuals() -> None:
    actual, point, residuals = _regime_shift(seed=4, calm=40, shock=40)
    scale = np.linspace(0.5, 2.0, actual.size)

    result = backtest_adaptive_interval(
        actual, point, scale=scale, seed_residuals=residuals[:30], min_residuals=10
    )

    assert result["state"].residuals_in_window >= 30
    assert np.all(np.isfinite(result["lower"]))


def test_calibration_rejects_impossible_configuration() -> None:
    with pytest.raises(ValueError, match="nominal_coverage"):
        AdaptiveConformalInterval(nominal_coverage=1.5)
    with pytest.raises(ValueError, match="gamma"):
        AdaptiveConformalInterval(gamma=0.0)
    with pytest.raises(ValueError, match="window"):
        AdaptiveConformalInterval(window=1)
    with pytest.raises(ValueError, match="min_residuals"):
        AdaptiveConformalInterval(min_residuals=1)
    with pytest.raises(ValueError, match="alpha_bounds"):
        AdaptiveConformalInterval(alpha_bounds=(0.5, 0.2))
    with pytest.raises(ValueError, match="point precisa ser finito"):
        AdaptiveConformalInterval().interval(float("nan"))
    with pytest.raises(ValueError, match="actual precisa ser finito"):
        AdaptiveConformalInterval().observe(float("inf"), 1.0, 2.0)


def test_backtest_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        backtest_adaptive_interval([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="nao podem ser vazias"):
        backtest_adaptive_interval([], [])
    with pytest.raises(ValueError, match="scale precisa ter o mesmo tamanho"):
        backtest_adaptive_interval([1.0, 2.0], [1.0, 2.0], scale=[1.0])
