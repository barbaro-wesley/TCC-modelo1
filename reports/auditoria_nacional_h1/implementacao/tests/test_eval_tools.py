from __future__ import annotations

import numpy as np
import pytest

from eval.drift import page_hinkley, psi, rolling_rmse
from eval.importance import permutation_importance
from eval.intervals import conformal_p10_p90, direction_probs


def test_conformal_intervals_use_only_past_residuals():
    residuals = np.arange(10, dtype=float)
    yhat = np.full(10, 100.0)
    lo, hi = conformal_p10_p90(residuals, yhat, min_resid=5)
    assert np.isnan(lo[:5]).all()
    assert lo[5] == pytest.approx(100 + np.quantile(residuals[:5], 0.1))
    assert hi[5] == pytest.approx(100 + np.quantile(residuals[:5], 0.9))
    assert lo[5] <= hi[5]


def test_direction_probabilities_are_disjoint_and_sum_to_one():
    residuals = np.array([-0.10, -0.01, 0.0, 0.01, 0.10])
    probs = direction_probs(residuals, yhat=10.0, y_now=10.0, band=0.02, min_resid=5)
    assert probs == pytest.approx({"p_alta": 0.2, "p_estavel": 0.6, "p_queda": 0.2})
    assert sum(probs.values()) == pytest.approx(1.0)
    insufficient = direction_probs(residuals[:2], 1.0, 1.0, min_resid=3)
    assert all(np.isnan(v) for v in insufficient.values())


def test_page_hinkley_shapes_and_ignores_nan():
    values = np.r_[np.zeros(20), np.ones(20)]
    values[2] = np.nan
    result = page_hinkley(values, delta=0.001, lam=0.05)
    assert result["flags"].shape == values.shape
    assert result["path"].shape == values.shape
    assert set(np.unique(result["flags"])).issubset({0, 1})


def test_rolling_rmse_known_window_and_nan_handling():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.0, 1.0, np.nan, 2.0])
    result = rolling_rmse(y, yhat, window=2)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(np.sqrt(0.5))
    assert result[2] == 1.0
    assert result[3] == 2.0


def test_psi_identical_constant_and_shifted_distributions():
    expected = np.linspace(0.0, 1.0, 100)
    assert psi(expected, expected, bins=10) == pytest.approx(0.0)
    assert psi(np.ones(20), np.ones(20), bins=10) == 0.0
    assert psi(expected, expected + 0.4, bins=10) > 0.1
    assert np.isnan(psi(np.arange(3), np.arange(3), bins=5))


def test_permutation_importance_ranks_signal_feature_first():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    y = 5.0 * X[:, 0]
    result = permutation_importance(
        lambda values: 5.0 * values[:, 0],
        X,
        y,
        ["signal", "noise"],
        n_repeats=4,
        rng=np.random.default_rng(2),
    )
    assert list(result)[0] == "signal"
    assert result["signal"] > result["noise"]

