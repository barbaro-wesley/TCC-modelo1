from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    coverage,
    diebold_mariano,
    directional_accuracy,
    mae,
    ndei,
    pinball,
    rmse,
    smape,
    summarize,
)


def test_error_metrics_known_values_and_alignment():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 1.0, 5.0, 99.0])
    assert rmse(y, yhat) == pytest.approx(np.sqrt(5.0 / 3.0))
    assert mae(y, yhat) == pytest.approx(1.0)
    assert smape(y, y) == 0.0
    assert ndei(y, y) == 0.0


def test_ndei_constant_target_is_nan_and_smape_handles_zeros():
    assert np.isnan(ndei(np.ones(3), np.ones(3)))
    assert smape(np.zeros(2), np.zeros(2)) == 0.0


def test_directional_accuracy_ignores_unchanged_targets():
    y_prev = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([2.0, 1.0, 1.0, 4.0])
    yhat = np.array([1.5, 9.0, 3.0, 2.0])
    assert directional_accuracy(y, yhat, y_prev) == pytest.approx(1.0 / 3.0)
    assert np.isnan(directional_accuracy(y_prev, y_prev, y_prev))


def test_pinball_and_coverage():
    y = np.array([0.0, 1.0, 2.0])
    assert pinball(y, y, 0.1) == 0.0
    assert coverage(y, np.array([-1.0, 0.5, 2.1]), np.array([0.1, 1.5, 3.0])) == pytest.approx(2 / 3)


def test_summarize_includes_direction_when_baseline_is_given():
    result = summarize(
        np.array([2.0, 3.0]),
        np.array([2.1, 2.9]),
        np.array([1.0, 2.0]),
    )
    assert result["n"] == 2.0
    assert result["dir_acc"] == 1.0
    assert set(result) == {"rmse", "mae", "ndei", "smape", "n", "dir_acc"}


def test_diebold_mariano_sign_and_identical_errors():
    equal = diebold_mariano(np.array([1.0, -1.0, 1.0]), np.array([1.0, -1.0, 1.0]))
    assert equal["dm_stat"] == 0.0
    assert equal["pvalue"] == 1.0
    worse = diebold_mariano(
        np.array([2.0, 2.2, 1.8, 2.1]),
        np.array([1.0, 1.1, 0.9, 1.0]),
        h=2,
    )
    assert worse["dm_stat"] > 0
    assert worse["mean_diff"] > 0

