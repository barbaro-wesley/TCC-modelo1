from __future__ import annotations

import numpy as np
import pytest

from vsepl_krls.paper import (
    PaperTargets,
    make_supervised,
    monthly_params,
    reproduction_verdict,
    scale_inputs_train_only,
    within_rel,
)


def test_monthly_params_supported_horizons_and_error():
    assert monthly_params(1)["gamma_bar"] == 0.006
    assert monthly_params(6)["alpha_vs2"] == 0.26
    assert monthly_params(12)["alpha_vs1"] == 0.78
    with pytest.raises(ValueError, match="horizonte"):
        monthly_params(2)


def test_make_supervised_respects_horizon_and_skips_nan():
    revenda = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    distribuicao = np.array([0.8, 1.8, 2.8, 3.8, 4.8])
    X, y, idx = make_supervised(revenda, distribuicao, horizon=1)
    assert X.tolist() == [[0.8, 1.0], [3.8, 4.0]]
    assert y.tolist() == [2.0, 5.0]
    assert idx.tolist() == [0, 3]


def test_scaler_is_fit_only_on_training_data():
    Xtr, Xte, scaler = scale_inputs_train_only(
        np.array([[0.0], [10.0]]),
        np.array([[20.0]]),
    )
    assert Xtr.ravel().tolist() == [0.0, 1.0]
    assert Xte[0, 0] == 2.0
    assert scaler.data_max_[0] == 10.0


def test_relative_tolerance_and_reproduction_verdicts():
    target = PaperTargets(rmse=1.0, mae=2.0, ndei=3.0, n_rules=1, n_train=10)
    exact = {"rmse": 1.0, "mae": 2.0, "ndei": 3.0, "n_rules": 1}
    different_rules = {**exact, "n_rules": 2}
    wrong = {**exact, "rmse": 2.0}
    assert within_rel(1.05, 1.0, 0.1)
    assert within_rel(0.05, 0.0, 0.1)
    assert reproduction_verdict(exact, target) == "REPRODUZIDO"
    assert reproduction_verdict(different_rules, target).startswith("PARCIAL")
    assert reproduction_verdict(wrong, target) == "NAO REPRODUZIDO"

