from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.hybrid import evaluate_residual_hybrid
from vs_epl_krls.selection import (
    S10Candidate,
    build_s10_supervised,
    expanding_validation_folds,
)


def _data():
    index = np.arange(150, dtype=float)
    history = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-02", periods=150, freq="7D"),
            "price": 5.0 + 0.004 * index + 0.07 * np.sin(index / 8.0),
        }
    )
    return build_s10_supervised(history, horizon=1, feature_set="lags")


def _candidate(target_mode="delta"):
    return S10Candidate(
        candidate_id="hybrid",
        feature_set="lags",
        target_mode=target_mode,
        alpha=0.05,
        beta_initial=0.05,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=1.0,
        kernel_sigma=0.5,
        regularization=1e-3,
        max_dictionary_size=5,
        max_rules=5,
        beta_recovery_rate=0.25,
        forgetting_factor=0.995,
        dictionary_usage_decay=0.99,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )


def test_residual_hybrid_returns_finite_prequential_trace():
    data = _data()
    folds, _ = expanding_validation_folds(
        data.n_samples,
        holdout_size=20,
        validation_size=12,
        n_folds=1,
        min_train_size=80,
    )
    base = data.origin_price.copy()
    result = evaluate_residual_hybrid(_candidate(), data, folds[0], base)
    assert result.predictions.shape == (12,)
    assert np.isfinite(result.predictions).all()
    assert np.allclose(result.predictions, result.base_predictions + result.corrections)
    assert result.correction_latency_ms_p95 >= 0
    assert 0 <= result.dictionary_replacement_rate <= 1
    assert result.correction_weight == 1.0
    assert result.correction_limit is None
    assert result.summary_row()["base_rmse"] >= 0


def test_residual_hybrid_applies_correction_guardrail():
    data = _data()
    folds, _ = expanding_validation_folds(
        data.n_samples,
        holdout_size=20,
        validation_size=12,
        n_folds=1,
        min_train_size=80,
    )
    candidate = replace(
        _candidate(),
        residual_correction_weight=0.5,
        residual_correction_limit=0.01,
    )
    result = evaluate_residual_hybrid(candidate, data, folds[0], data.origin_price)
    assert np.max(np.abs(result.corrections)) <= 0.01 + 1e-12
    assert result.correction_weight == 0.5
    assert result.correction_limit == 0.01


def test_residual_hybrid_does_not_consume_current_unobserved_target():
    data = _data()
    fold = expanding_validation_folds(
        data.n_samples,
        holdout_size=20,
        validation_size=12,
        n_folds=1,
        min_train_size=80,
    )[0][0]
    base = data.origin_price.copy()
    original = evaluate_residual_hybrid(_candidate(), data, fold, base)
    modified = replace(data, target_price=data.target_price.copy())
    modified.target_price[fold.validation_start] += 50.0
    changed = evaluate_residual_hybrid(_candidate(), modified, fold, base)
    assert original.predictions[0] == changed.predictions[0]
    assert original.actual[0] != changed.actual[0]


def test_residual_hybrid_validates_candidate_and_base_contracts():
    data = _data()
    fold = expanding_validation_folds(
        data.n_samples,
        holdout_size=20,
        validation_size=12,
        n_folds=1,
        min_train_size=80,
    )[0][0]
    with pytest.raises(ValueError, match="target_mode"):
        evaluate_residual_hybrid(_candidate("level"), data, fold, data.origin_price)
    with pytest.raises(ValueError, match="one value"):
        evaluate_residual_hybrid(_candidate(), data, fold, np.ones(3))
    bad_base = data.origin_price.copy()
    bad_base[fold.validation_start] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_residual_hybrid(_candidate(), data, fold, bad_base)
