from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.selection import (
    S10Candidate,
    build_s10_feature_frame,
    build_s10_supervised,
    candidate_grid,
    evaluate_temporal_fold,
    expanding_validation_folds,
    rank_candidates,
)


def _history(n: int = 150) -> pd.DataFrame:
    index = np.arange(n, dtype=float)
    price = 5.0 + 0.003 * index + 0.08 * np.sin(index / 8.0)
    return pd.DataFrame(
        {"date": pd.date_range("2022-01-02", periods=n, freq="7D"), "price": price}
    )


def _candidate(identifier: str = "small") -> S10Candidate:
    return S10Candidate(
        candidate_id=identifier,
        feature_set="dynamics",
        target_mode="delta",
        alpha=0.05,
        beta_initial=0.03,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=1.0,
        kernel_sigma=0.5,
        regularization=1e-3,
        novelty_factor=0.2,
        max_dictionary_size=4,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )


@pytest.mark.parametrize("feature_set", ["price", "lags", "dynamics"])
def test_s10_features_are_finite_aligned_and_causal(feature_set):
    history = _history()
    table = build_s10_feature_frame(history, feature_set=feature_set)
    assert len(table) == len(history) - 12
    assert table["date"].iloc[-1] == history["date"].iloc[-1]
    assert np.isfinite(table.select_dtypes(include="number")).all().all()

    changed_future = history.copy()
    changed_future.loc[130:, "price"] += 100.0
    original = build_s10_feature_frame(history.iloc[:130], feature_set=feature_set)
    changed = build_s10_feature_frame(changed_future, feature_set=feature_set)
    pd.testing.assert_frame_equal(original, changed.iloc[: len(original)].reset_index(drop=True))


def test_supervised_targets_and_dates_follow_requested_horizon():
    history = _history()
    data = build_s10_supervised(history, horizon=4, feature_set="price")
    assert data.n_samples == len(history) - 12 - 4
    assert np.all(data.target_dates - data.dates == np.timedelta64(28, "D"))
    assert np.allclose(data.target_price - data.origin_price, history["price"].diff(4).iloc[16:])


def test_exogenous_features_require_and_keep_only_lagged_covariates():
    panel = _history()
    panel["brent_l1"] = np.linspace(70.0, 80.0, len(panel))
    panel["usdbrl_l1"] = np.linspace(4.8, 5.2, len(panel))
    panel["brent_brl_l1"] = panel["brent_l1"] * panel["usdbrl_l1"]
    panel["petrobras_reajuste_l1"] = 0.0
    table = build_s10_feature_frame(panel, feature_set="exogenous")
    assert len(table) == len(panel) - 12
    assert {"brent_l1", "usdbrl_l1", "brent_brl_l1"}.issubset(table.columns)
    assert not {"brent", "usdbrl"}.intersection(table.columns)
    with pytest.raises(ValueError, match="missing causal columns"):
        build_s10_feature_frame(_history(), feature_set="exogenous")


def test_expanding_folds_keep_final_holdout_untouched():
    folds, holdout = expanding_validation_folds(
        300,
        holdout_size=60,
        validation_size=30,
        n_folds=3,
        min_train_size=120,
    )
    assert [(fold.validation_start, fold.validation_end) for fold in folds] == [
        (150, 180),
        (180, 210),
        (210, 240),
    ]
    assert (holdout.validation_start, holdout.validation_end) == (240, 300)
    assert folds[-1].validation_end == holdout.validation_start
    with pytest.raises(ValueError, match="not enough"):
        expanding_validation_folds(100, min_train_size=80)


def test_fold_prediction_does_not_consume_its_current_target():
    data = build_s10_supervised(_history(), horizon=2, feature_set="dynamics")
    fold = expanding_validation_folds(
        data.n_samples,
        holdout_size=24,
        validation_size=12,
        n_folds=1,
        min_train_size=70,
    )[0][0]
    baseline = evaluate_temporal_fold(_candidate(), data, fold)
    modified = replace(data, target_price=data.target_price.copy())
    modified.target_price[fold.validation_start :] += 50.0
    changed = evaluate_temporal_fold(_candidate(), modified, fold)
    # Those labels cannot be revealed until two origins later.
    assert np.array_equal(baseline.predictions[:2], changed.predictions[:2])
    assert not np.array_equal(baseline.actual, changed.actual)


def test_candidate_grid_is_deterministic_and_parameters_are_bounded():
    first = candidate_grid(horizon=1, random_state=7, n_random=6)
    second = candidate_grid(horizon=1, random_state=7, n_random=6)
    assert first == second
    assert len(first) == 9
    assert len({candidate.candidate_id for candidate in first}) == len(first)
    delta = next(candidate for candidate in first if candidate.target_mode == "delta")
    params = delta.model_parameters()
    assert params["error_normalization"] == "running_std"
    assert params["max_dictionary_size"] == delta.max_dictionary_size
    assert params["max_rules"] == delta.max_rules
    assert params["input_bounds"] == (0.0, 1.0)
    with pytest.raises(ValueError, match="invalid"):
        candidate_grid(horizon=0)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"residual_correction_weight": 0.0}, "correction_weight"),
        ({"residual_correction_weight": 1.1}, "correction_weight"),
        ({"residual_correction_limit": 0.0}, "correction_limit"),
    ],
)
def test_candidate_rejects_invalid_residual_guardrails(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(_candidate(), **changes)


def test_small_candidate_ranking_returns_all_folds():
    data = build_s10_supervised(_history(), horizon=1, feature_set="dynamics")
    folds, _ = expanding_validation_folds(
        data.n_samples,
        holdout_size=20,
        validation_size=10,
        n_folds=1,
        min_train_size=80,
    )
    candidates = [_candidate("a"), replace(_candidate("b"), kernel_sigma=0.8)]
    ranking, results = rank_candidates(candidates, {"dynamics": data}, folds)
    assert set(ranking["candidate_id"]) == {"a", "b"}
    assert len(results) == 2
    assert np.isfinite(ranking["selection_score"]).all()
    assert all(result.predictions.size == 10 for result in results)


@pytest.mark.parametrize(
    "frame, message",
    [
        (pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]}), "date and price"),
        (pd.DataFrame({"date": pd.date_range("2024-01-01", periods=79), "price": 5.0}), "at least 80"),
    ],
)
def test_s10_input_contract_rejects_invalid_frames(frame, message):
    with pytest.raises(ValueError, match=message):
        build_s10_feature_frame(frame)
