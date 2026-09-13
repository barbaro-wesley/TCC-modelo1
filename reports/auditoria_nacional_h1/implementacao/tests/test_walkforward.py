from __future__ import annotations

import numpy as np
import pytest

from eval.walkforward import expanding_origin_indices, walk_forward_batch, walk_forward_online


class RecordingOnlineModel:
    def __init__(self):
        self.seen = []
        self.beta = 0.2

    @property
    def n_rules(self):
        return int(bool(self.seen))

    def update(self, x, y):
        self.seen.append((float(np.asarray(x).ravel()[0]), float(y)))

    def predict_one(self, x):
        return self.seen[-1][1] if self.seen else 0.0


def test_expanding_origin_indices_respects_end_horizon():
    assert expanding_origin_indices(10, n_min_train=4, horizon=2) == [3, 4, 5, 6, 7]


def test_online_walkforward_delays_updates_by_full_horizon():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(10, 16, dtype=float)
    result = walk_forward_online(lambda: RecordingOnlineModel(), X, y, n_min_train=2, horizon=2)
    assert np.isnan(result["yhat"][:2]).all()
    assert result["yhat"][2] == 10.0
    assert result["yhat"][3] == 11.0
    assert result["model"].seen == [(0.0, 10.0), (1.0, 11.0), (2.0, 12.0), (3.0, 13.0)]
    assert not result["mask"][:2].any()


def test_online_walkforward_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon"):
        walk_forward_online(lambda: RecordingOnlineModel(), np.ones((2, 1)), np.ones(2), 1, horizon=0)


def test_batch_walkforward_uses_only_targets_already_realized():
    calls = []

    def fit_predict(Xtr, ytr, Xte, model=None):
        calls.append((len(Xtr), None if model is None else model))
        fitted = {"train_size": len(Xtr)} if model is None else model
        return float(np.mean(ytr)), fitted

    X = np.arange(7, dtype=float).reshape(-1, 1)
    y = np.arange(7, dtype=float)
    result = walk_forward_batch(
        fit_predict,
        X,
        y,
        n_min_train=3,
        refit_every=2,
        horizon=2,
    )
    assert [size for size, _ in calls] == [2, 3, 4, 5]
    assert result["yhat"][3] == pytest.approx(0.5)
    assert result["mask"].sum() == 4

