"""Dependency-free regression metrics for experiments and reports."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _pairs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(y_pred, dtype=float).ravel()
    if truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal shape")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric inputs must be finite")
    return truth, prediction


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth, prediction = _pairs(y_true, y_pred)
    return float(np.mean((truth - prediction) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth, prediction = _pairs(y_true, y_pred)
    return float(np.mean(np.abs(truth - prediction)))


def smape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-12) -> float:
    """Symmetric MAPE in percent, with zero/zero terms defined as zero."""

    truth, prediction = _pairs(y_true, y_pred)
    denominator = np.abs(truth) + np.abs(prediction)
    terms = np.where(denominator > epsilon, 2.0 * np.abs(truth - prediction) / denominator, 0.0)
    return float(100.0 * np.mean(terms))


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
