from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _align(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _align(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def ndei(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _align(y, yhat)
    s = float(np.std(y, ddof=1))
    if s == 0:
        return float("nan")
    return rmse(y, yhat) / s


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _align(y, yhat)
    denom = np.abs(y) + np.abs(yhat)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y - yhat) / denom))


def directional_accuracy(y: np.ndarray, yhat: np.ndarray, y_prev: np.ndarray) -> float:
    y, yhat = _align(y, yhat)
    y_prev = np.asarray(y_prev, dtype=float).ravel()[: len(y)]
    dy = np.sign(y - y_prev)
    dhat = np.sign(yhat - y_prev)
    mask = dy != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(dy[mask] == dhat[mask]))


def pinball(y: np.ndarray, qhat: np.ndarray, tau: float) -> float:
    y, qhat = _align(y, qhat)
    u = y - qhat
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    n = min(len(y), len(lo), len(hi))
    return float(np.mean((y[:n] >= lo[:n]) & (y[:n] <= hi[:n])))


def summarize(y: np.ndarray, yhat: np.ndarray, y_prev: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "ndei": ndei(y, yhat),
        "smape": smape(y, yhat),
        "n": float(len(np.asarray(y).ravel())),
    }
    if y_prev is not None:
        out["dir_acc"] = directional_accuracy(y, yhat, y_prev)
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Dict[str, float]:
    """DM test on squared error; positive stat means model 1 is worse than model 2."""
    d = np.asarray(e1, dtype=float).ravel() ** 2 - np.asarray(e2, dtype=float).ravel() ** 2
    n = len(d)
    dbar = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    var = gamma0
    for lag in range(1, h):
        cov = float(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
        var += 2.0 * (1.0 - lag / h) * cov
    se = np.sqrt(max(var, 1e-18) / n)
    stat = dbar / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return {"dm_stat": float(stat), "pvalue": float(p), "mean_diff": dbar}


def _align(y, yhat):
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    n = min(len(y), len(yhat))
    return y[:n], yhat[:n]
