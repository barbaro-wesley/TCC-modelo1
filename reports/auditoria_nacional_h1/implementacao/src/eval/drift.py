from __future__ import annotations

from typing import Dict

import numpy as np


def page_hinkley(x: np.ndarray, delta: float = 0.005, lam: float = 0.05) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    mean = 0.0
    m = 0.0
    flags = np.zeros(len(x), dtype=int)
    path = np.zeros(len(x), dtype=float)
    n = 0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            path[i] = m
            continue
        n += 1
        mean += (v - mean) / n
        m = min(0.0, m + (v - mean - delta))
        path[i] = m
        if abs(m) > lam:
            flags[i] = 1
            m = 0.0
            mean = v
            n = 1
    return {"flags": flags, "path": path}


def rolling_rmse(y: np.ndarray, yhat: np.ndarray, window: int = 12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    e2 = (y - yhat) ** 2
    out = np.full(len(y), np.nan)
    for t in range(window - 1, len(y)):
        sl = e2[t - window + 1 : t + 1]
        sl = sl[np.isfinite(sl)]
        if len(sl):
            out[t] = float(np.sqrt(np.mean(sl)))
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, 1)
    a = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((a - e) * np.log(a / e)))
