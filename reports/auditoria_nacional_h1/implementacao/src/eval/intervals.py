from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def conformal_p10_p90(
    residuals: np.ndarray,
    yhat: np.ndarray,
    min_resid: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expanding residual quantiles applied to the current point forecast."""
    n = len(yhat)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    r = np.asarray(residuals, dtype=float)
    for t in range(n):
        hist = r[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_resid:
            continue
        q10, q90 = np.quantile(hist, [0.10, 0.90])
        lo[t] = yhat[t] + q10
        hi[t] = yhat[t] + q90
    return lo, hi


def direction_probs(
    residuals: np.ndarray,
    yhat: float,
    y_now: float,
    band: float = 0.02,
    min_resid: int = 20,
) -> Dict[str, float]:
    hist = np.asarray(residuals, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < min_resid:
        return {"p_alta": np.nan, "p_estavel": np.nan, "p_queda": np.nan}
    sims = yhat + hist
    delta = sims - y_now
    p_alta = float(np.mean(delta > band))
    p_queda = float(np.mean(delta < -band))
    p_estavel = float(np.mean(np.abs(delta) <= band))
    return {"p_alta": p_alta, "p_estavel": p_estavel, "p_queda": p_queda}
