from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from eval.temporal import training_ends


def conformal_p10_p90(
    residuals: np.ndarray,
    yhat: np.ndarray,
    min_resid: int = 20,
    horizon: int = 1,
    origin_dates=None,
    target_dates=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantiles of residuals whose targets are already observed at the origin."""
    n = len(yhat)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    r = np.asarray(residuals, dtype=float)
    if len(r) != n:
        raise ValueError("Residuals and predictions must have equal lengths")
    ends = training_ends(n, horizon, origin_dates, target_dates)
    for t in range(n):
        hist = r[:ends[t]]
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
    p_estavel = float(1.0 - p_alta - p_queda)
    return {"p_alta": p_alta, "p_estavel": p_estavel, "p_queda": p_queda}
