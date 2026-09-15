from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from eval.temporal import training_ends


def expanding_origin_indices(n: int, n_min_train: int, horizon: int) -> List[int]:
    """Origins t such that we predict y[t+horizon] using data up to t."""
    last = n - horizon - 1
    return list(range(n_min_train - 1, last + 1))


def walk_forward_online(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_min_train: int,
    horizon: int = 1,
    origin_dates=None,
    target_dates=None,
) -> Dict[str, np.ndarray]:
    """Predict shifted y[t] using only labels mature at the current origin.

    Supply dates for irregular/filtered panels. Without dates, rows must be
    equally spaced and label k becomes observable at k+horizon.
    """
    model = model_factory()
    n = len(y)
    yhat = np.full(n, np.nan)
    n_rules = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    ends = training_ends(n, horizon, origin_dates, target_dates)
    learned = 0
    for t in range(n):
        while learned < ends[t]:
            model.update(X[learned], float(y[learned]))
            learned += 1
        if t >= n_min_train:
            yhat[t] = model.predict_one(X[t]) if model.n_rules else np.nan
        n_rules[t] = getattr(model, "n_rules", np.nan)
        betas[t] = getattr(model, "beta", np.nan)
    mask = ~np.isnan(yhat)
    mask[: max(n_min_train, 1)] = False
    return {
        "yhat": yhat,
        "mask": mask,
        "n_rules": n_rules,
        "beta": betas,
        "model": model,
    }


def walk_forward_batch(
    fit_predict: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_min_train: int,
    refit_every: int = 4,
    extra: Optional[dict] = None,
    horizon: int = 1,
    origin_dates=None,
    target_dates=None,
) -> Dict[str, np.ndarray]:
    n = len(y)
    yhat = np.full(n, np.nan)
    last_model = None
    last_fit_end = -10**9
    extra = extra or {}
    ends = training_ends(n, horizon, origin_dates, target_dates)
    for t in range(n_min_train, n):
        te = ends[t]
        if te == 0:
            continue
        if t - last_fit_end >= refit_every or last_model is None:
            last_model = fit_predict(X[:te], y[:te], X[t : t + 1], **extra)
            last_fit_end = t
            yhat[t] = last_model[0]
        else:
            yhat[t] = fit_predict(X[:te], y[:te], X[t : t + 1], model=last_model[1], **extra)[0]
    mask = ~np.isnan(yhat)
    return {"yhat": yhat, "mask": mask}
