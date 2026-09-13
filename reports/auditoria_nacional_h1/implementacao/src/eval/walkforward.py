from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


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
) -> Dict[str, np.ndarray]:
    """Predict y[t+h] from X[t] after training on pairs up to origin t.

    For an online model the factory is called once; ``y[k]`` is assumed to be
    the already shifted h-step target for ``X[k]``.  It can therefore only be
    used to update the model at origin ``k + horizon``.

    Leak-safe protocol:
      at origin t we may update through pair ``t - horizon`` and then predict
      from ``X[t]``.  Targets with a later realization date are never used.
    """
    model = model_factory()
    n = len(y)
    yhat = np.full(n, np.nan)
    n_rules = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    for t in range(n):
        if t >= horizon:
            model.update(X[t - horizon], float(y[t - horizon]))
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
) -> Dict[str, np.ndarray]:
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    n = len(y)
    yhat = np.full(n, np.nan)
    last_model = None
    last_fit_end = -10**9
    extra = extra or {}
    for t in range(n_min_train, n):
        train_end = max(0, t - horizon + 1)
        if train_end == 0:
            continue
        if t - last_fit_end >= refit_every or last_model is None:
            last_model = fit_predict(X[:train_end], y[:train_end], X[t : t + 1], **extra)
            last_fit_end = t
            yhat[t] = last_model[0]
        else:
            yhat[t] = fit_predict(
                X[:train_end], y[:train_end], X[t : t + 1], model=last_model[1], **extra
            )[0]
    mask = ~np.isnan(yhat)
    return {"yhat": yhat, "mask": mask}
