from __future__ import annotations

from typing import Dict, List

import numpy as np


def permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    n_repeats: int = 5,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    rng = rng or np.random.default_rng(0)
    base = _mse(y, predict_fn(X))
    out = {}
    for j, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(_mse(y, predict_fn(Xp)) - base)
        out[name] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _mse(y, yhat) -> float:
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    n = min(len(y), len(yhat))
    return float(np.mean((y[:n] - yhat[:n]) ** 2))
